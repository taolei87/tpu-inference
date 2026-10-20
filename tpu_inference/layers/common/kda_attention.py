# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KDA (Kimi Delta Attention) over ragged batches, in JAX.

KDA is a gated delta rule with a per-channel decay: for every head the state
S [d_k, d_v] evolves as

    S <- diag(exp(g_t)) S ;  S <- S + k_t (beta_t (v_t - k_t^T S))^T ;  o_t = q_t^T S

with g_t [d_k] <= 0 (GDN's decay is one scalar per head). q and k are
l2-normalized and q is scaled by d_k^-0.5 inside the op.

Batches follow the GDN ops' conventions (`kernels/gdn/reference`): tokens are
packed [num_tokens, ...]; `query_start_loc` [max_reqs + 1], `distribution`
(decode_end, prefill_end, num_valid_seqs) and `state_indices` [max_reqs] (slot
0 is the null slot) describe the requests; `has_initial_state` [max_reqs] is
False for requests whose slot must be treated as zero (new prefills).
With prefix caching ("align" mode) a step reads a request's state from one
slot and writes it to another (`align_state_slots`): `read_indices` [max_reqs]
(None: the same as `state_indices`, which are then the write slots).

This is the correctness implementation: mixed batches pad every sequence to
whole chunks and run the chunked form one chunk at a time; decode-only batches
run one recurrent step per request.
"""
import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.gdn.reference.ragged_gated_delta_rule_chunked import \
    pack_inputs_single_stream
from tpu_inference.layers.common.ragged_conv1d_jax import ragged_conv1d
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product


def _l2norm(x, eps=1e-6):
    x = x.astype(jnp.float32)
    return x * lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def has_initial_state_from_lens(query_start_loc: jax.Array,
                                seq_lens: jax.Array) -> jax.Array:
    """True for requests that already computed tokens before this step."""
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    return (seq_lens - query_lens) > 0


def align_state_slots(block_tables: jax.Array, query_start_loc: jax.Array,
                      seq_lens: jax.Array, block_size: int, num_slots: int):
    """Prefix caching ("align" mode): the state slots a step reads and
    writes, from the requests' state block tables.

    Block j of a request holds its state after (j + 1) * block_size tokens
    (the scheduler ends prefill chunks on block boundaries), so a step reads
    the block of the last computed token and writes the block of its own last
    token; they differ when the step crosses into a new block (or resumes
    from a cached prefix). block_tables [max_reqs * blocks_per_req] or
    [max_reqs, blocks_per_req]. Returns (read, write) [max_reqs], clipped to
    the slot range (padding rows point at the null block 0).
    """
    max_reqs = seq_lens.shape[0]
    bt = block_tables.reshape(max_reqs, -1)
    computed = seq_lens - (query_start_loc[1:] - query_start_loc[:-1])
    rows = jnp.arange(max_reqs)

    def slot(n):
        col = jnp.clip((n - 1) // block_size, 0, bt.shape[1] - 1)
        return jnp.clip(bt[rows, col], 0, num_slots - 1).astype(jnp.int32)

    return slot(jnp.maximum(computed, 1)), slot(jnp.maximum(seq_lens, 1))


def _decay_mix(x, y, gc, sub):
    """M[h, i, j] = sum_d x[h, i, d] y[h, j, d] exp(gc[h, i, d] - gc[h, j, d])
    for j <= i (zero above the diagonal); gc [H, c, d] non-increasing in i.

    Sub-chunks of `sub` positions: across sub-chunks the decay is factored
    through the start s of row i's sub-chunk, exp(gc_i - gc_s) and
    exp(gc_s - gc_j), both <= 1 (no overflow at any decay rate) so the block
    is a matmul; within a sub-chunk it is formed exactly, on [sub, sub, d].
    """
    H, c, d = x.shape
    n = c // sub
    xs, ys, gs = (a.reshape(H, n, sub, d) for a in (x, y, gc))
    ref = gs[:, :, :1, :]  # [H, n, 1, d]: gc at each sub-chunk's start
    left = xs * jnp.exp(gs - ref)  # [H, n, sub, d]
    right = y[:, None] * jnp.exp(jnp.minimum(ref - gc[:, None], 0.0))
    off = jnp.einsum("hnid,hnjd->hnij", left, right).reshape(H, c, c)
    blk = jnp.arange(c) // sub
    off = jnp.where(blk[:, None] > blk[None, :], off, 0.0)
    dd = jnp.exp(jnp.minimum(gs[:, :, :, None] - gs[:, :, None], 0.0))
    diag = jnp.einsum("hnid,hnjd,hnijd->hnij", xs, ys, dd)  # [H, n, sub, sub]
    diag = jnp.einsum("hnij,nm->hnimj", diag,
                      jnp.eye(n, dtype=diag.dtype)).reshape(H, c, c)
    return off + jnp.where(jnp.tril(jnp.ones((c, c), bool)), diag, 0.0)


def _unit_lower_inverse(a, blk=8):
    """(I + a)^-1 for strictly lower a [H, c, c], by block forward
    substitution: the inverses D_i of the blk x blk diagonal blocks, then the
    blocks below them by matmuls, X[i, :i] = -D_i (L[i, :i] X[:i, :i])."""
    H, c, _ = a.shape
    n = c // blk
    eye = jnp.eye(blk, dtype=a.dtype)
    lb = a.reshape(H, n, blk, n, blk)
    diag = jnp.stack([lb[:, i, :, i] for i in range(n)], 1)  # [H, n, b, b]
    # Diagonal blocks without a triangular solve (a TPU custom call with a
    # large fixed cost): for strictly lower N of size b, N^b = 0 and
    # (I - N)^-1 = (I + N)(I + N^2)(I + N^4)... exactly; at b = 8 the powers
    # grow at most by C(7, 3) = 35, harmless in f32.
    nn = -diag
    d_inv = eye + nn
    p = nn
    for _ in range(max(1, (blk - 1).bit_length()) - 1):
        p = p @ p
        d_inv = d_inv + d_inv @ p
    rows = [
        jnp.concatenate([d_inv[:, 0]] +
                        [jnp.zeros_like(eye)[None].repeat(H, 0)] * (n - 1), -1)
    ]  # block row 0: [H, b, c]
    for i in range(1, n):
        x_prev = jnp.concatenate(rows, 1)[:, :, :i * blk]  # X[:i, :i]
        l_row = a[:, i * blk:(i + 1) * blk, :i * blk]  # L[i, :i]
        below = -d_inv[:, i] @ (l_row @ x_prev)  # [H, b, i * b]
        rows.append(
            jnp.concatenate([below, d_inv[:, i]] +
                            [jnp.zeros(
                                (H, blk, (n - 1 - i) * blk), a.dtype)], -1))
    return jnp.concatenate(rows, 1)


def _kda_chunk_step(S, q, k, v, g, beta, sub: int = 16):
    """One chunk of the chunked KDA form for all heads.

    S [H, dk, dv] f32 (the state before the chunk); q, k [H, c, dk] (q already
    scaled); v [H, c, dv]; g [H, c, dk] per-token log decays; beta [H, c].
    Returns (S after the chunk, out [H, c, dv]). Padded positions must have
    beta = 0 and g = 0 so they neither update nor decay the state.
    """
    c = q.shape[1]
    gc = jnp.cumsum(g, axis=1)  # [H, c, dk], <= 0, non-increasing
    sub = sub if c % sub == 0 else c
    strict = jnp.tril(jnp.ones((c, c), bool), k=-1)
    k_beta = k * beta[..., None]
    # a[i, j] = sum_d k_beta_i k_j exp(gc_i - gc_j), strictly lower
    a = jnp.where(strict, _decay_mix(k_beta, k, gc, sub), 0.0)
    # T = (I + A)^-1 with A strictly lower (the WY representation).
    t = _unit_lower_inverse(a, 8 if c % 8 == 0 else c)
    u = t @ (v * beta[..., None])  # [H, c, dv]
    w = t @ (k_beta * jnp.exp(gc))  # [H, c, dk]
    attn = _decay_mix(q, k, gc, sub)  # lower triangle incl. the diagonal
    v_new = u - w @ S
    out = (q * jnp.exp(gc)) @ S + attn @ v_new
    g_last = gc[:, -1, :]  # [H, dk]
    S = S * jnp.exp(g_last)[:, :, None] + jnp.swapaxes(
        k * jnp.exp(g_last[:, None, :] - gc), -1, -2) @ v_new
    return S, out


def ragged_kda_chunked(q,
                       k,
                       v,
                       g,
                       beta,
                       recurrent_state,
                       query_start_loc,
                       state_indices,
                       distribution,
                       has_initial_state,
                       *,
                       chunk_size: int = 64,
                       read_indices=None):
    """Chunked KDA over a mixed (prefill + decode) ragged batch.

    q, k [T, H, dk]; v [T, H, dv]; g [T, H, dk] f32; beta [T, H] f32;
    recurrent_state [num_slots, H, dk, dv]. Returns (new recurrent_state,
    out [T, H, dv] f32).
    """
    dk = k.shape[-1]
    q = _l2norm(q) * (dk**-0.5)
    k = _l2norm(k)
    pq, pk, pv, pg, pbeta, reset, new_qsl, idx = pack_inputs_single_stream(
        q,
        k,
        v.astype(jnp.float32),
        g,
        beta,
        query_start_loc,
        distribution,
        chunk_size,
        compute_dtype=jnp.float32)
    n_chunks = pq.shape[0] // chunk_size
    H, dv = v.shape[1], v.shape[2]

    def to_chunks(x):  # [n*c, H, ...] -> [n, H, c, ...]
        return jnp.swapaxes(x.reshape((n_chunks, chunk_size) + x.shape[1:]), 1,
                            2)

    read = state_indices if read_indices is None else read_indices
    init = recurrent_state[read].astype(jnp.float32)
    init = jnp.where(has_initial_state[:, None, None, None], init, 0.0)
    start_chunks = new_qsl[:-1] // chunk_size
    # Chunks that start a sequence load its initial state. Invalid trailing
    # sequences have zero length and point past the packed stream; drop them.
    init_per_chunk = jnp.zeros((n_chunks, H, dk, dv), jnp.float32)
    init_per_chunk = init_per_chunk.at[start_chunks].set(init, mode="drop")

    def body(S, xs):
        qc, kc, vc, gc, bc, rst, s0 = xs
        S = jnp.where(rst, s0, S)
        S, out = _kda_chunk_step(S, qc, kc, vc, gc, bc)
        return S, (out, S)

    xs = (to_chunks(pq), to_chunks(pk), to_chunks(pv), to_chunks(pg),
          to_chunks(pbeta), reset, init_per_chunk)
    _, (out, s_after) = lax.scan(body, jnp.zeros((H, dk, dv), jnp.float32), xs)
    out = jnp.swapaxes(out, 1, 2).reshape(-1, H, dv)[idx]  # [T, H, dv]

    last_chunk = new_qsl[1:] // chunk_size - 1
    final = s_after[jnp.clip(last_chunk, 0, n_chunks - 1)]
    valid = jnp.arange(state_indices.shape[0]) < distribution[2]
    cur = recurrent_state[state_indices]
    new = jnp.where(valid[:, None, None, None],
                    final.astype(recurrent_state.dtype), cur)
    return recurrent_state.at[state_indices].set(new), out


def ragged_kda_decode(q,
                      k,
                      v,
                      g,
                      beta,
                      recurrent_state,
                      state_indices,
                      distribution,
                      read_indices=None):
    """One recurrent KDA step per request; token i belongs to request i.

    Shapes as in `ragged_kda_chunked` (T <= max_reqs: the caller passes the
    first max_reqs tokens of a padded decode step); tokens at
    i >= distribution[2] are padding and leave the state untouched.
    """
    T = q.shape[0]
    dk = k.shape[-1]
    q = _l2norm(q) * (dk**-0.5)
    k = _l2norm(k)
    slots = state_indices[:T]
    read = slots if read_indices is None else read_indices[:T]
    S = recurrent_state[read].astype(jnp.float32)  # [T, H, dk, dv]
    S = S * jnp.exp(g.astype(jnp.float32))[..., None]
    kv = jnp.einsum("thkv,thk->thv", S, k)
    delta = (v.astype(jnp.float32) - kv) * beta.astype(jnp.float32)[..., None]
    S = S + k[..., None] * delta[..., None, :]
    out = jnp.einsum("thkv,thk->thv", S, q)
    valid = jnp.arange(T) < distribution[2]
    new = jnp.where(valid[:, None, None, None],
                    S.astype(recurrent_state.dtype), recurrent_state[slots])
    return recurrent_state.at[slots].set(new), out


# ---------------------------------------------------------------------------
# Speculative decoding. Each request owns `group` = num_spec + 1 consecutive
# state slots (base = state_indices[r]). The state after the token at
# position p is kept in slot base + p % group, and a step starts from the
# state after its first token's predecessor, base + (p_first - 1) % group. A
# verify window has at most `group` tokens, so after any number of accepted
# drafts the next step's first position points at the state after the last
# accepted token: rejected tokens' states are never read and nothing has to
# be rolled back. group == 1 is the plain one-slot layout.


def spec_slots(positions, query_start_loc, state_indices, distribution,
               group: int):
    """(read slots [R], per-request window: token index [R, group] of the
    last <= group tokens of each request, valid mask [R, group], write slots
    [R, group])."""
    T, R = positions.shape[0], state_indices.shape[0]
    starts, lens = query_start_loc[:-1], jnp.diff(query_start_loc)
    valid = jnp.arange(R) < distribution[2]
    lens = jnp.where(valid, lens, 0)
    first = positions[jnp.clip(starts, 0, T - 1)]
    read = jnp.where(valid, state_indices + jnp.mod(first - 1, group),
                     state_indices)
    tail = jnp.minimum(lens, group)
    t = jnp.arange(group)[None]
    idx = jnp.clip((starts + lens - tail)[:, None] + t, 0, T - 1)
    ok = t < tail[:, None]
    write = state_indices[:, None] + jnp.mod(positions[idx], group)
    return read, idx, ok, write


def _decode_step(S, q, k, v, g, beta):
    """One recurrent KDA token per row: S [R, H, dk, dv] f32; q, k
    normalized. Returns (S, out [R, H, dv])."""
    S = S * jnp.exp(g.astype(jnp.float32))[..., None]
    kv = jnp.einsum("thkv,thk->thv", S, k)
    delta = (v.astype(jnp.float32) - kv) * beta.astype(jnp.float32)[..., None]
    S = S + k[..., None] * delta[..., None, :]
    return S, jnp.einsum("thkv,thk->thv", S, q)


def ragged_kda_spec(q, k, v, g, beta, recurrent_state, query_start_loc,
                    state_indices, distribution, has_initial_state, positions,
                    *, group: int, chunk_size: int):
    """`ragged_kda` with per-position state checkpoints (see above): the
    chunked pass runs with each request's last <= group tokens made no-ops
    (beta = 0, g = 0), which gives the state before them; those tokens then
    run recurrently, writing the state after each one."""
    T = q.shape[0]
    read, idx, ok, write = spec_slots(positions, query_start_loc,
                                      state_indices, distribution, group)
    in_window = jnp.zeros(T, bool).at[idx].max(ok)
    g0 = jnp.where(in_window[:, None, None], 0.0, g)
    b0 = jnp.where(in_window[:, None], 0.0, beta)
    state, out = ragged_kda_chunked(q,
                                    k,
                                    v,
                                    g0,
                                    b0,
                                    recurrent_state,
                                    query_start_loc,
                                    read,
                                    distribution,
                                    has_initial_state,
                                    chunk_size=chunk_size)
    R = min(state_indices.shape[0], T)  # rows that can hold a request
    S = state[read[:R]].astype(jnp.float32)  # before each request's window
    return _recurrent_windows(S, state, out, q, k, v, g, beta, idx[:R], ok[:R],
                              write[:R])


def _recurrent_windows(S, state, out, q, k, v, g, beta, idx, ok, write):
    """Runs each request's window tokens (idx [R, group], valid ok) from its
    state S [R, H, dk, dv] f32, writing every token's output into out and the
    state after it into its write slot."""
    T, dk = q.shape[0], k.shape[-1]
    qn, kn = _l2norm(q) * (dk**-0.5), _l2norm(k)
    oob = state.shape[0]
    for t in range(idx.shape[1]):
        i = idx[:, t]
        S_new, o = _decode_step(S, qn[i], kn[i], v[i], g[i], beta[i])
        m = ok[:, t]
        S = jnp.where(m[:, None, None, None], S_new, S)
        out = out.at[jnp.where(m, i, T)].set(o, mode="drop")
        state = state.at[jnp.where(m, write[:, t],
                                   oob)].set(S.astype(state.dtype),
                                             mode="drop")
    return state, out


def ragged_kda_windows(q, k, v, g, beta, recurrent_state, query_start_loc,
                       state_indices, distribution, has_initial_state,
                       positions, *, group: int):
    """A step where every request has at most `group` tokens (a speculative
    verify step): all tokens run recurrently from the read slots, without
    the chunked pass (which pads every request to a whole chunk)."""
    # every valid request has a token: at most T of the max_reqs rows matter
    R = min(state_indices.shape[0], q.shape[0])
    read, idx, ok, write = spec_slots(positions, query_start_loc[:R + 1],
                                      state_indices[:R], distribution, group)
    S = recurrent_state[read].astype(jnp.float32)
    S = jnp.where(has_initial_state[:R, None, None, None], S, 0.0)
    out = jnp.zeros(v.shape, jnp.float32)
    return _recurrent_windows(S, recurrent_state, out, q, k, v, g, beta, idx,
                              ok, write)


def ragged_kda_decode_spec(q, k, v, g, beta, recurrent_state, positions,
                           state_indices, distribution, *, group: int):
    """`ragged_kda_decode` (one token per request) with the slot group: read
    the state of position p - 1, write the state of p."""
    T, dk = q.shape[0], k.shape[-1]
    slots = state_indices[:T]
    valid = jnp.arange(T) < distribution[2]
    read = jnp.where(valid, slots + jnp.mod(positions - 1, group), slots)
    write = jnp.where(valid, slots + jnp.mod(positions, group),
                      recurrent_state.shape[0])
    S, out = _decode_step(recurrent_state[read].astype(jnp.float32),
                          _l2norm(q) * (dk**-0.5), _l2norm(k), v, g, beta)
    return recurrent_state.at[write].set(S.astype(recurrent_state.dtype),
                                         mode="drop"), out


def ragged_kda(q,
               k,
               v,
               g,
               beta,
               recurrent_state,
               query_start_loc,
               state_indices,
               distribution,
               has_initial_state,
               *,
               chunk_size: int = 64,
               positions=None,
               spec_group: int = 1,
               read_indices=None):
    """Dispatches to the decode-only path when every request has one token.
    spec_group > 1: the speculative-decoding slot groups (positions needed).
    read_indices: the slots to read from (prefix caching), not with
    spec_group > 1."""
    T, max_reqs = q.shape[0], state_indices.shape[0]
    if spec_group > 1:
        assert read_indices is None, "prefix caching with speculative decoding"
        args = (q, k, v, g, beta, recurrent_state, query_start_loc,
                state_indices, distribution, has_initial_state, positions)
        lens = jnp.where(
            jnp.arange(max_reqs) < distribution[2], jnp.diff(query_start_loc),
            0)

        def spec_or_windows():
            # a verify step (every request <= spec_group tokens) needs no
            # chunked pass
            return lax.cond(
                jnp.max(lens) <= spec_group,
                lambda: ragged_kda_windows(*args, group=spec_group),
                lambda: ragged_kda_spec(
                    *args, group=spec_group, chunk_size=chunk_size))

        R = min(T, max_reqs)

        def decode_spec():
            state, out = ragged_kda_decode_spec(q[:R],
                                                k[:R],
                                                v[:R],
                                                g[:R],
                                                beta[:R],
                                                recurrent_state,
                                                positions[:R],
                                                state_indices,
                                                distribution,
                                                group=spec_group)
            return state, jnp.pad(out, ((0, T - R), (0, 0), (0, 0)))

        return lax.cond(distribution[0] == distribution[2], decode_spec,
                        spec_or_windows)
    chunked = functools.partial(ragged_kda_chunked,
                                chunk_size=chunk_size,
                                read_indices=read_indices)
    # A decode-only step has at most max_reqs real tokens (one per request),
    # even when the step is padded past max_reqs (decode buckets start at 16
    # tokens): run the decode path on the first R tokens.
    R = min(T, max_reqs)

    def decode():
        state, out = ragged_kda_decode(q[:R], k[:R], v[:R], g[:R], beta[:R],
                                       recurrent_state, state_indices,
                                       distribution, read_indices)
        return state, jnp.pad(out, ((0, T - R), (0, 0), (0, 0)))

    return lax.cond(
        distribution[0] == distribution[2],
        decode,
        lambda: chunked(q, k, v, g, beta, recurrent_state, query_start_loc,
                        state_indices, distribution, has_initial_state),
    )


def kda_conv_and_core(qkv,
                      g,
                      beta,
                      conv_state,
                      recurrent_state,
                      conv_weight,
                      query_start_loc,
                      state_indices,
                      distribution,
                      seq_lens,
                      *,
                      num_heads: int,
                      head_dim: int,
                      kernel_size: int,
                      chunk_size: int = 64,
                      positions=None,
                      spec_group: int = 1,
                      read_indices=None):
    """Single-device KDA layer core: causal conv + SiLU + ragged KDA.

    qkv [T, 3C] (the q | k | v projections before the conv, C = num_heads *
    head_dim); g [T, H, d] f32 log decays; beta [T, H] f32; conv_state
    [num_slots, K-1, 3C]; recurrent_state [num_slots, H, d, d];
    conv_weight [3C, 1, K]. Returns ((new conv_state, new recurrent_state),
    core [T, H, d] f32 (before the gated output norm)).
    """
    has_init = has_initial_state_from_lens(query_start_loc, seq_lens)
    # The conv state holds raw inputs in its own (model) dtype.
    qkv = qkv.astype(conv_state.dtype)
    conv_slots = state_indices
    if read_indices is not None:
        # The conv state is small: copy each request's state to its write
        # slot first (the read slot itself, or a fresh block of its own).
        conv_state = conv_state.at[state_indices].set(conv_state[read_indices])
    if spec_group > 1:
        conv_slots, idx, ok, write = spec_slots(positions, query_start_loc,
                                                state_indices, distribution,
                                                spec_group)
        prev = jnp.where(has_init[:, None, None], conv_state[conv_slots], 0)
    x, new_conv = ragged_conv1d(qkv,
                                conv_state,
                                conv_weight,
                                None,
                                query_start_loc,
                                conv_slots,
                                distribution,
                                has_init,
                                kernel_size=kernel_size)
    if spec_group > 1:
        R = min(state_indices.shape[0], qkv.shape[0])
        new_conv = _write_conv_checkpoints(new_conv, qkv, prev[:R],
                                           query_start_loc[:R + 1], idx[:R],
                                           ok[:R], write[:R])
    x = jax.nn.silu(x.astype(jnp.float32))
    T, C = x.shape[0], num_heads * head_dim
    q, k, v = (x[:, i * C:(i + 1) * C].reshape(T, num_heads, head_dim)
               for i in range(3))
    new_state, core = ragged_kda(q,
                                 k,
                                 v,
                                 g,
                                 beta,
                                 recurrent_state,
                                 query_start_loc,
                                 state_indices,
                                 distribution,
                                 has_init,
                                 chunk_size=chunk_size,
                                 positions=positions,
                                 spec_group=spec_group,
                                 read_indices=read_indices)
    return (new_conv, new_state), core


def _write_conv_checkpoints(conv_state, x, prev, query_start_loc, idx, ok,
                            write):
    """The conv state after each window token (its K-1 latest inputs, the
    earlier ones from the request's previous state `prev` [R, K-1, C]) into
    its write slot."""
    K1 = conv_state.shape[1]
    starts = query_start_loc[:-1]
    for t in range(idx.shape[1]):
        i = idx[:, t]  # [R]
        src = i[:, None] - (K1 - 1) + jnp.arange(K1)[None]  # [R, K-1]
        mine = src >= starts[:, None]
        rows = jnp.where(
            mine[..., None], x[jnp.clip(src, 0, x.shape[0] - 1)],
            prev[jnp.arange(prev.shape[0])[:, None],
                 jnp.clip(src - starts[:, None] + K1, 0, K1 - 1)])
        dst = jnp.where(ok[:, t], write[:, t], conv_state.shape[0])
        conv_state = conv_state.at[dst].set(rows.astype(conv_state.dtype),
                                            mode="drop")
    return conv_state


def _kda_heads_local(q, k, v, g, beta, conv_state, recurrent_state, conv_q,
                     conv_k, conv_v, query_start_loc, state_indices,
                     distribution, seq_lens, positions, *, head_dim: int,
                     kernel_size: int, chunk_size: int, kda_kernel,
                     spec_group: int, align_block_size: Optional[int]):
    """Per-shard body of `run_jax_kda_attention`: lays the local heads' q | k |
    v channels side by side (the conv state's layout) and runs the core.
    With align_block_size, state_indices is the (rank-local) state block
    table."""
    T, H = q.shape[0], q.shape[1]
    kw = {} if spec_group == 1 else dict(positions=positions,
                                         spec_group=spec_group)
    if align_block_size is not None:
        kw["read_indices"], state_indices = align_state_slots(
            state_indices, query_start_loc, seq_lens, align_block_size,
            recurrent_state.shape[0])
    qkv = jnp.concatenate([x.reshape(T, H * head_dim) for x in (q, k, v)], -1)
    # (weights keep the checkpoint's dtype; the conv runs in the state's)
    conv_w = jnp.concatenate([
        w.reshape(H * head_dim, kernel_size) for w in (conv_q, conv_k, conv_v)
    ], 0)[:, None, :].astype(conv_state.dtype)
    return kda_kernel(qkv,
                      g,
                      beta,
                      conv_state,
                      recurrent_state,
                      conv_w,
                      query_start_loc,
                      state_indices,
                      distribution,
                      seq_lens,
                      num_heads=H,
                      head_dim=head_dim,
                      kernel_size=kernel_size,
                      chunk_size=chunk_size,
                      **kw)


def run_jax_kda_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_q: jax.Array,
    conv_k: jax.Array,
    conv_v: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    chunk_size: int = 64,
    kda_kernel: Optional[callable] = None,
    positions: Optional[jax.Array] = None,
    spec_group: int = 1,
    block_tables: Optional[jax.Array] = None,
    align_block_size: Optional[int] = None,
):
    """KDA layer core under the mesh: heads are sharded on ATTN_HEAD, requests
    on ATTN_DATA (like `run_jax_gdn_attention`).

    q, k, v [T, H, d] are the projections before the conv; g [T, H, d] f32
    log decays; beta [T, H] f32; conv_q / conv_k / conv_v [H, d, K] the
    depthwise conv weights. conv_state [num_slots, K-1, 3C] holds, per shard,
    the local heads' q | k | v channels side by side (for one shard this is
    the plain q | k | v layout); recurrent_state [num_slots, H, d, d].
    Returns ((new conv_state, new recurrent_state), core [T, H, d] f32, before
    the gated output norm).

    spec_group = num_speculative_tokens + 1 > 1 keeps per-position state
    checkpoints in each request's slot group (`spec_slots`); positions [T]
    are then required.

    Prefix caching ("align" mode): align_block_size = the mamba block size,
    block_tables the KDA group's block tables (state_indices is then unused);
    each step reads and writes the slots `align_state_slots` picks.
    """
    if positions is None:
        assert spec_group == 1, "speculative decoding needs positions"
        positions = jnp.zeros(q.shape[0], jnp.int32)
    tp = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    H_total, d = q.shape[1], q.shape[2]
    assert H_total % tp == 0, (H_total, tp)
    D, H = ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD
    local = functools.partial(_kda_heads_local,
                              head_dim=d,
                              kernel_size=conv_q.shape[-1],
                              chunk_size=chunk_size,
                              kda_kernel=kda_kernel or kda_conv_and_core,
                              spec_group=spec_group,
                              align_block_size=align_block_size)
    if align_block_size is not None:
        assert spec_group == 1, "prefix caching with speculative decoding"
        state_indices = block_tables
    heads = P(D, H, None)
    conv_w = P(H, None, None)
    fn = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(heads, heads, heads, heads, P(D,
                                                H), P(D, None,
                                                      H), P(D, H, None, None),
                  conv_w, conv_w, conv_w, P(D), P(D), P(D), P(D), P(D)),
        out_specs=((P(D, None, H), P(D, H, None, None)), heads),
        check_vma=False,
    )
    return fn(q, k, v, g, beta, conv_state, recurrent_state, conv_q, conv_k,
              conv_v, query_start_loc, state_indices, distribution, seq_lens,
              positions)
