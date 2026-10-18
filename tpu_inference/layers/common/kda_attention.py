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


def _kda_chunk_step(S, q, k, v, g, beta):
    """One chunk of the chunked KDA form for all heads.

    S [H, dk, dv] f32 (the state before the chunk); q, k [H, c, dk] (q already
    scaled); v [H, c, dv]; g [H, c, dk] per-token log decays; beta [H, c].
    Returns (S after the chunk, out [H, c, dv]). Padded positions must have
    beta = 0 and g = 0 so they neither update nor decay the state.
    """
    c = q.shape[1]
    gc = jnp.cumsum(g, axis=1)  # [H, c, dk], <= 0, non-increasing
    # decay[h, i, j, d] = exp(gc_i - gc_j) for i >= j; the upper triangle is
    # never used and is clamped so it cannot overflow.
    decay = jnp.exp(jnp.minimum(gc[:, :, None, :] - gc[:, None, :, :], 0.0))
    strict = jnp.tril(jnp.ones((c, c), bool), k=-1)
    incl = jnp.tril(jnp.ones((c, c), bool))
    k_beta = k * beta[..., None]
    a = jnp.einsum("hid,hjd,hijd->hij", k_beta, k, decay)
    a = jnp.where(strict, a, 0.0)
    eye = jnp.eye(c, dtype=jnp.float32)
    # T = (I + A)^-1 with A strictly lower (the WY representation).
    t = jax.vmap(lambda m: jax.scipy.linalg.solve_triangular(
        m, eye, lower=True, unit_diagonal=True))(eye + a)
    u = t @ (v * beta[..., None])  # [H, c, dv]
    w = t @ (k_beta * jnp.exp(gc))  # [H, c, dk]
    attn = jnp.where(incl, jnp.einsum("hid,hjd,hijd->hij", q, k, decay), 0.0)
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
                       chunk_size: int = 64):
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

    init = recurrent_state[state_indices].astype(jnp.float32)
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


def ragged_kda_decode(q, k, v, g, beta, recurrent_state, state_indices,
                      distribution):
    """One recurrent KDA step per request; token i belongs to request i.

    Shapes as in `ragged_kda_chunked` (T <= max_reqs here); tokens at
    i >= distribution[2] are padding and leave the state untouched.
    """
    T = q.shape[0]
    dk = k.shape[-1]
    q = _l2norm(q) * (dk**-0.5)
    k = _l2norm(k)
    slots = state_indices[:T]
    S = recurrent_state[slots].astype(jnp.float32)  # [T, H, dk, dv]
    S = S * jnp.exp(g.astype(jnp.float32))[..., None]
    kv = jnp.einsum("thkv,thk->thv", S, k)
    delta = (v.astype(jnp.float32) - kv) * beta.astype(jnp.float32)[..., None]
    S = S + k[..., None] * delta[..., None, :]
    out = jnp.einsum("thkv,thk->thv", S, q)
    valid = jnp.arange(T) < distribution[2]
    new = jnp.where(valid[:, None, None, None],
                    S.astype(recurrent_state.dtype), recurrent_state[slots])
    return recurrent_state.at[slots].set(new), out


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
               chunk_size: int = 64):
    """Dispatches to the decode-only path when every request has one token."""
    T, max_reqs = q.shape[0], state_indices.shape[0]
    chunked = functools.partial(ragged_kda_chunked, chunk_size=chunk_size)
    if T > max_reqs:  # cannot be decode-only
        return chunked(q, k, v, g, beta, recurrent_state, query_start_loc,
                       state_indices, distribution, has_initial_state)
    return lax.cond(
        distribution[0] == distribution[2],
        lambda: ragged_kda_decode(q, k, v, g, beta, recurrent_state,
                                  state_indices, distribution),
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
                      chunk_size: int = 64):
    """Single-device KDA layer core: causal conv + SiLU + ragged KDA.

    qkv [T, 3C] (the q | k | v projections before the conv, C = num_heads *
    head_dim); g [T, H, d] f32 log decays; beta [T, H] f32; conv_state
    [num_slots, K-1, 3C]; recurrent_state [num_slots, H, d, d];
    conv_weight [3C, 1, K]. Returns ((new conv_state, new recurrent_state),
    core [T, H, d] f32 (before the gated output norm)).
    """
    has_init = has_initial_state_from_lens(query_start_loc, seq_lens)
    x, new_conv = ragged_conv1d(qkv,
                                conv_state,
                                conv_weight,
                                None,
                                query_start_loc,
                                state_indices,
                                distribution,
                                has_init,
                                kernel_size=kernel_size)
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
                                 chunk_size=chunk_size)
    return (new_conv, new_state), core


def _kda_heads_local(q, k, v, g, beta, conv_state, recurrent_state, conv_q,
                     conv_k, conv_v, query_start_loc, state_indices,
                     distribution, seq_lens, *, head_dim: int,
                     kernel_size: int, chunk_size: int, kda_kernel):
    """Per-shard body of `run_jax_kda_attention`: lays the local heads' q | k |
    v channels side by side (the conv state's layout) and runs the core."""
    T, H = q.shape[0], q.shape[1]
    qkv = jnp.concatenate([x.reshape(T, H * head_dim) for x in (q, k, v)], -1)
    conv_w = jnp.concatenate([
        w.reshape(H * head_dim, kernel_size) for w in (conv_q, conv_k, conv_v)
    ], 0)[:, None, :]
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
                      chunk_size=chunk_size)


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
    """
    tp = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    H_total, d = q.shape[1], q.shape[2]
    assert H_total % tp == 0, (H_total, tp)
    D, H = ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD
    local = functools.partial(_kda_heads_local,
                              head_dim=d,
                              kernel_size=conv_q.shape[-1],
                              chunk_size=chunk_size,
                              kda_kernel=kda_kernel or kda_conv_and_core)
    heads = P(D, H, None)
    conv_w = P(H, None, None)
    fn = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(heads, heads, heads, heads, P(D, H), P(D, None, H),
                  P(D, H, None,
                    None), conv_w, conv_w, conv_w, P(D), P(D), P(D), P(D)),
        out_specs=((P(D, None, H), P(D, H, None, None)), heads),
        check_vma=False,
    )
    return fn(q, k, v, g, beta, conv_state, recurrent_state, conv_q, conv_k,
              conv_v, query_start_loc, state_indices, distribution, seq_lens)
