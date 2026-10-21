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
"""Ragged KDA (GLM-5.3-Flash linear attention) vs the per-sequence reference
(`tests/models/jax/glm5_next_ref/ref_model.py`)."""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import functools

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from tests.models.jax.glm5_next_ref import ref_model as R  # noqa: E402
from tpu_inference.layers.common import kda_attention as K  # noqa: E402
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES  # noqa: E402

H, D_HEAD, KS = 4, 16, 4


@pytest.fixture(autouse=True)
def _highest_precision():
    with jax.default_matmul_precision("highest"):
        yield


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def _batch(rng, lens, has_state, max_reqs, num_slots, pad_tokens=5):
    """A ragged batch: per-request inputs, slots, initial states and the
    packed arrays the op takes. Requests with has_state get a random initial
    (conv, recurrent) state; the others' slots hold garbage that must be
    ignored."""
    T_valid = sum(lens)
    T = T_valid + pad_tokens
    C = H * D_HEAD
    f = np.float32
    qkv = rng.standard_normal((T, 3 * C)).astype(f)
    g = -5.0 * (1 / (1 + np.exp(-rng.standard_normal((T, H, D_HEAD)))))
    g = g.astype(f)  # the real model's gate_lower_bound form, in [-5, 0]
    beta = (1 / (1 + np.exp(-rng.standard_normal((T, H))))).astype(f)
    conv_w = (0.5 * rng.standard_normal((3 * C, 1, KS))).astype(f)
    slots = rng.permutation(np.arange(1,
                                      num_slots))[:max_reqs].astype(np.int32)
    conv_state = rng.standard_normal((num_slots, KS - 1, 3 * C)).astype(f)
    rec_state = (0.3 * rng.standard_normal(
        (num_slots, H, D_HEAD, D_HEAD))).astype(f)
    qsl = np.full(max_reqs + 1, T_valid, np.int32)
    qsl[:len(lens) + 1] = np.concatenate([[0], np.cumsum(lens)])
    n_dec = sum(1 for n, s in zip(lens, has_state) if n == 1 and s)
    dist = np.array([n_dec, len(lens), len(lens)], np.int32)
    seq_lens = np.zeros(max_reqs, np.int32)
    for i, (n, s) in enumerate(zip(lens, has_state)):
        seq_lens[i] = n + (7 if s else 0)
    return dict(qkv=qkv,
                g=g,
                beta=beta,
                conv_w=conv_w,
                slots=slots,
                conv_state=conv_state,
                rec_state=rec_state,
                qsl=qsl,
                dist=dist,
                seq_lens=seq_lens,
                lens=lens,
                has_state=has_state)


def _reference(b):
    """Per request: the reference conv + SiLU + chunked KDA from its initial
    state (zeros for new requests) -> (out [T_valid, H, d], final conv /
    recurrent state per request)."""
    C = H * D_HEAD
    outs, convs, recs = [], [], []
    for i, (n, s) in enumerate(zip(b["lens"], b["has_state"])):
        a, e = b["qsl"][i], b["qsl"][i + 1]
        slot = b["slots"][i]
        conv0 = b["conv_state"][slot][None] if s else None
        x, conv1 = R.causal_conv1d(
            jnp.asarray(b["qkv"][a:e][None]), jnp.asarray(b["conv_w"][:,
                                                                      0, :]),
            None if conv0 is None else jnp.asarray(conv0))
        x = R.silu(x)
        q, k, v = (x[..., j * C:(j + 1) * C].reshape(1, n, H, D_HEAD)
                   for j in range(3))
        s0 = (jnp.asarray(b["rec_state"][slot][None]) if s else jnp.zeros(
            (1, H, D_HEAD, D_HEAD)))
        o, s1 = R.kda_chunked(q, k, v, jnp.asarray(b["g"][a:e][None]),
                              jnp.asarray(b["beta"][a:e][None]), s0)
        outs.append(np.asarray(o[0]))
        convs.append(np.asarray(conv1[0]))
        recs.append(np.asarray(s1[0]))
    return np.concatenate(outs), convs, recs


def _check(b, new_conv, new_rec, core):
    ref_out, ref_conv, ref_rec = _reference(b)
    T_valid = sum(b["lens"])
    assert _rel(core[:T_valid], ref_out) < 1e-4
    touched = set()
    for i in range(len(b["lens"])):
        slot = int(b["slots"][i])
        touched.add(slot)
        assert _rel(new_conv[slot], ref_conv[i]) < 1e-5, i
        assert _rel(new_rec[slot], ref_rec[i]) < 1e-4, i
    for slot in range(b["rec_state"].shape[0]):  # other slots untouched
        if slot not in touched:
            np.testing.assert_array_equal(new_rec[slot], b["rec_state"][slot])
            np.testing.assert_array_equal(new_conv[slot],
                                          b["conv_state"][slot])


def _run_local(b, chunk_size=16):
    (conv, rec), core = jax.jit(
        K.kda_conv_and_core,
        static_argnames=("num_heads", "head_dim", "kernel_size",
                         "chunk_size"))(jnp.asarray(b["qkv"]),
                                        jnp.asarray(b["g"]),
                                        jnp.asarray(b["beta"]),
                                        jnp.asarray(b["conv_state"]),
                                        jnp.asarray(b["rec_state"]),
                                        jnp.asarray(b["conv_w"]),
                                        jnp.asarray(b["qsl"]),
                                        jnp.asarray(b["slots"]),
                                        jnp.asarray(b["dist"]),
                                        jnp.asarray(b["seq_lens"]),
                                        num_heads=H,
                                        head_dim=D_HEAD,
                                        kernel_size=KS,
                                        chunk_size=chunk_size)
    return np.asarray(conv), np.asarray(rec), np.asarray(core)


def test_mixed_batch_matches_reference():
    """Decodes with state, a new prefill, a chunked-prefill continuation
    crossing chunk boundaries, a long new prefill, and padding requests."""
    rng = np.random.default_rng(0)
    b = _batch(rng,
               lens=[1, 1, 37, 40, 75],
               has_state=[True, True, False, True, False],
               max_reqs=8,
               num_slots=12)
    _check(b, *_run_local(b))


def test_decode_only_batch_matches_reference():
    rng = np.random.default_rng(1)
    b = _batch(rng,
               lens=[1, 1, 1, 1],
               has_state=[True] * 4,
               max_reqs=8,
               num_slots=10,
               pad_tokens=4)
    _check(b, *_run_local(b))


@pytest.mark.parametrize("chunk_size", [8, 64])
def test_chunk_size_invariance(chunk_size):
    rng = np.random.default_rng(2)
    b = _batch(rng,
               lens=[1, 70, 9],
               has_state=[True, True, False],
               max_reqs=4,
               num_slots=6)
    _check(b, *_run_local(b, chunk_size=chunk_size))


def test_continuation_equals_one_shot():
    """Prefill in two scheduler steps (conv + recurrent state carried through
    the slot) == the same sequence in one step."""
    rng = np.random.default_rng(3)
    b = _batch(rng,
               lens=[50],
               has_state=[False],
               max_reqs=2,
               num_slots=3,
               pad_tokens=0)
    _, rec_full, core_full = _run_local(b)
    first = dict(b,
                 qkv=b["qkv"][:23],
                 g=b["g"][:23],
                 beta=b["beta"][:23],
                 qsl=np.array([0, 23, 23], np.int32),
                 seq_lens=np.array([23, 0], np.int32))
    conv1, rec1, core1 = _run_local(first)
    second = dict(b,
                  qkv=b["qkv"][23:],
                  g=b["g"][23:],
                  beta=b["beta"][23:],
                  qsl=np.array([0, 27, 27], np.int32),
                  seq_lens=np.array([50, 0], np.int32),
                  conv_state=conv1,
                  rec_state=rec1)
    _, rec2, core2 = _run_local(second)
    slot = b["slots"][0]
    assert _rel(np.concatenate([core1[:23], core2[:27]]), core_full) < 1e-4
    assert _rel(rec2[slot], rec_full[slot]) < 1e-4


def _per_head(b):
    """The batch's q / k / v projections [T, H, d] and conv weights
    [H, d, K]."""
    T, C = b["qkv"].shape[0], H * D_HEAD
    qkv = [
        b["qkv"][:, j * C:(j + 1) * C].reshape(T, H, D_HEAD) for j in range(3)
    ]
    conv = [
        b["conv_w"][j * C:(j + 1) * C, 0].reshape(H, D_HEAD, KS)
        for j in range(3)
    ]
    return qkv, conv


@pytest.mark.parametrize("tp", [1, 2])
def test_sharded_matches_local(tp):
    """Heads sharded over the model axis (8 virtual CPU devices). For tp > 1 the
    conv state holds each shard's q | k | v channels side by side."""
    if len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} devices")
    rng = np.random.default_rng(4)
    b = _batch(rng,
               lens=[1, 30, 12],
               has_state=[True, False, True],
               max_reqs=4,
               num_slots=6)
    devs = np.array(jax.devices()[:tp]).reshape(1, 1, 1, 1, tp, 1, 1)
    mesh = Mesh(devs, MESH_AXIS_NAMES)

    def interleave(x, axis):  # [.., q|k|v ..] -> per-shard [q_s|k_s|v_s]
        parts = np.split(x, 3, axis=axis)
        per = [np.split(p, tp, axis=axis) for p in parts]
        return np.concatenate([per[j][s] for s in range(tp) for j in range(3)],
                              axis=axis)

    (q, k, v), (cq, ck, cv) = _per_head(b)
    with mesh:
        (conv, rec), core = jax.jit(
            K.run_jax_kda_attention, static_argnames=("mesh", "chunk_size"))(
                *map(jnp.asarray,
                     (q, k, v, b["g"], b["beta"], interleave(
                         b["conv_state"], 2), b["rec_state"], cq, ck, cv,
                      b["slots"], b["qsl"], b["dist"], b["seq_lens"])),
                mesh=mesh,
                chunk_size=16)
    conv_l, rec_l, core_l = _run_local(b)
    assert _rel(core, core_l) < 1e-5
    assert _rel(rec, rec_l) < 1e-5
    assert _rel(conv, interleave(conv_l, 2)) < 1e-6


@pytest.mark.parametrize("sub", [8, 16])
@pytest.mark.parametrize("scale", [0.1, 5.0])
def test_decay_mix_matches_float64(sub, scale):
    """The sub-chunked decay product against a float64 reference, up to the
    gate's lower bound (-5 per token: -320 over a 64-token chunk, far past
    f32 exp range), with no inf / nan."""
    from tpu_inference.layers.common.kda_attention import _decay_mix
    rng = np.random.default_rng(0)
    H, c, d = 2, 64, 32
    x = rng.standard_normal((H, c, d))
    y = rng.standard_normal((H, c, d))
    g = -scale * rng.random((H, c, d))
    gc = np.cumsum(g, axis=1)
    ref = np.einsum("hid,hjd,hijd->hij", x, y,
                    np.exp(np.minimum(gc[:, :, None] - gc[:, None], 0)))
    ref = np.where(np.tril(np.ones((c, c), bool)), ref, 0)
    out = np.asarray(
        _decay_mix(jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32),
                   jnp.asarray(gc, jnp.float32), sub))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)


def _spec_step(stream,
               toks,
               conv_state,
               rec_state,
               slots,
               group,
               max_reqs=4,
               chunk_size=8):
    """One spec-decode scheduler step: toks [(request, first position, [token
    row indices into stream])]; positions come from first position."""
    T = sum(len(t) for _, _, t in toks)
    pad = 3
    qkv = np.concatenate([stream[r]["qkv"][t] for r, _, t in toks] +
                         [np.zeros((pad, stream[0]["qkv"].shape[1]))])
    g = np.concatenate([stream[r]["g"][t]
                        for r, _, t in toks] + [np.zeros((pad, H, D_HEAD))])
    beta = np.concatenate([stream[r]["beta"][t]
                           for r, _, t in toks] + [np.zeros((pad, H))])
    pos = np.concatenate([p0 + np.arange(len(t)) for _, p0, t in toks] +
                         [np.zeros(pad, int)]).astype(np.int32)
    qsl = np.full(max_reqs + 1, T, np.int32)
    qsl[:len(toks) + 1] = np.concatenate([[0],
                                          np.cumsum([len(t)
                                                     for *_, t in toks])])
    sl = np.zeros(max_reqs, np.int32)
    st = np.zeros(max_reqs, np.int32)
    for i, (r, p0, t) in enumerate(toks):
        sl[i] = p0 + len(t)
        st[i] = slots[r]
    n_dec = sum(1 for _, p0, t in toks if len(t) == 1 and p0 > 0)
    dist = np.array([n_dec, len(toks), len(toks)], np.int32)
    f = functools.partial(K.kda_conv_and_core,
                          num_heads=H,
                          head_dim=D_HEAD,
                          kernel_size=KS,
                          chunk_size=chunk_size,
                          spec_group=group)
    (conv, rec), core = jax.jit(f)(jnp.asarray(qkv, jnp.float32),
                                   jnp.asarray(g, jnp.float32),
                                   jnp.asarray(beta, jnp.float32),
                                   conv_state,
                                   rec_state,
                                   jnp.asarray(stream["w"]),
                                   jnp.asarray(qsl),
                                   jnp.asarray(st),
                                   jnp.asarray(dist),
                                   jnp.asarray(sl),
                                   positions=jnp.asarray(pos))
    out, t0 = {}, 0
    for r, p0, t in toks:
        out[r] = np.asarray(core[t0:t0 + len(t)])
        t0 += len(t)
    return conv, rec, out


@pytest.mark.parametrize("max_reqs", [4, 16])
def test_spec_decode_slot_groups_match_accepted_stream(max_reqs):
    """Speculative decoding with slot groups of 4 (3 drafts): a prefill, a
    verify window whose last two drafts are rejected (garbage inputs) next
    to a 1-token decode, a decode-only step, then full windows. Each
    request's accepted outputs and its state equal one plain run over the
    accepted tokens only; rejected drafts leave no trace."""
    rng = np.random.default_rng(9)
    group, n = 4, 24
    C = H * D_HEAD
    stream = {
        "w": (0.5 * rng.standard_normal((3 * C, 1, KS))).astype(np.float32)
    }
    for r in range(2):
        stream[r] = dict(
            qkv=rng.standard_normal((n + 8, 3 * C)).astype(np.float32),
            g=(-5.0 /
               (1 + np.exp(-rng.standard_normal((n + 8, H, D_HEAD))))).astype(
                   np.float32),
            beta=(1 / (1 + np.exp(-rng.standard_normal((n + 8, H))))).astype(
                np.float32))
    junk = list(range(n, n + 8))  # rows that stand for rejected drafts
    slots = {0: 1, 1: 6}  # group bases: slots 1-4 and 6-9
    conv = jnp.asarray(rng.standard_normal((12, KS - 1, 3 * C)), jnp.float32)
    rec = jnp.asarray(rng.standard_normal((12, H, D_HEAD, D_HEAD)),
                      jnp.float32)
    acc = {0: [], 1: []}
    steps = [
        [(0, 0, list(range(10))), (1, 0, list(range(7)))],
        # r0: window x10, x11 + 2 rejected drafts; r1: one token
        [(0, 10, [10, 11, junk[0], junk[1]]), (1, 7, [7])],
        [(0, 12, [12]), (1, 8, [8])],  # decode-only
        [(0, 13, [13, 14, 15, 16]), (1, 9, [9, 10, junk[2], junk[3]])],
        [(0, 17, [17, 18, 19]), (1, 11, list(range(11, 20)))],
    ]
    keep = [None, {0: 2, 1: 1}, None, {0: 4, 1: 2}, None]
    for step, k in zip(steps, keep):
        conv, rec, out = _spec_step(stream,
                                    step,
                                    conv,
                                    rec,
                                    slots,
                                    group,
                                    max_reqs=max_reqs)
        for r, p0, t in step:
            m = len(t) if k is None else k[r]
            acc[r].append(out[r][:m])
    for r, total in ((0, 20), (1, 20)):
        b = _batch(np.random.default_rng(0), [total], [False], 2, 3, 0)
        b.update(qkv=stream[r]["qkv"][:total],
                 g=stream[r]["g"][:total],
                 beta=stream[r]["beta"][:total],
                 conv_w=stream["w"])
        ref_out, ref_conv, ref_rec = _reference(b)
        assert _rel(np.concatenate(acc[r]), ref_out) < 1e-4, r
        last = slots[r] + (total - 1) % group
        assert _rel(np.asarray(rec[last]), ref_rec[0]) < 1e-4, r
        assert _rel(np.asarray(conv[last]), ref_conv[0]) < 1e-5, r


def test_decode_only_batch_padded_past_max_reqs():
    """A decode-only step padded past max_reqs (decode buckets start at 16
    tokens; max_num_seqs may be 8): the decode path on the first max_reqs
    tokens, same result."""
    rng = np.random.default_rng(5)
    b = _batch(rng,
               lens=[1, 1],
               has_state=[True, True],
               max_reqs=2,
               num_slots=5,
               pad_tokens=6)
    _check(b, *_run_local(b))


def _align_step(stream,
                toks,
                conv_state,
                rec_state,
                block_tables,
                mesh,
                block_size,
                max_reqs=4):
    """One prefix-caching ("align" mode) step through the sharded op: toks
    [(request, row of its block table, first position, last position + 1)],
    decodes first; the state slots come from the block tables."""
    C = H * D_HEAD
    pad = 3
    rows = [np.arange(a, e) for _, _, a, e in toks]
    T = sum(len(r) for r in rows)

    def cat(key, shape):
        return np.concatenate(
            [stream[r][key][x] for (r, *_), x in zip(toks, rows)] +
            [np.zeros((pad, ) + shape, np.float32)])

    qkv, g, beta = cat("qkv", (3 * C, )), cat("g",
                                              (H, D_HEAD)), cat("beta", (H, ))
    T_all = T + pad
    q, k, v = (qkv[:, j * C:(j + 1) * C].reshape(T_all, H, D_HEAD)
               for j in range(3))
    cq, ck, cv = (stream["w"][j * C:(j + 1) * C, 0].reshape(H, D_HEAD, KS)
                  for j in range(3))
    qsl = np.full(max_reqs + 1, T, np.int32)
    qsl[:len(toks) + 1] = np.concatenate([[0],
                                          np.cumsum([len(r) for r in rows])])
    sl = np.zeros(max_reqs, np.int32)
    bt = np.zeros((max_reqs, block_tables.shape[1]), np.int32)
    for i, (_, row, a, e) in enumerate(toks):
        sl[i], bt[i] = e, block_tables[row]
    n_dec = sum(1 for *_, a, e in toks if e - a == 1 and a > 0)
    dist = np.array([n_dec, len(toks), len(toks)], np.int32)
    with mesh:
        (conv, rec), core = jax.jit(
            K.run_jax_kda_attention,
            static_argnames=("mesh", "chunk_size", "align_block_size"))(
                *map(jnp.asarray, (q, k, v, g, beta)),
                conv_state,
                rec_state,
                *map(jnp.asarray, (cq, ck, cv)),
                None,
                *map(jnp.asarray, (qsl, dist, sl)),
                mesh=mesh,
                chunk_size=8,
                block_tables=jnp.asarray(bt.reshape(-1)),
                align_block_size=block_size)
    out, t0 = {}, 0
    for (r, *_), x in zip(toks, rows):
        out[r] = np.asarray(core[t0:t0 + len(x)])
        t0 += len(x)
    return conv, rec, out


@pytest.mark.parametrize("tp", [1, 2])
def test_prefix_caching_align_mode(tp):
    """Prefix caching ("align" mode, block size 8): request A prefills in
    block-aligned chunks, then a last partial chunk, then decodes across a
    block boundary; request B shares A's first 32 tokens and resumes from
    A's cached block 3 (a prefix-cache hit) next to A's steps. Every output
    equals a one-shot run, and the block of the last token of every step that
    ends on a block boundary holds the state after exactly that many tokens."""
    if len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} devices")
    rng = np.random.default_rng(11)
    C, bs = H * D_HEAD, 8
    stream = {
        "w": (0.5 * rng.standard_normal((3 * C, 1, KS))).astype(np.float32)
    }
    for r, n in ((0, 45), (1, 52)):
        stream[r] = dict(
            qkv=rng.standard_normal((n, 3 * C)).astype(np.float32),
            g=(-5.0 /
               (1 + np.exp(-rng.standard_normal((n, H, D_HEAD))))).astype(
                   np.float32),
            beta=(1 / (1 + np.exp(-rng.standard_normal((n, H))))).astype(
                np.float32))
    for key in ("qkv", "g", "beta"):  # B's prompt starts with A's 32 tokens
        stream[1][key][:32] = stream[0][key][:32]
    # A owns slots 3, 7, 2, 9, 4, 11; B hits A's block 3 (slot 9), then its
    # own 12, 13, 14 (blocks before the hit are null)
    block_tables = np.array(
        [[3, 7, 2, 9, 4, 11, 0, 0], [0, 0, 0, 9, 12, 13, 14, 0]], np.int32)
    steps = [[(0, 0, 0, 16)], [(0, 0, 16, 32)], [(0, 0, 32, 37),
                                                 (1, 1, 32, 40)],
             [(0, 0, 37, 38), (1, 1, 40, 50)]]
    steps += [[(0, 0, p, p + 1), (1, 1, 50 + i, 51 + i)]
              for i, p in enumerate(range(38, 40))]
    steps += [[(0, 0, p, p + 1)] for p in range(40, 45)]
    devs = np.array(jax.devices()[:tp]).reshape(1, 1, 1, 1, tp, 1, 1)
    mesh = Mesh(devs, MESH_AXIS_NAMES)
    conv = jnp.asarray(rng.standard_normal((16, KS - 1, 3 * C)), jnp.float32)
    rec = jnp.asarray(rng.standard_normal((16, H, D_HEAD, D_HEAD)),
                      jnp.float32)
    acc = {0: [], 1: []}
    after = {}  # (request, tokens) -> (conv, rec) of the block that holds it
    for step in steps:
        conv, rec, out = _align_step(stream, step, conv, rec, block_tables,
                                     mesh, bs)
        for r, row, a, e in step:
            acc[r].append(out[r])
            if e % bs == 0 or e in (45, 52):
                slot = block_tables[row][(e - 1) // bs]
                after[r, e] = (np.asarray(conv[slot]), np.asarray(rec[slot]))

    def reference(r, n):
        b = _batch(np.random.default_rng(0), [n], [False], 2, 3, 0)
        b.update(qkv=stream[r]["qkv"][:n],
                 g=stream[r]["g"][:n],
                 beta=stream[r]["beta"][:n],
                 conv_w=stream["w"])
        return _reference(b)

    for r, n in ((0, 45), (1, 52)):
        out = np.concatenate(acc[r])
        start = 0 if r == 0 else 32
        assert _rel(out, reference(r, n)[0][start:]) < 1e-4, r
    assert {k
            for k in after} == {(0, 16), (0, 32), (0, 40), (0, 45), (1, 40),
                                (1, 52)}
    for (r, n), (c, s) in after.items():
        _, ref_conv, ref_rec = reference(r, n)
        assert _rel(s, ref_rec[0]) < 1e-4, (r, n)
        if tp == 1:  # (the conv state's channels are per-shard for tp > 1)
            assert _rel(c, ref_conv[0]) < 1e-5, (r, n)


@pytest.mark.parametrize("tp", [1, 2])
def test_prefix_caching_with_spec_decode(tp):
    """Prefix caching ("align" mode, block size 8) with speculative decoding
    (3 drafts): the per-position states live in block-table columns
    (`group_slot`). Request A prefills 0-16 and 16-21, then runs verify
    windows with rejected drafts (junk inputs) across the 24 and 32 block
    boundaries; request B starts from A's cached block 1 (16 tokens) and
    runs next to A. Accepted outputs equal one plain run over the accepted
    tokens, every block whose last position was accepted holds the state
    after exactly that many tokens, and the shared block is never
    overwritten."""
    if len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} devices")
    rng = np.random.default_rng(13)
    C, bs, group = H * D_HEAD, 8, 4
    n = 44
    stream = {
        "w": (0.5 * rng.standard_normal((3 * C, 1, KS))).astype(np.float32)
    }
    for r in range(2):
        stream[r] = dict(
            qkv=rng.standard_normal((n + 12, 3 * C)).astype(np.float32),
            g=(-5.0 /
               (1 + np.exp(-rng.standard_normal((n + 12, H, D_HEAD))))).astype(
                   np.float32),
            beta=(1 / (1 + np.exp(-rng.standard_normal((n + 12, H))))).astype(
                np.float32))
    for key in ("qkv", "g", "beta"):  # B's prompt starts with A's 16 tokens
        stream[1][key][:16] = stream[0][key][:16]
    junk = list(range(n, n + 12))
    # A's columns; B: null, A's block 1 (cached), then its own
    block_tables = np.array([[3, 7, 2, 9, 4, 11, 5, 13, 15, 17, 0, 0],
                             [0, 7, 20, 21, 22, 23, 24, 25, 26, 27, 0, 0]],
                            np.int32)
    steps = [
        [(0, 0, list(range(16)))],
        [(0, 16, list(range(16, 21)))],
        # A: x21, x22 + 2 rejected; B: prefill from the hit
        [(0, 21, [21, 22, junk[0], junk[1]]), (1, 16, list(range(16, 24)))],
        # A crosses 24: 23, 24, 25 accepted, the last draft rejected
        [(0, 23, [23, 24, 25, junk[2]]), (1, 24, [24, 25, 26, 27])],
        [(0, 26, [26, 27, 28, 29]), (1, 28, [28, junk[3], junk[4], junk[5]])],
        # A crosses 32 with every draft rejected but the first token
        [(0, 30, [30, 31, junk[6], junk[7]]), (1, 29, [29, 30, 31, 32])],
        [(0, 32, [32, 33, 34, 35]), (1, 33, [33, 34, 35, 36])],
    ]
    keep = [
        None, None, {
            0: 2,
            1: 8
        }, {
            0: 3,
            1: 4
        }, {
            0: 4,
            1: 1
        }, {
            0: 2,
            1: 4
        }, None
    ]
    devs = np.array(jax.devices()[:tp]).reshape(1, 1, 1, 1, tp, 1, 1)
    mesh = Mesh(devs, MESH_AXIS_NAMES)
    conv = jnp.asarray(rng.standard_normal((32, KS - 1, 3 * C)), jnp.float32)
    rec = jnp.asarray(rng.standard_normal((32, H, D_HEAD, D_HEAD)),
                      jnp.float32)
    (cq, ck, cv) = (stream["w"][j * C:(j + 1) * C, 0].reshape(H, D_HEAD, KS)
                    for j in range(3))
    acc = {0: [], 1: []}
    shared = None
    for step, k in zip(steps, keep):
        T = sum(len(t) for _, _, t in step)
        pad, max_reqs = 3, 4
        cat = lambda key, shape: np.concatenate(  # noqa: E731
            [stream[r][key][t]
             for r, _, t in step] + [np.zeros((pad, ) + shape, np.float32)])
        qkv, g, beta = cat("qkv",
                           (3 * C, )), cat("g",
                                           (H, D_HEAD)), cat("beta", (H, ))
        q, kk, v = (qkv[:, j * C:(j + 1) * C].reshape(T + pad, H, D_HEAD)
                    for j in range(3))
        pos = np.concatenate([p0 + np.arange(len(t)) for _, p0, t in step] +
                             [np.zeros(pad, int)]).astype(np.int32)
        qsl = np.full(max_reqs + 1, T, np.int32)
        qsl[:len(step) + 1] = np.concatenate([[0],
                                              np.cumsum(
                                                  [len(t) for *_, t in step])])
        sl = np.zeros(max_reqs, np.int32)
        bt = np.zeros((max_reqs, block_tables.shape[1]), np.int32)
        for i, (r, p0, t) in enumerate(step):
            sl[i], bt[i] = p0 + len(t), block_tables[r]
        dist = np.array([0, len(step), len(step)], np.int32)
        with mesh:
            (conv, rec), core = jax.jit(
                K.run_jax_kda_attention,
                static_argnames=("mesh", "chunk_size", "align_block_size",
                                 "spec_group"))(
                                     *map(jnp.asarray, (q, kk, v, g, beta)),
                                     conv,
                                     rec,
                                     *map(jnp.asarray, (cq, ck, cv)),
                                     None,
                                     *map(jnp.asarray, (qsl, dist, sl)),
                                     mesh=mesh,
                                     chunk_size=8,
                                     positions=jnp.asarray(pos),
                                     spec_group=group,
                                     block_tables=jnp.asarray(bt.reshape(-1)),
                                     align_block_size=bs)
        t0 = 0
        for r, p0, t in step:
            m = len(t) if k is None else k[r]
            acc[r].append(np.asarray(core[t0:t0 + m]))
            t0 += len(t)
        if shared is None:  # A's block 1 after the first step: the cache
            shared = np.asarray(rec[7]), np.asarray(conv[7])
        np.testing.assert_array_equal(np.asarray(rec[7]), shared[0])
        np.testing.assert_array_equal(np.asarray(conv[7]), shared[1])

    def reference(r, total):
        b = _batch(np.random.default_rng(0), [total], [False], 2, 3, 0)
        b.update(qkv=stream[r]["qkv"][:total],
                 g=stream[r]["g"][:total],
                 beta=stream[r]["beta"][:total],
                 conv_w=stream["w"])
        return _reference(b)

    for r, start, total in ((0, 0, 36), (1, 16, 37)):
        out = np.concatenate(acc[r])
        assert out.shape[0] == total - start, r
        assert _rel(out, reference(r, total)[0][start:]) < 1e-4, r
    for r, blocks in ((0, (1, 2, 3)), (1, (2, 3))):
        for j in blocks:  # block j: the state after (j + 1) * 8 tokens
            _, ref_conv, ref_rec = reference(r, (j + 1) * bs)
            slot = block_tables[r][j]
            assert _rel(np.asarray(rec[slot]), ref_rec[0]) < 1e-4, (r, j)
            if tp == 1:
                assert _rel(np.asarray(conv[slot]), ref_conv[0]) < 1e-5, (r, j)
