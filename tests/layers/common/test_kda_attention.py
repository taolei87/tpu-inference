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
