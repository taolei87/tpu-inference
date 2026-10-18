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
"""GLM-5.3-Flash building blocks (mHC, dense MLP, MoE, dense NoPE MLA) loaded
from a tiny real-format checkpoint vs the functional reference."""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from flax import nnx  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

pytest.importorskip("transformers.models.glm5_next")

from vllm.config import VllmConfig  # noqa: E402
from vllm.config import ParallelConfig, set_current_vllm_config

from tests.models.jax.glm5_next_ref import make_tiny_ckpt as MT  # noqa: E402
from tests.models.jax.glm5_next_ref import ref_checkpoint as RC  # noqa: E402
from tests.models.jax.glm5_next_ref import ref_model as R  # noqa: E402
from tpu_inference.layers.common import fused_moe_gmm  # noqa: E402
from tpu_inference.layers.common.attention_metadata import \
    AttentionMetadata  # noqa: E402
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES  # noqa: E402
from tpu_inference.layers.jax import JaxModule, mhc  # noqa: E402
from tpu_inference.models.jax import glm5_next as G  # noqa: E402
from tpu_inference.models.jax.utils.weight_utils import \
    JaxAutoWeightsLoader  # noqa: E402

DENSE_LAYER, MOE_LAYER, MLA_LAYER = 0, 1, 2  # in the tiny model


@pytest.fixture(autouse=True)
def _highest_precision():
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("glm5_next_layers"))
    _, cfg = MT.write_tiny_checkpoint(d, seed=11)
    text = cfg.text_config
    ref = jax.tree.map(jnp.asarray, RC.load_ref_params(d, text,
                                                       with_mtp=False))
    return d, text, ref, R.Cfg.from_hf(text)


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def _mesh(tp=1):
    if len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} devices")
    return Mesh(
        np.array(jax.devices()[:tp]).reshape(1, 1, 1, 1, tp, 1, 1),
        MESH_AXIS_NAMES)


class _Root(JaxModule):
    """Nests the module under test one level down, as in a real model (vLLM's
    loader inspects the root and its direct children for a quant config)."""

    def __init__(self, module):
        self.m = module


def _load(module, ckpt_dir, prefix, mesh, skip=()):
    r = RC.CheckpointReader(ckpt_dir)
    weights = [("m." + n[len(prefix):], r.raw(n)) for n in r.names()
               if n.startswith(prefix) and not any(s in n for s in skip)]
    with jax.set_mesh(mesh):
        JaxAutoWeightsLoader(_Root(module)).load_weights(iter(weights))
    return module


# ----------------------------------------------------------------------------- mHC
def test_mhc_xla_matches_reference(tiny):
    d, text, ref, rcfg = tiny
    p = ref["layers"][0]["hc_attn"]  # fn [hc*D, M] in the reference layout
    fn = jnp.asarray(np.asarray(p["fn"]).T)  # checkpoint layout [M, hc*D]
    cfg = mhc.MHCConfig(hc=text.hc_mult,
                        sinkhorn_iters=text.hc_sinkhorn_iters,
                        hc_eps=text.hc_eps,
                        rms_eps=text.rms_norm_eps)
    rng = np.random.default_rng(0)
    T, D = 7, text.hidden_size
    streams = jnp.asarray(rng.standard_normal((T, text.hc_mult, D)),
                          jnp.float32)
    y = jnp.asarray(rng.standard_normal((T, D)), jnp.float32)
    post, comb, x = mhc.mhc_pre_xla(streams, fn, p["scale"], p["base"], cfg)
    rpost, rcomb, rx = R.hc_pre(p, streams[None], rcfg)
    assert _rel(post, rpost[0]) < 1e-6 and _rel(comb, rcomb[0]) < 1e-6
    assert _rel(x, rx[0]) < 1e-6
    new = mhc.mhc_post_xla(y, streams, post, comb)
    assert _rel(new, R.hc_post(rpost, rcomb, y[None], streams[None])[0]) < 1e-6
    fused = mhc.mhc_post_pre(y,
                             streams,
                             post,
                             comb,
                             fn,
                             p["scale"],
                             p["base"],
                             cfg,
                             use_kernel=False)
    assert _rel(fused[0], new) == 0.0
    np.testing.assert_allclose(mhc.mhc_head(streams), streams.mean(1))


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas kernels")
def test_mhc_kernels_match_xla(tiny):
    d, text, ref, _ = tiny
    p = ref["layers"][0]["hc_attn"]
    fn = jnp.asarray(np.asarray(p["fn"]).T)
    cfg = mhc.MHCConfig(text.hc_mult, text.hc_sinkhorn_iters, text.hc_eps,
                        text.rms_norm_eps)
    rng = np.random.default_rng(1)
    T, D = 64, text.hidden_size
    streams = jnp.asarray(rng.standard_normal((T, text.hc_mult, D)),
                          jnp.bfloat16)
    y = jnp.asarray(rng.standard_normal((T, D)), jnp.bfloat16)
    xla = mhc.mhc_pre(streams,
                      fn,
                      p["scale"],
                      p["base"],
                      cfg,
                      use_kernel=False)
    ker = mhc.mhc_pre(streams, fn, p["scale"], p["base"], cfg, use_kernel=True)
    for a, b in zip(ker, xla):
        assert _rel(a, b) < 2e-2
    xla = mhc.mhc_post_pre(y,
                           streams,
                           xla[0],
                           xla[1],
                           fn,
                           p["scale"],
                           p["base"],
                           cfg,
                           use_kernel=False)
    ker = mhc.mhc_post_pre(y,
                           streams,
                           ker[0],
                           ker[1],
                           fn,
                           p["scale"],
                           p["base"],
                           cfg,
                           use_kernel=True)
    for a, b in zip(ker, xla):
        assert _rel(a, b) < 2e-2


# ----------------------------------------------------------------------------- MLP / MoE
def test_dense_mlp_matches_reference(tiny):
    d, text, ref, rcfg = tiny
    mesh = _mesh()
    with jax.set_mesh(mesh):
        mlp = G.Glm5NextMLP(text.hidden_size, text.intermediate_size,
                            text.swiglu_limit, jnp.float32, nnx.Rngs(0))
    _load(mlp, d, f"{RC.PREFIX}layers.{DENSE_LAYER}.mlp.", mesh)
    x = jnp.asarray(
        np.random.default_rng(2).standard_normal((9, text.hidden_size)) * 3.0,
        jnp.float32)  # large enough to clamp
    out = mlp(x)
    assert _rel(out, R.mlp(ref["layers"][DENSE_LAYER]["mlp"], x, rcfg)) < 1e-5


def _fake_gmm(lhs,
              rhs,
              rhs_scale,
              rhs_bias,
              group_sizes,
              group_offset,
              fuse_act=None,
              preferred_element_type=None):
    """Pure-JAX stand-in for the TPU-only grouped matmul (gmm_v2): rows are
    sorted by group; rows outside the local groups produce zeros; quantized
    weights carry per-(K block, N) scales [G, num_blocks, 1, N]; a fused
    activation splits the output into [gate | up]."""
    m, G_local = lhs.shape[0], rhs.shape[0]
    gid = jnp.searchsorted(jnp.cumsum(group_sizes),
                           jnp.arange(m),
                           side="right") - group_offset.reshape(-1)[0]
    ok = (gid >= 0) & (gid < G_local)
    rhs = rhs.astype(jnp.float32)
    if rhs_scale is not None:
        K, nb = rhs.shape[1], rhs_scale.shape[1]
        rhs = rhs * jnp.repeat(
            rhs_scale[:, :, 0, :].astype(jnp.float32), K // nb, axis=1)
    w = rhs[jnp.clip(gid, 0, G_local - 1)]
    out = jnp.einsum("mk,mkn->mn", lhs.astype(jnp.float32),
                     w.astype(jnp.float32))
    if rhs_bias is not None:
        out = out + rhs_bias[jnp.clip(gid, 0, G_local - 1), 0]
    if fuse_act is not None:
        n = out.shape[-1] // 2
        g, u = out[:, :n], out[:, n:]
        if fuse_act == "silu_and_mul_with_clamp":
            out = G.swiglu_clamped(g, u, 10.0)
        elif fuse_act == "silu":
            out = jax.nn.silu(g) * u
        else:
            raise NotImplementedError(fuse_act)
    out = jnp.where(ok[:, None], out, 0.0)
    return out.astype(preferred_element_type or lhs.dtype)


@pytest.mark.parametrize("tp", [1, 2])
def test_moe_matches_reference(tiny, tp, monkeypatch):
    """The MoE block through the real GMM_TP path (routing with the
    correction bias, permutation, combine), with only the TPU grouped-matmul
    kernel replaced."""
    if jax.default_backend() != "tpu":
        monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", _fake_gmm)
    d, text, ref, rcfg = tiny
    mesh = _mesh(tp)
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig())):
        with jax.set_mesh(mesh):
            moe = G.Glm5NextMoE(text, jnp.float32, nnx.Rngs(0), mesh, None,
                                f"layers.{MOE_LAYER}.mlp")
        _load(moe, d, f"{RC.PREFIX}layers.{MOE_LAYER}.mlp.", mesh)
        x = jnp.asarray(
            np.random.default_rng(3).standard_normal((16, text.hidden_size)),
            jnp.float32)
        with jax.set_mesh(mesh):
            out = jax.jit(lambda mod, x: mod(x))(moe, x)
    want = R.moe_dense(ref["layers"][MOE_LAYER]["mlp"], x[None], rcfg)[0]
    assert _rel(out, want) < 1e-4


# ----------------------------------------------------------------------------- MLA
class _Steps:
    """Persistent per-request pages and slots, and the AttentionMetadata of
    each scheduler step ([(request id, first position, num tokens)], decodes
    first)."""

    def __init__(self, max_reqs, pages_per_seq, page_size, seed=0):
        rng = np.random.default_rng(seed)
        self.max_reqs, self.pps = max_reqs, pages_per_seq
        self.num_pages = max_reqs * pages_per_seq + 1
        self.bt = rng.permutation(np.arange(1, self.num_pages)).astype(
            np.int32).reshape(max_reqs, pages_per_seq)
        self.slots = rng.permutation(np.arange(1, max_reqs + 2))[:max_reqs]
        self.num_slots = max_reqs + 2

    def metadata(self, step):
        n = sum(k for _, _, k in step)
        T = -(-(n + 1) // 16) * 16  # padded like the runner's token buckets
        pos = np.zeros(T, np.int32)
        qsl = np.full(self.max_reqs + 1, n, np.int32)
        qsl[0] = 0
        seq_lens = np.zeros(self.max_reqs, np.int32)
        bt = np.zeros((self.max_reqs, self.pps), np.int32)
        slots = np.zeros(self.max_reqs, np.int32)
        t = 0
        for i, (rid, p0, k) in enumerate(step):
            pos[t:t + k] = np.arange(p0, p0 + k)
            t += k
            qsl[i + 1] = t
            seq_lens[i] = p0 + k
            bt[i] = self.bt[rid]
            slots[i] = self.slots[rid]
        n_dec = sum(1 for _, p0, k in step if k == 1 and p0 > 0)
        return AttentionMetadata(
            input_positions=jnp.asarray(pos),
            block_tables=jnp.asarray(bt.reshape(-1)),
            seq_lens=jnp.asarray(seq_lens),
            query_start_loc=jnp.asarray(qsl),
            request_distribution=jnp.asarray(
                [n_dec, len(step), len(step)], jnp.int32),
            mamba_state_indices=jnp.asarray(slots)), T


# Three requests over four steps: chunk boundaries at every position mod 4
# (pools complete across steps through the tail), decodes batched first.
_SEQ_LENS = {0: 45, 1: 23, 2: 11}
_STEPS = [
    [(0, 0, 18), (1, 0, 7), (2, 0, 9)],
    [(2, 9, 1), (0, 18, 5), (1, 7, 9)],
    [(0, 23, 1), (2, 10, 1), (1, 16, 7)],
    [(0, 24, 21)],
]


@pytest.mark.parametrize("index_topk", [64, 16])
def test_mla_matches_reference(tmp_path, index_topk):
    """MLA with the k-pool indexer over multi-step ragged batches == the
    reference over each whole sequence: dense regime (index_topk 64 covers
    every request) and sparse (16: 4 pools per query)."""
    d = str(tmp_path)
    _, cfg = MT.write_tiny_checkpoint(d, seed=12, index_topk=index_topk)
    text = cfg.text_config
    ref = jax.tree.map(jnp.asarray, RC.load_ref_params(d, text))
    rcfg = R.Cfg.from_hf(text)
    mesh = _mesh()
    with jax.set_mesh(mesh):
        mla = G.Glm5NextMLA(text, jnp.float32, nnx.Rngs(0), mesh, None,
                            f"layers.{MLA_LAYER}.self_attn")
    _load(mla, d, f"{RC.PREFIX}layers.{MLA_LAYER}.self_attn.", mesh)
    p = ref["layers"][MLA_LAYER]["attn"]
    rng = np.random.default_rng(4)
    xs = {
        r: rng.standard_normal((n, text.hidden_size)).astype(np.float32)
        for r, n in _SEQ_LENS.items()
    }
    page_size = 8
    st = _Steps(max_reqs=4, pages_per_seq=8, page_size=page_size)
    kvr_pad = -(-text.kv_lora_rank // 128) * 128
    kp, ihd = text.index_kpool, text.index_head_dim
    cache = ((jnp.zeros((st.num_pages, page_size, 1, kvr_pad), jnp.float32),
              jnp.zeros((st.num_pages, page_size // kp, ihd),
                        jnp.float32)), (jnp.zeros((st.num_slots, kp, 2 * ihd),
                                                  jnp.float32), ))
    run = jax.jit(lambda m, c, x, md: m(c, x, (md, md)))
    got = {r: [] for r in xs}
    for step in _STEPS:
        md, T = st.metadata(step)
        x = np.concatenate([xs[r][p0:p0 + k] for r, p0, k in step])
        cache, y = run(mla, cache,
                       jnp.asarray(np.pad(x, ((0, T - len(x)), (0, 0)))), md)
        t = 0
        for r, _, k in step:
            got[r].append(np.asarray(y[t:t + k]))
            t += k
    sparse = "always" if index_topk < max(_SEQ_LENS.values()) else None
    for r, x in xs.items():
        want, _ = R.mla_attention(p, jnp.asarray(x[None]), rcfg, sparse=sparse)
        assert _rel(np.concatenate(got[r]), want[0]) < 1e-4, r
    if sparse:  # the test exercises sparsity: dense attention differs
        dense, _ = R.mla_attention(p, jnp.asarray(xs[0][None]), rcfg)
        assert _rel(np.concatenate(got[0]), dense[0]) > 1e-3
