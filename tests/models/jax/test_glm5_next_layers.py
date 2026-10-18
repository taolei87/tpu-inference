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

from vllm.config import (
    ParallelConfig,
    VllmConfig,  # noqa: E402
    set_current_vllm_config)

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
    sorted by group; rows outside the local groups produce zeros; a fused
    activation splits the output into [gate | up]."""
    assert rhs_scale is None
    m, G_local = lhs.shape[0], rhs.shape[0]
    gid = jnp.searchsorted(jnp.cumsum(group_sizes),
                           jnp.arange(m),
                           side="right") - group_offset.reshape(-1)[0]
    ok = (gid >= 0) & (gid < G_local)
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
def _paged_md(lens, ctx, max_reqs, pages_per_seq, page_size, rng):
    """AttentionMetadata of a ragged step: request i has ctx[i] cached tokens
    and lens[i] new ones."""
    num_pages = max_reqs * pages_per_seq + 1
    pages = rng.permutation(np.arange(1, num_pages)).astype(np.int32)
    bt = pages[:max_reqs * pages_per_seq]
    qsl = np.full(max_reqs + 1, sum(lens), np.int32)
    qsl[:len(lens) + 1] = np.concatenate([[0], np.cumsum(lens)])
    pos = np.concatenate([np.arange(c, c + n) for n, c in zip(lens, ctx)])
    T = len(pos) + 3  # padding tokens
    pos = np.pad(pos, (0, 3)).astype(np.int32)
    seq_lens = np.zeros(max_reqs, np.int32)
    seq_lens[:len(lens)] = np.array(lens) + np.array(ctx)
    n_dec = sum(1 for n, c in zip(lens, ctx) if n == 1 and c > 0)
    md = AttentionMetadata(input_positions=jnp.asarray(pos),
                           block_tables=jnp.asarray(bt),
                           seq_lens=jnp.asarray(seq_lens),
                           query_start_loc=jnp.asarray(qsl),
                           request_distribution=jnp.asarray(
                               [n_dec, len(lens), len(lens)], jnp.int32))
    return md, num_pages, T


def test_dense_mla_matches_reference(tiny):
    """A request's prefill split over two scheduler steps (the second step
    batched with a decode of another request) == the reference over the
    whole sequence."""
    d, text, ref, rcfg = tiny
    mesh = _mesh()
    with jax.set_mesh(mesh):
        mla = G.Glm5NextMLA(text, jnp.float32, nnx.Rngs(0), mesh, None,
                            f"layers.{MLA_LAYER}.self_attn")
    # the indexer (sparse attention) is added in M4
    _load(mla,
          d,
          f"{RC.PREFIX}layers.{MLA_LAYER}.self_attn.",
          mesh,
          skip=(".indexer.", ))
    p = ref["layers"][MLA_LAYER]["attn"]
    rng = np.random.default_rng(4)
    D, page_size, pps, max_reqs = text.hidden_size, 8, 8, 4
    xa = rng.standard_normal((30, D)).astype(np.float32)  # request 0
    xb = rng.standard_normal((12, D)).astype(np.float32)  # request 1
    kvr_pad = -(-text.kv_lora_rank // 128) * 128

    def step(cache, md, x):
        x = np.pad(x, ((0, md.input_positions.shape[0] - len(x)), (0, 0)))
        (cache,
         _), y = jax.jit(lambda m, c, x, md: m(c, x, md))(mla, (cache, None),
                                                          jnp.asarray(x), md)
        return cache, np.asarray(y)

    md1, num_pages, _ = _paged_md([18, 11], [0, 0], max_reqs, pps, page_size,
                                  np.random.default_rng(5))
    cache = jnp.zeros((num_pages, page_size, 1, kvr_pad), jnp.float32)
    cache, y1 = step(cache, md1, np.concatenate([xa[:18], xb[:11]]))
    # second step: the rest of request 0 and one decode token of request 1
    md2 = AttentionMetadata(
        input_positions=jnp.asarray(
            np.pad(np.concatenate([np.arange(18, 30), [11]]),
                   (0, 3)).astype(np.int32)),
        block_tables=md1.block_tables,
        seq_lens=jnp.asarray(np.array([30, 12, 0, 0], np.int32)),
        query_start_loc=jnp.asarray(np.array([0, 12, 13, 13, 13], np.int32)),
        request_distribution=jnp.asarray([0, 2, 2], jnp.int32))
    cache, y2 = step(cache, md2, np.concatenate([xa[18:], xb[11:]]))
    ra, _ = R.mla_attention(p, jnp.asarray(xa[None]), rcfg, sparse=None)
    rb, _ = R.mla_attention(p, jnp.asarray(xb[None]), rcfg, sparse=None)
    assert _rel(y1[:18], ra[0, :18]) < 1e-4
    assert _rel(y1[18:29], rb[0, :11]) < 1e-4
    assert _rel(y2[:12], ra[0, 18:]) < 1e-4
    assert _rel(y2[12:13], rb[0, 11:]) < 1e-4
