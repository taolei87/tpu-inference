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
"""GLM-5.3-Flash JAX model (`models/jax/glm5_next.py`) vs the functional
reference, through the runner-facing interface: ragged batches over paged MLA
caches and slot-indexed KDA / indexer-tail state, across scheduler steps."""
import dataclasses
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

pytest.importorskip("transformers.models.glm5_next")

from vllm.config import ModelConfig  # noqa: E402
from vllm.config import (CacheConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig, set_current_vllm_config)

from tests.models.jax.glm5_next_ref import make_tiny_ckpt as MT  # noqa: E402
from tests.models.jax.glm5_next_ref import ref_checkpoint as RC  # noqa: E402
from tests.models.jax.glm5_next_ref import ref_model as R  # noqa: E402
from tests.models.jax.test_glm5_next_layers import _fake_gmm  # noqa: E402
from tpu_inference.layers.common import fused_moe_gmm  # noqa: E402
from tpu_inference.layers.common.attention_metadata import \
    AttentionMetadata  # noqa: E402
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES  # noqa: E402
from tpu_inference.models.jax.glm5_next import \
    Glm5NextForConditionalGeneration  # noqa: E402
from tpu_inference.runner import glm5_next_kv_cache as KV  # noqa: E402

PAGE, PAGES_PER_SEQ, MAX_REQS = 16, 8, 4

# MLA models on TPU run with the new model design and attention DP (enforced by
# the TPU platform's config check, which is active on TPU hosts).
_GLM_ADDITIONAL_CONFIG = {
    "sharding": {
        "sharding_strategy": {
            "enable_dp_attention": True
        }
    }
}


@pytest.fixture(autouse=True)
def _new_model_design(monkeypatch):
    monkeypatch.setenv("NEW_MODEL_DESIGN", "1")


@pytest.fixture(autouse=True)
def _highest_precision_and_cpu_gmm(monkeypatch):
    if jax.default_backend() != "tpu":
        monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", _fake_gmm)
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("glm5_next_model"))
    MT.write_tiny_checkpoint(d, seed=21)
    return d


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def _vllm_config(model_dir, dtype="float32"):
    return VllmConfig(
        model_config=ModelConfig(model=model_dir,
                                 skip_tokenizer_init=True,
                                 dtype=dtype,
                                 max_model_len=PAGE * PAGES_PER_SEQ),
        cache_config=CacheConfig(block_size=PAGE, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(max_num_seqs=MAX_REQS,
                                         max_model_len=PAGE * PAGES_PER_SEQ,
                                         is_encoder_decoder=False),
        parallel_config=ParallelConfig(),
        additional_config=_GLM_ADDITIONAL_CONFIG,
    )


def _build(model_dir, tp=1, dtype="float32", attn_dp=1):
    """The registered model class, built and loaded the way model_loader
    does (abstract mesh context, then load_weights over the checkpoint)."""
    if len(jax.devices()) < tp * attn_dp:
        pytest.skip(f"needs {tp * attn_dp} devices")
    vllm_config = _vllm_config(model_dir, dtype)
    from tpu_inference.layers.jax.quantization import \
        get_tpu_quantization_config
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config)
    # as get_flax_model does: models are built with the JAX dtype
    from tpu_inference.utils import to_jax_dtype
    vllm_config.model_config.dtype = to_jax_dtype(
        vllm_config.model_config.dtype)
    mesh = Mesh(
        np.array(jax.devices()[:tp * attn_dp]).reshape(1, attn_dp, 1, 1, tp, 1,
                                                       1), MESH_AXIS_NAMES)
    with set_current_vllm_config(vllm_config), jax.set_mesh(mesh):
        model = Glm5NextForConditionalGeneration(vllm_config,
                                                 jax.random.key(0), mesh)
        r = RC.CheckpointReader(model_dir)
        model.load_weights(iter((n, r.raw(n)) for n in r.names()))
    return vllm_config, model, mesh


class _Caches:
    """The per-layer caches the runner allocates (glm5_next_kv_cache layout)
    and one AttentionMetadata per scheduler step."""

    def __init__(self,
                 text,
                 dtype=jnp.float32,
                 seed=0,
                 latent_dtype=None,
                 num_spec=0):
        lay = KV.Glm5NextCacheLayout.from_config(text, None, num_spec=num_spec)
        self.lay = lay
        rng = np.random.default_rng(seed)
        self.num_pages = MAX_REQS * PAGES_PER_SEQ + 1
        self.block_tables = rng.permutation(np.arange(
            1, self.num_pages)).astype(np.int32)
        # each request owns a group of num_spec + 1 slots (base = its slot)
        group = num_spec + 1
        self.slots = (1 + group * rng.permutation(MAX_REQS)).astype(np.int32)
        slots = group * MAX_REQS + 2
        kvr = -(-text.kv_lora_rank // 128) * 128
        self.kv_caches, self.index = [], {}
        for i in range(text.num_hidden_layers):
            if i in lay.kda_layers:
                entry = (jnp.zeros((slots, *lay.conv_shape), dtype),
                         jnp.zeros((slots, *lay.state_shape), jnp.float32))
            else:
                entry = (jnp.zeros((self.num_pages, PAGE, 1, kvr), latent_dtype
                                   or dtype),
                         jnp.zeros(lay.pooled_shape(self.num_pages, PAGE),
                                   dtype))
                self.index[KV.indexer_tail_name(i)] = len(self.kv_caches) + 1
            self.index[KV.layer_name(i)] = len(self.kv_caches)
            self.kv_caches.append(entry)
            if i in lay.mla_layers:
                self.kv_caches.append((jnp.zeros((slots, *lay.tail_shape),
                                                 dtype), ))

    def metadata(self, step):
        """step: [(request id, first position, num tokens)], decodes first."""
        # padding tokens, to a multiple of 16 like the runner's token buckets
        T = -(-(sum(n for _, _, n in step) + 1) // 16) * 16
        pos = np.zeros(T, np.int32)
        qsl = np.full(MAX_REQS + 1, sum(n for _, _, n in step), np.int32)
        qsl[0] = 0
        seq_lens = np.zeros(MAX_REQS, np.int32)
        bt = np.zeros((MAX_REQS, PAGES_PER_SEQ), np.int32)
        state = np.zeros(MAX_REQS, np.int32)
        own = self.block_tables.reshape(MAX_REQS, PAGES_PER_SEQ)
        t = 0
        for i, (rid, p0, n) in enumerate(step):
            pos[t:t + n] = np.arange(p0, p0 + n)
            t += n
            qsl[i + 1] = t
            seq_lens[i] = p0 + n
            bt[i] = own[rid]
            state[i] = self.slots[rid]
        n_dec = sum(1 for _, p0, n in step if n == 1 and p0 > 0)
        return AttentionMetadata(
            input_positions=jnp.asarray(pos),
            block_tables=jnp.asarray(bt.reshape(-1)),
            seq_lens=jnp.asarray(seq_lens),
            query_start_loc=jnp.asarray(qsl),
            request_distribution=jnp.asarray(
                [n_dec, len(step), len(step)], jnp.int32),
            mamba_state_indices=jnp.asarray(state)), T


def _run(model, mesh, caches, step, ids):
    md, T = caches.metadata(step)
    tokens = np.concatenate([ids[rid][p0:p0 + n] for rid, p0, n in step])
    tokens = np.pad(tokens, (0, T - len(tokens)))
    names = tuple(caches.index.items())

    @jax.jit
    def fwd(model, kv, tok, md):
        kv, h, _, _ = model(kv, tok, md, None, None, names)
        return kv, h, model.compute_logits(h)

    with jax.set_mesh(mesh):
        caches.kv_caches, h, logits = fwd(model, caches.kv_caches,
                                          jnp.asarray(tokens), md)
    out, t = {}, 0
    for rid, p0, n in step:
        out[rid] = (np.asarray(h[t:t + n]), np.asarray(logits[t:t + n]))
        t += n
    return out


class _DPCaches:
    """Data-parallel attention: one `_Caches` per rank (rank-local pages and
    slots), stacked along the page / slot axis like the runner's
    ATTN_DATA-sharded caches, with the runner's packed metadata layout (one
    equally padded segment of tokens and of every metadata field per rank)."""

    def __init__(self, text, ranks):
        self.ranks = [_Caches(text, seed=r) for r in range(ranks)]
        self.index = self.ranks[0].index
        self.kv_caches = jax.tree.map(lambda *a: jnp.concatenate(a),
                                      *[c.kv_caches for c in self.ranks])

    def run(self, model, mesh, steps, ids):
        """steps: one step per rank, [((rank, local id), first position,
        num tokens)]; ids keyed by (rank, local id)."""
        mds = [
            c.metadata([(lid, p0, n) for (_, lid), p0, n in st])
            for c, st in zip(self.ranks, steps)
        ]
        T = max(t for _, t in mds)
        tokens, fields = [], []
        for (md, _), st in zip(mds, steps):
            tk = np.concatenate([ids[g][p0:p0 + n] for g, p0, n in st])
            tokens.append(np.pad(tk, (0, T - len(tk))))
            fields.append(
                dataclasses.replace(md,
                                    input_positions=jnp.pad(
                                        md.input_positions,
                                        (0, T - md.input_positions.shape[0]))))
        md = jax.tree.map(lambda *a: jnp.concatenate(a), *fields)
        names = tuple(self.index.items())

        @jax.jit
        def fwd(model, kv, tok, md):
            kv, h, _, _ = model(kv, tok, md, None, None, names)
            return kv, h, model.compute_logits(h)

        with jax.set_mesh(mesh):
            self.kv_caches, h, logits = fwd(
                model, self.kv_caches, jnp.asarray(np.concatenate(tokens)), md)
        out = {}
        for r, st in enumerate(steps):
            t = r * T
            for g, p0, n in st:
                out[g] = (np.asarray(h[t:t + n]), np.asarray(logits[t:t + n]))
                t += n
        return out


def _reference(model_dir, ids, sparse="auto"):
    from transformers import Glm5NextConfig
    vllm_text = Glm5NextConfig.from_pretrained(model_dir).text_config
    params = jax.tree.map(jnp.asarray,
                          RC.load_ref_params(model_dir, vllm_text))
    rcfg = R.Cfg.from_hf(vllm_text)
    out = {}
    for rid, seq in ids.items():
        h, _ = R.forward(params, jnp.asarray(seq[None]), rcfg, sparse=sparse)
        out[rid] = (np.asarray(h[0]), np.asarray(R.logits(params, h)[0]))
    return out


@pytest.mark.parametrize("tp", [1, 2])
def test_model_matches_reference_over_two_steps(tiny_dir, tp):
    vllm_config, model, mesh = _build(tiny_dir, tp)
    text = vllm_config.model_config.hf_text_config
    rng = np.random.default_rng(1)
    ids = {
        r: rng.integers(0, text.vocab_size, n)
        for r, n in ((0, 18), (1, 10), (2, 2))
    }
    ref = _reference(tiny_dir, ids)
    caches = _Caches(text)
    # step 1: three new requests in one ragged batch
    s1 = _run(model, mesh, caches, [(0, 0, 13), (1, 0, 9), (2, 0, 1)], ids)
    # step 2: two decodes first, then request 0's chunked-prefill remainder
    s2 = _run(model, mesh, caches, [(1, 9, 1), (2, 1, 1), (0, 13, 5)], ids)
    for rid, ((h1, l1), (h2, l2)) in {r: (s1[r], s2[r]) for r in ids}.items():
        h, logits = np.concatenate([h1, h2]), np.concatenate([l1, l2])
        assert _rel(h, ref[rid][0]) < 1e-4, rid
        assert _rel(logits, ref[rid][1]) < 1e-4, rid


def test_model_matches_reference_with_attention_dp(tmp_path):
    """Two attention-DP ranks, each with its own requests, rank-local pages
    and slots, over three steps (decodes first, chunked prefill remainders),
    in the sparse regime (index_topk 16) so the pooled keys and the tails
    carried across steps decide each query's tokens."""
    d = str(tmp_path)
    MT.write_tiny_checkpoint(d, seed=23, index_topk=16)
    vllm_config, model, mesh = _build(d, attn_dp=2)
    text = vllm_config.model_config.hf_text_config
    rng = np.random.default_rng(4)
    lens = {(0, 0): 37, (0, 1): 9, (1, 0): 30, (1, 1): 34}
    ids = {g: rng.integers(0, text.vocab_size, n) for g, n in lens.items()}
    ref = _reference(d, ids, sparse="always")
    caches = _DPCaches(text, 2)
    steps = [
        [[((0, 0), 0, 13), ((0, 1), 0, 7)], [((1, 0), 0, 18), ((1, 1), 0, 9)]],
        [[((0, 1), 7, 1), ((0, 0), 13, 15)], [((1, 1), 9, 14),
                                              ((1, 0), 18, 11)]],
        [[((0, 1), 8, 1), ((0, 0), 28, 9)], [((1, 0), 29, 1),
                                             ((1, 1), 23, 11)]],
    ]
    outs = [caches.run(model, mesh, st, ids) for st in steps]
    for g in ids:
        parts = [o[g] for o in outs if g in o]
        h = np.concatenate([p[0] for p in parts])
        logits = np.concatenate([p[1] for p in parts])
        assert h.shape[0] == lens[g], g
        assert _rel(h, ref[g][0]) < 1e-4, g
        assert _rel(logits, ref[g][1]) < 1e-4, g


def test_registry(tiny_dir):
    try:
        from tpu_inference.models.common.model_loader import \
            _get_model_architecture
        hf_config = _vllm_config(tiny_dir).model_config.hf_config
        cls = _get_model_architecture(hf_config)
    except (AttributeError, ImportError) as e:
        # The registry imports every model; some need a TPU jaxlib
        # (jax.experimental.transfer).
        pytest.skip(f"model registry not importable here: {e}")
    assert cls is Glm5NextForConditionalGeneration


def test_loaded_parameters(tiny_dir):
    vllm_config, model, _ = _build(tiny_dir)
    names = {n for n, _ in model.named_parameters()}
    assert "model.layers.0.hc_attn_fn" in names
    assert "model.layers.2.self_attn.kv_b_proj.weight" in names
    assert "model.layers.2.self_attn.indexer.index_kpool_compress_ape" in names


def test_fp8_checkpoint(tmp_path):
    """A tiny FP8 checkpoint (block-128 e4m3 on the real FP8 tensors) loads
    through Fp8Config and stays close to the reference on the dequantized
    weights."""
    d = str(tmp_path)
    MT.write_tiny_checkpoint(d, seed=22, fp8=True)
    # vLLM requires a 16-bit model dtype with FP8 weights
    vllm_config, model, mesh = _build(d, dtype="bfloat16")
    assert type(vllm_config.quant_config).__name__ == "Fp8Config"
    text = vllm_config.model_config.hf_text_config
    rng = np.random.default_rng(2)
    ids = {0: rng.integers(0, text.vocab_size, 12)}
    ref = _reference(d, ids)  # f32 on the exactly dequantized FP8 weights
    # The FP8 GMM kernel's bf16 dots can't lower at "highest" precision
    with jax.default_matmul_precision("default"):
        out = _run(model, mesh, _Caches(text, dtype=jnp.bfloat16),
                   [(0, 0, 12)], ids)
    # TPU's default precision also runs the f32 matmuls (mHC, KDA state) in
    # bf16: about 0.11 there against 0.064 on CPU
    assert _rel(out[0][0], ref[0][0]) < 0.15
    # top-1 agreement on most positions
    agree = (out[0][1].argmax(-1) == ref[0][1].argmax(-1)).mean()
    assert agree >= 0.75, agree


def test_sparse_model_matches_reference(tmp_path):
    """index_topk 16 (4 pools per query) with 37- and 30-token requests over
    three steps: the sparse indexer through the whole model."""
    d = str(tmp_path)
    MT.write_tiny_checkpoint(d, seed=23, index_topk=16)
    vllm_config, model, mesh = _build(d)
    text = vllm_config.model_config.hf_text_config
    rng = np.random.default_rng(3)
    ids = {
        0: rng.integers(0, text.vocab_size, 37),
        1: rng.integers(0, text.vocab_size, 30)
    }
    ref = _reference(d, ids, sparse="always")
    caches = _Caches(text)
    steps = [[(0, 0, 14), (1, 0, 17)], [(1, 17, 1), (0, 14, 22)],
             [(0, 36, 1), (1, 18, 12)]]
    got = {r: [] for r in ids}
    for step in steps:
        out = _run(model, mesh, caches, step, ids)
        for r in out:
            got[r].append(out[r])
    for r in ids:
        h = np.concatenate([g[0] for g in got[r]])
        logits = np.concatenate([g[1] for g in got[r]])
        assert _rel(h, ref[r][0]) < 1e-4, r
        assert _rel(logits, ref[r][1]) < 1e-4, r


def test_num_valid_tokens_for_moe_padding():
    """The MoE routes the step's padding rows to expert 0, from the token
    count in query_start_loc; attention DP (rank-interleaved padding) opts
    out."""
    from tpu_inference.models.jax.glm5_next import _num_valid_tokens
    md = AttentionMetadata(input_positions=jnp.zeros(16, jnp.int32),
                           query_start_loc=jnp.asarray([0, 3, 5, 5, 5]))
    devs = np.array(jax.devices())
    mesh = Mesh(devs[:1].reshape(1, 1, 1, 1, 1, 1, 1), MESH_AXIS_NAMES)
    assert int(_num_valid_tokens(md, mesh)) == 5
    assert int(_num_valid_tokens((md, md), mesh)) == 5  # MLA: (latent, tail)
    if len(devs) >= 2:
        dp = Mesh(devs[:2].reshape(1, 2, 1, 1, 1, 1, 1), MESH_AXIS_NAMES)
        assert _num_valid_tokens(md, dp) is None


def test_fp8_fused_expert_all_reduce(tmp_path):
    """At TP > 1 with fp8 weights the routed and shared experts' partial sums
    are added and all-reduced once; same output as two all-reduces."""
    d = str(tmp_path)
    MT.write_tiny_checkpoint(d, seed=22, fp8=True)
    vllm_config, model, mesh = _build(d, tp=2, dtype="bfloat16")
    text = vllm_config.model_config.hf_text_config
    moes = [
        layer.mlp for layer in model.model.layers
        if type(layer.mlp).__name__ == "Glm5NextMoE"
    ]
    assert moes and all(m.reduce_axes for m in moes)
    ids = {0: np.random.default_rng(2).integers(0, text.vocab_size, 12)}
    with jax.default_matmul_precision("default"):
        fused = _run(model, mesh, _Caches(text, dtype=jnp.bfloat16),
                     [(0, 0, 12)], ids)
        for m in moes:
            m.reduce_axes = ()
            m.shared_experts.down_proj.quant_method.linear_config.defer_all_reduce = False
        split = _run(model, mesh, _Caches(text, dtype=jnp.bfloat16),
                     [(0, 0, 12)], ids)
    ref = _reference(d, ids)
    e_fused, e_split = _rel(fused[0][0],
                            ref[0][0]), _rel(split[0][0], ref[0][0])
    print("FUSEDREDUCE", e_fused, e_split, _rel(fused[0][0], split[0][0]))
    # bf16 rounding of the partial sums: as close to the f32 reference
    assert e_fused < 1.1 * e_split + 1e-3, (e_fused, e_split)


def test_moe_token_splits(tiny_dir, monkeypatch):
    """GLM5_MOE_TOKEN_SPLITS runs the routed experts as token slices: the
    same output as one call (routing and GMM rows are per token)."""
    from tpu_inference.models.jax import glm5_next as G
    vllm_config, model, mesh = _build(tiny_dir)
    text = vllm_config.model_config.hf_text_config
    ids = {0: np.random.default_rng(3).integers(0, text.vocab_size, 30)}
    monkeypatch.setattr(G, "MOE_SPLIT_MIN_TOKENS", 16)
    monkeypatch.setenv("GLM5_MOE_TOKEN_SPLITS", "1")
    one = _run(model, mesh, _Caches(text), [(0, 0, 30)], ids)
    monkeypatch.setenv("GLM5_MOE_TOKEN_SPLITS", "2")
    two = _run(model, mesh, _Caches(text), [(0, 0, 30)], ids)
    np.testing.assert_allclose(two[0][0], one[0][0], rtol=1e-5, atol=1e-5)


def test_image_prompt_matches_hf(tiny_dir):
    """An image prompt end to end: the vision tower's embeddings merged over
    the image tokens (embed_multimodal -> embed_input_ids -> the model on
    inputs_embeds) give transformers' logits."""
    import torch
    from transformers import \
        Glm5NextForConditionalGeneration as HFGlm5NextForConditionalGeneration
    vllm_config, model, mesh = _build(tiny_dir)
    assert model.has_vision
    hf_cfg = vllm_config.model_config.hf_config
    grid = (1, 4, 6)  # patches; 6 image tokens after the 2x2 merge
    rng = np.random.default_rng(7)
    pixels = torch.as_tensor(rng.standard_normal((24, 3 * 2 * 14 * 14)),
                             dtype=torch.float32)
    ids = np.array([5, 9, 17, hf_cfg.image_start_token_id] +
                   [hf_cfg.image_token_id] * 6 +
                   [hf_cfg.image_end_token_id, 33, 41, 2], np.int64)
    hf = HFGlm5NextForConditionalGeneration.from_pretrained(
        tiny_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        ref = hf(input_ids=torch.as_tensor(ids[None]),
                 pixel_values=pixels,
                 image_grid_thw=torch.tensor([grid])).logits[0].numpy()

    mm = model.embed_multimodal(pixel_values=pixels,
                                image_grid_thw=torch.tensor([grid]))
    assert len(mm) == 1 and mm[0].shape[0] == 6
    text = vllm_config.model_config.hf_text_config
    caches = _Caches(text)
    md, T = caches.metadata([(0, 0, len(ids))])
    tokens = jnp.asarray(np.pad(ids, (0, T - len(ids))).astype(np.int32))
    embeds = model.embed_input_ids(tokens, jnp.concatenate(mm))
    names = tuple(caches.index.items())

    @jax.jit
    def fwd(model, kv, tok, md, embeds):
        kv, h, _, _ = model(kv, tok, md, embeds, None, names)
        return model.compute_logits(h)

    with jax.set_mesh(mesh):
        logits = np.asarray(fwd(model, caches.kv_caches, tokens, md,
                                embeds))[:len(ids)]
    assert _rel(logits, ref) < 1e-4


@pytest.mark.parametrize("grid", [(1, 4, 6), (2, 4, 4)])
def test_vision_tower_matches_hf(grid):
    """The vision tower alone against transformers' Glm5NextVisionModel
    (random weights, same names): a single frame and a two-frame grid
    (attention stays within each frame)."""
    import torch
    from flax import nnx
    from transformers.models.glm5_next.configuration_glm5_next import \
        Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import \
        Glm5NextVisionModel as HFVision

    from tpu_inference.models.jax.glm5_next_vision import Glm5NextVisionModel
    torch.manual_seed(0)
    cfg = Glm5NextVisionConfig(depth=2,
                               hidden_size=64,
                               num_heads=4,
                               intermediate_size=128,
                               out_hidden_size=96,
                               projection_intermediate_size=160)
    hf = HFVision(cfg).float().eval()
    with torch.no_grad():
        for p in hf.parameters():
            p.normal_(0, 0.05)
    ours = Glm5NextVisionModel(cfg, jnp.float32, nnx.Rngs(0), None)
    sd = hf.state_dict()
    params = dict(ours.named_parameters())
    assert set(params) == set(sd)
    for name, p in params.items():  # the loader's convention
        w = sd[name].numpy()
        p.set_value(jnp.asarray(w.T if w.ndim == 2 else w))
    x = torch.randn(int(np.prod(grid)), 3 * 2 * 14 * 14)
    with torch.no_grad():
        ref = hf(x, grid_thw=torch.tensor([grid])).pooler_output.numpy()
    out = np.asarray(ours(jnp.asarray(x.numpy()), grid))
    assert out.shape == ref.shape
    assert _rel(out, ref) < 1e-5


def test_text_only_skips_vision(tiny_dir, monkeypatch):
    """language_model_only: no vision tower is built or loaded."""
    make = _vllm_config

    def text_only(*args, **kwargs):
        cfg = make(*args, **kwargs)
        cfg.model_config.multimodal_config.language_model_only = True
        return cfg

    monkeypatch.setitem(globals(), "_vllm_config", text_only)
    _, model, _ = _build(tiny_dir)
    assert not model.has_vision and not hasattr(model.model, "visual")


def test_vision_bucketed_encode_matches_exact(tiny_dir):
    """embed_multimodal pads each grid's patches to a power-of-two bucket
    (padding is its own attention segment, sliced off): the same embeddings
    as encoding the exact grid, and grids sharing a bucket share a compile."""
    import torch
    _, model, _ = _build(tiny_dir)
    rng = np.random.default_rng(1)
    for grid in [(1, 4, 6), (1, 6, 4), (2, 4, 4), (1, 10, 6)]:
        n = int(np.prod(grid))
        x = jnp.asarray(rng.standard_normal((n, 3 * 2 * 14 * 14)), jnp.float32)
        exact = np.asarray(model.model.visual(x, grid))
        mm = model.embed_multimodal(pixel_values=torch.as_tensor(
            np.asarray(x)),
                                    image_grid_thw=torch.tensor([grid]))
        np.testing.assert_allclose(np.asarray(mm[0]),
                                   exact,
                                   rtol=1e-5,
                                   atol=1e-5)


def _mtp_caches(text, caches: "_Caches", dtype=jnp.float32):
    """The MTP layer's caches next to a _Caches (same pages and slots): its
    latent + pooled keys and its indexer tail, under the decoder naming at
    index num_hidden_layers."""
    lay, i = caches.lay, text.num_hidden_layers
    kvr = -(-text.kv_lora_rank // 128) * 128
    kv = list(caches.kv_caches)
    index = dict(caches.index)
    index[KV.layer_name(i)] = len(kv)
    kv.append((jnp.zeros((caches.num_pages, PAGE, 1, kvr), dtype),
               jnp.zeros(lay.pooled_shape(caches.num_pages, PAGE), dtype)))
    index[KV.indexer_tail_name(i)] = len(kv)
    kv.append((jnp.zeros((MAX_REQS + 2, *lay.tail_shape), dtype), ))
    return kv, index


def test_mtp_layer_matches_reference(tiny_dir):
    """The MTP draft model's layer over a prefill step, on the reference
    target's hidden states: the reference mtp_layer's output (embeddings of
    the next tokens, zeroed at position 0 as vLLM does)."""
    from tpu_inference.models.jax.glm5_next_mtp import Glm5NextMTPModel
    vllm_config, _, mesh = _build(tiny_dir)
    text = vllm_config.model_config.hf_text_config
    from transformers import Glm5NextConfig
    rtext = Glm5NextConfig.from_pretrained(tiny_dir).text_config
    params = jax.tree.map(jnp.asarray,
                          RC.load_ref_params(tiny_dir, rtext, with_mtp=True))
    rcfg = R.Cfg.from_hf(rtext)
    with set_current_vllm_config(vllm_config), jax.set_mesh(mesh):
        mtp = Glm5NextMTPModel(vllm_config, jax.random.key(1), mesh)
        r = RC.CheckpointReader(tiny_dir)
        mtp.load_weights(iter((n, r.raw(n)) for n in r.names()))
    # the proposer shares the target's embedding (left unloaded here)
    emb_w = r.raw("model.language_model.embed_tokens.weight")
    mtp.model.embed_tokens.weight.set_value(jnp.asarray(emb_w.float().numpy()))
    rng = np.random.default_rng(6)
    ids = rng.integers(0, text.vocab_size, 21)
    h_ref, _ = R.forward(params, jnp.asarray(ids[None]), rcfg)
    nxt = np.roll(ids, -1)[:-1]  # the draft input at position t is token t+1
    emb = params["embed"][jnp.asarray(nxt[None])]
    emb = emb.at[:, 0].set(0)
    ref, _ = R.mtp_layer(params["mtp"][0], emb, h_ref[:, :-1], rcfg)

    caches = _Caches(text)
    kv, index = _mtp_caches(text, caches)
    md, T = caches.metadata([(0, 0, len(nxt))])
    tok = jnp.asarray(np.pad(nxt, (0, T - len(nxt))).astype(np.int32))
    hid = jnp.pad(h_ref[0, :-1], ((0, T - len(nxt)), (0, 0)))
    names = tuple(index.items())

    @jax.jit
    def fwd(mtp, kv, tok, hid, md):
        _, h, _, _ = mtp(kv, tok, hid, md, names)
        return h

    with jax.set_mesh(mesh):
        h = np.asarray(fwd(mtp, kv, tok, hid, md))[:len(nxt)]
    assert _rel(h, ref[0]) < 1e-4


def test_spec_decode_rejected_drafts_leave_no_trace(tmp_path, monkeypatch):
    """The whole model with speculative-decoding state (slot groups of 4,
    ring tails, index_topk 16 so the indexer selects): request 0 runs a
    verify window whose 3 drafts are junk and rejected, then resumes after
    the one accepted token, while request 1 decodes alongside. Every accepted
    token's hidden state and logits equal the reference over the true
    tokens: KDA states, the indexer tail and the MLA latents roll back."""
    from tpu_inference.layers.jax.attention import kda as kda_layer
    monkeypatch.setattr(kda_layer, "_spec_group", lambda: 4)
    d = str(tmp_path)
    MT.write_tiny_checkpoint(d, seed=23, index_topk=16)
    vllm_config, model, mesh = _build(d)
    text = vllm_config.model_config.hf_text_config
    rng = np.random.default_rng(12)
    ids = {
        0: rng.integers(0, text.vocab_size, 22),
        1: rng.integers(0, text.vocab_size, 14)
    }
    ref = _reference(d, ids, sparse="always")
    caches = _Caches(text, num_spec=3)
    junk = rng.integers(0, text.vocab_size, 3)
    seqs = {0: np.concatenate([ids[0][:14], junk, ids[0][14:]]), 1: ids[1]}

    def window(rid, p0, n, start):  # rows start.. of seqs[rid] at p0..
        return (rid, p0, n), seqs[rid][start:start + n]

    steps = [
        [window(0, 0, 13, 0), window(1, 0, 9, 0)],
        # r0: true token 13 + 3 junk drafts (only 13 accepted); r1 decodes
        [window(1, 9, 1, 9), window(0, 13, 4, 13)],
        [window(1, 10, 1, 10), window(0, 14, 8, 17)],
        [window(1, 11, 3, 11)],
    ]
    keep = [None, {0: 1, 1: 1}, None, None]
    got = {0: ([], []), 1: ([], [])}
    for step, k in zip(steps, keep):
        sched = [w for w, _ in step]
        seq_ids = {w[0]: t for w, t in step}
        out = _run_spec(model, mesh, caches, sched, seq_ids)
        for rid, p0, n in sched:
            m = n if k is None else k[rid]
            got[rid][0].append(out[rid][0][:m])
            got[rid][1].append(out[rid][1][:m])
    for rid in ids:
        h, logits = (np.concatenate(x) for x in got[rid])
        assert h.shape[0] == len(ids[rid]), rid
        assert _rel(h, ref[rid][0]) < 1e-4, rid
        assert _rel(logits, ref[rid][1]) < 1e-4, rid


def _run_spec(model, mesh, caches, step, toks):
    """_run with explicit tokens per scheduled request."""
    md, T = caches.metadata(step)
    tokens = np.concatenate([toks[rid] for rid, _, _ in step])
    tokens = np.pad(tokens, (0, T - len(tokens)))
    names = tuple(caches.index.items())

    @jax.jit
    def fwd(model, kv, tok, md):
        kv, h, _, _ = model(kv, tok, md, None, None, names)
        return kv, h, model.compute_logits(h)

    with jax.set_mesh(mesh):
        caches.kv_caches, h, logits = fwd(model, caches.kv_caches,
                                          jnp.asarray(tokens), md)
    out, t = {}, 0
    for rid, p0, n in step:
        out[rid] = (np.asarray(h[t:t + n]), np.asarray(logits[t:t + n]))
        t += n
    return out
