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
    )


def _build(model_dir, tp=1, dtype="float32"):
    """The registered model class, built and loaded the way model_loader
    does (abstract mesh context, then load_weights over the checkpoint)."""
    if len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} devices")
    vllm_config = _vllm_config(model_dir, dtype)
    from tpu_inference.layers.jax.quantization import \
        get_tpu_quantization_config
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config)
    # as get_flax_model does: models are built with the JAX dtype
    from tpu_inference.utils import to_jax_dtype
    vllm_config.model_config.dtype = to_jax_dtype(
        vllm_config.model_config.dtype)
    mesh = Mesh(
        np.array(jax.devices()[:tp]).reshape(1, 1, 1, 1, tp, 1, 1),
        MESH_AXIS_NAMES)
    with set_current_vllm_config(vllm_config), jax.set_mesh(mesh):
        model = Glm5NextForConditionalGeneration(vllm_config,
                                                 jax.random.key(0), mesh)
        r = RC.CheckpointReader(model_dir)
        model.load_weights(iter((n, r.raw(n)) for n in r.names()))
    return vllm_config, model, mesh


class _Caches:
    """The per-layer caches the runner allocates (glm5_next_kv_cache layout)
    and one AttentionMetadata per scheduler step."""

    def __init__(self, text, dtype=jnp.float32, seed=0):
        lay = KV.Glm5NextCacheLayout.from_config(text, None)
        self.lay = lay
        rng = np.random.default_rng(seed)
        self.num_pages = MAX_REQS * PAGES_PER_SEQ + 1
        self.block_tables = rng.permutation(np.arange(
            1, self.num_pages)).astype(np.int32)
        self.slots = rng.permutation(np.arange(1, MAX_REQS + 2))[:MAX_REQS]
        slots, kvr = MAX_REQS + 2, -(-text.kv_lora_rank // 128) * 128
        self.kv_caches, self.index = [], {}
        for i in range(text.num_hidden_layers):
            if i in lay.kda_layers:
                entry = (jnp.zeros((slots, *lay.conv_shape), dtype),
                         jnp.zeros((slots, *lay.state_shape), jnp.float32))
            else:
                entry = (jnp.zeros((self.num_pages, PAGE, 1, kvr), dtype),
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
    out = _run(model, mesh, _Caches(text, dtype=jnp.bfloat16), [(0, 0, 12)],
               ids)
    assert _rel(out[0][0], ref[0][0]) < 0.1
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
