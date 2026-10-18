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
"""KV cache specs and allocation for GLM-5.3-Flash on the JAX path
(`tpu_inference/runner/glm5_next_kv_cache.py`), checked against vLLM's own KV
cache planner."""
import json
import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from jax.sharding import Mesh, PartitionSpec  # noqa: E402
from vllm.config import (
    CacheConfig,
    ModelConfig,  # noqa: E402
    ParallelConfig,
    SchedulerConfig,
    VllmConfig)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from tests.models.jax.glm5_next_ref import make_tiny_ckpt as MT  # noqa: E402
from tpu_inference.core import hybrid_coordinator  # noqa: E402
from tpu_inference.layers.common.sharding import (  # noqa: E402
    MESH_AXIS_NAMES, ShardingAxisName)
from tpu_inference.runner import glm5_next_kv_cache as G  # noqa: E402
from tpu_inference.runner.kv_cache import KVCacheMetadata  # noqa: E402
from tpu_inference.runner.kv_cache_manager import KVCacheManager  # noqa: E402

REAL_CONFIG = os.path.join(os.path.dirname(__file__), "..", "models", "jax",
                           "glm5_next_ref", "data", "real_config.json")
BLOCK_SIZE = 64
MAX_REQS = 8
HBM_PER_DEVICE = 8 << 30


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("glm5_next_tiny_kv"))
    MT.write_tiny_checkpoint(d, seed=0)
    return d


@pytest.fixture(autouse=True)
def _reset_global_mamba_blocks():
    yield
    hybrid_coordinator._GLOBAL_MAMBA_NUM_BLOCKS = None


def _vllm_config(model_dir):
    return VllmConfig(
        model_config=ModelConfig(model=model_dir,
                                 skip_tokenizer_init=True,
                                 dtype="bfloat16",
                                 max_model_len=4096),
        # Prefix caching for KDA state (mamba "align" mode) comes later (M9).
        cache_config=CacheConfig(block_size=BLOCK_SIZE,
                                 gpu_memory_utilization=0.9,
                                 enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(max_num_seqs=MAX_REQS,
                                         max_model_len=4096,
                                         is_encoder_decoder=False),
        parallel_config=ParallelConfig(),
    )


def _mesh(tp):
    if len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} devices (XLA_FLAGS set after JAX init)")
    devs = np.array(jax.devices()[:tp]).reshape(1, 1, 1, 1, tp, 1, 1)
    return Mesh(devs, MESH_AXIS_NAMES)


def _runner(vllm_config, mesh):
    return types.SimpleNamespace(
        vllm_config=vllm_config,
        cache_config=vllm_config.cache_config,
        model_config=vllm_config.model_config,
        parallel_config=vllm_config.parallel_config,
        speculative_config=None,
        mesh=mesh,
        kv_cache_dtype=torch.bfloat16,
        attention_backend=None,
        max_num_reqs=MAX_REQS,
        kv_caches=[],
        layer_name_to_kvcache_index={},
    )


def _specs(tiny_dir, tp, monkeypatch):
    vllm_config = _vllm_config(tiny_dir)
    runner = _runner(vllm_config, _mesh(tp))
    monkeypatch.setattr("tpu_inference.utils.hbm_usage_bytes",
                        lambda devices: [(0, HBM_PER_DEVICE) for _ in devices])
    manager = KVCacheManager(runner)
    return vllm_config, runner, manager, manager.get_kv_cache_spec()


def test_layout_of_real_config():
    from transformers import Glm5NextConfig
    with open(REAL_CONFIG) as f:
        text = Glm5NextConfig.from_dict(json.load(f)).text_config
    lay = G.Glm5NextCacheLayout.from_config(text, torch.bfloat16)
    assert len(lay.kda_layers) == 34 and lay.mla_layers == tuple(
        range(3, 45, 4))
    assert lay.conv_shape == (3, 3 * 64 * 128)
    assert lay.state_shape == (64, 128, 128)
    assert lay.tail_shape == (4, 256)
    assert lay.mla_head_size == 512
    # 4 MiB recurrent state + 144 KiB conv per KDA layer and request
    assert lay.kda_page_bytes == 64 * 128 * 128 * 4 + 3 * 24576 * 2
    assert lay.tail_page_bytes == 4 * 256 * 2
    assert G.vllm_num_groups_per_type([11, 34, 11]) == [1, 4, 1]


def test_is_glm5_next(tiny_dir):
    assert G.is_glm5_next(_vllm_config(tiny_dir))


@pytest.mark.parametrize("tp", [1, 2])
def test_specs(tiny_dir, tp, monkeypatch):
    vllm_config, runner, manager, specs = _specs(tiny_dir, tp, monkeypatch)
    text = vllm_config.model_config.hf_text_config
    lay = G.Glm5NextCacheLayout.from_config(text, torch.bfloat16)
    want = {G.layer_name(i) for i in range(text.num_hidden_layers)}
    want |= {G.indexer_tail_name(i) for i in lay.mla_layers}
    assert set(specs) == want
    for i in range(text.num_hidden_layers):
        s = specs[G.layer_name(i)]
        if i in lay.kda_layers:
            assert isinstance(s, MambaSpec)
            assert s.shapes == (lay.conv_shape, lay.state_shape)
            assert s.dtypes == (torch.bfloat16, torch.float32)
        else:
            assert isinstance(s, MLAAttentionSpec)
            assert s.head_size == lay.mla_head_size and s.num_kv_heads == 1
            tail = specs[G.indexer_tail_name(i)]
            assert isinstance(tail, MambaSpec)
            assert tail.shapes == (lay.tail_shape, )
    # one padded page size for every spec, as the uniform-page grouping needs
    assert len({s.page_size_bytes for s in specs.values()}) == 1
    # the compact mamba pool: every request resident, plus the null slot
    assert manager._mamba_num_blocks >= MAX_REQS + 1
    assert vllm_config.cache_config.num_gpu_blocks_override > 0


@pytest.mark.parametrize("tp", [1, 2])
def test_vllm_planner_and_allocation(tiny_dir, tp, monkeypatch):
    """vLLM's planner accepts the specs; every layer gets its own TPU array(s)
    with the real shapes and shardings."""
    from vllm.v1.attention.backends.utils import (
        get_supported_kv_cache_layouts, resolve_kv_cache_layout)
    from vllm.v1.core.kv_cache_planning import get_kv_cache_configs
    vllm_config, runner, manager, specs = _specs(tiny_dir, tp, monkeypatch)
    text = vllm_config.model_config.hf_text_config
    lay = G.Glm5NextCacheLayout.from_config(text, torch.bfloat16)
    # What the engine core does at startup: the flax path registers no vLLM
    # attention backends, so the worker reports the default layouts.
    worker_layouts = [x.name for x in get_supported_kv_cache_layouts([])]
    resolve_kv_cache_layout(vllm_config, [worker_layouts], specs.values())
    page = next(iter(specs.values())).page_size_bytes
    kv_config = get_kv_cache_configs(vllm_config, [specs], [page * 1000])[0]
    # groups: 1 MLA, 1 indexer tail, KDA split to the MLA group size
    n_kda_groups = -(-len(lay.kda_layers) // len(lay.mla_layers))
    kinds = []
    for g in kv_config.kv_cache_groups:
        layer_specs = {type(specs[n]) for n in g.layer_names}
        assert len(layer_specs) == 1
        kinds.append(
            (layer_specs.pop().__name__,
             all(n.endswith(G.INDEXER_TAIL_SUFFIX) for n in g.layer_names)))
    assert sorted(kinds) == sorted([("MLAAttentionSpec",
                                     False), ("MambaSpec", True)] +
                                   [("MambaSpec", False)] * n_kda_groups)
    assert kv_config.num_blocks == vllm_config.cache_config.num_gpu_blocks_override

    layer_name_to_spec = {
        n: specs[n]
        for g in kv_config.kv_cache_groups
        for n in g.layer_names
    }
    metadata = {"mamba": KVCacheMetadata(), "regular_attn": KVCacheMetadata()}
    num_blocks_list = []
    G.initialize_glm5_next_kv_cache(manager, kv_config, layer_name_to_spec,
                                    runner.kv_caches, num_blocks_list,
                                    metadata)
    idx = runner.layer_name_to_kvcache_index
    assert set(idx) == set(specs)
    assert len(runner.kv_caches) == len(specs)
    slots = manager._mamba_num_blocks
    nb = kv_config.num_blocks
    for i in range(text.num_hidden_layers):
        entry = runner.kv_caches[idx[G.layer_name(i)]]
        if i in lay.kda_layers:
            conv, state = entry
            assert conv.shape == (slots, *lay.conv_shape)
            assert state.shape == (slots, *lay.state_shape)
            assert state.dtype == np.float32
            # heads (conv channels) sharded over the model axis
            assert conv.sharding.spec[2] is not None
            assert state.sharding.spec[1] is not None
        else:
            latent, pooled = entry
            assert latent.shape[0] == nb
            assert latent.shape[-1] == lay.mla_head_size
            # (the CPU platform may override the block size for MLA)
            block_size = vllm_config.cache_config.block_size
            assert pooled.shape == (nb, block_size // lay.index_kpool,
                                    lay.index_head_dim)
            (tail, ) = runner.kv_caches[idx[G.indexer_tail_name(i)]]
            assert tail.shape == (slots, *lay.tail_shape)
            # the indexer is replicated over TP
            assert tail.sharding.spec == PartitionSpec(
                ShardingAxisName.ATTN_DATA, None, None)
            assert tail.sharding.is_fully_replicated


def _planned(vllm_config, specs):
    from vllm.v1.attention.backends.utils import (
        get_supported_kv_cache_layouts, resolve_kv_cache_layout)
    from vllm.v1.core.kv_cache_planning import get_kv_cache_configs
    worker_layouts = [x.name for x in get_supported_kv_cache_layouts([])]
    resolve_kv_cache_layout(vllm_config, [worker_layouts], specs.values())
    page = next(iter(specs.values())).page_size_bytes
    return get_kv_cache_configs(vllm_config, [specs], [page * 1000])[0]


def test_planner_at_real_scale(tmp_path, monkeypatch):
    """The real layer counts (34 KDA, 11 MLA): vLLM forms 1 MLA group, 4 KDA
    groups and 1 indexer-tail group, and the uniform page is one layer of
    each."""
    with open(REAL_CONFIG) as f, open(tmp_path / "config.json", "w") as g:
        g.write(f.read())
    vllm_config = _vllm_config(str(tmp_path))
    runner = _runner(vllm_config, _mesh(1))
    monkeypatch.setattr("tpu_inference.utils.hbm_usage_bytes",
                        lambda devices: [(0, 64 << 30) for _ in devices])
    specs = KVCacheManager(runner).get_kv_cache_spec()
    kv_config = _planned(vllm_config, specs)
    sizes = sorted(len(g.layer_names) for g in kv_config.kv_cache_groups)
    assert len(kv_config.kv_cache_groups) == 6
    assert sizes == [8, 8, 9, 9, 11, 11]  # KDA 34 split 4 ways (layers[i::4])
    lay = G.Glm5NextCacheLayout.from_config(
        vllm_config.model_config.hf_text_config, torch.bfloat16)
    from tpu_inference.runner.kv_cache import get_attention_page_size_bytes
    bs = vllm_config.cache_config.block_size
    attn = (get_attention_page_size_bytes(
        runner.mesh, bs, 1, 512, torch.bfloat16, True) + bs // 4 * 128 * 2)
    want = attn + 4 * lay.kda_page_bytes + lay.tail_page_bytes
    assert {s.page_size_bytes for s in specs.values()} == {want}


def test_manager_initialize_kv_cache_dispatches(tiny_dir, monkeypatch):
    vllm_config, runner, manager, specs = _specs(tiny_dir, 1, monkeypatch)
    kv_config = _planned(vllm_config, specs)
    monkeypatch.setattr(manager, "maybe_reinitialize_input_batch",
                        lambda cfg: None)
    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(vllm_config):
        manager.initialize_kv_cache(kv_config)
    assert set(runner.layer_name_to_kvcache_index) == set(specs)
    mla = G.layer_name(2)  # tiny layer 2 is MLA
    latent, pooled = runner.kv_caches[runner.layer_name_to_kvcache_index[mla]]
    assert pooled.ndim == 3
