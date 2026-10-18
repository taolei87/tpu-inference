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
"""KV cache specs and allocation for GLM-5.3-Flash (`glm5_next`) on the JAX
(flax_nnx) path.

Per decoder layer i the model owns:

  KDA layers      `layer.{i}`               MambaSpec: (conv [K-1, 3C],
                                            recurrent state [H, d, d] f32), in
                                            the compact per-request slot pool.
  MLA layers      `layer.{i}`               MLAAttentionSpec, head = the 512-d
                                            latent (NoPE: no rope part). On TPU
                                            the layer's cache is a pair
                                            (latent pages, pooled indexer keys
                                            [num_blocks, block_size / kpool,
                                            index_head_dim]): the pooled keys
                                            ride on the latent's block table.
                  `layer.{i}.indexer_tail`  MambaSpec: the raw indexer keys and
                                            gates [kpool, 2 * index_head_dim] of
                                            the request's incomplete k-pool,
                                            replicated over TP.

The indexer tail is per-request, fixed-size state, so it lives in the same
compact slot pool as the KDA state (same `mamba_state_indices`, chunked prefill
continuation, and later the same speculative-decoding slot groups). vLLM's GPU
implementation uses an MLA spec with `tokens_per_state` for the pooled keys
and a `CircularBufferSpec` for the tail; those only group with the packed
(BLHNC) KV layout, which the TPU path does not use.

Like the other hybrid models on TPU, every spec reports one uniform padded
page size so vLLM's uniform-page grouping applies; the attention / mamba HBM
split is computed from the real per-layer bytes
(`KVCacheManager._maybe_set_compact_mamba_num_blocks_override`), and
`initialize_glm5_next_kv_cache` allocates each layer with its real shapes.
"""
import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch
from jax.sharding import NamedSharding, PartitionSpec
from torchax.ops.mappings import t2j_dtype
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, MambaSpec

from tpu_inference import utils as common_utils
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.runner.kv_cache import (create_kv_caches,
                                           create_mamba_cache,
                                           get_attention_page_size_bytes)

if TYPE_CHECKING:
    from tpu_inference.runner.kv_cache_manager import KVCacheManager

logger = init_logger(__name__)

GLM5_NEXT_ARCHITECTURES = frozenset(
    {"Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"})
INDEXER_TAIL_SUFFIX = ".indexer_tail"


def is_glm5_next(vllm_config) -> bool:
    model_config = vllm_config.model_config
    return model_config is not None and bool(
        GLM5_NEXT_ARCHITECTURES.intersection(model_config.architectures or []))


def layer_name(i: int) -> str:
    return f"layer.{i}"


def indexer_tail_name(i: int) -> str:
    return f"layer.{i}{INDEXER_TAIL_SUFFIX}"


@dataclasses.dataclass(frozen=True)
class Glm5NextCacheLayout:
    """Per-layer cache shapes (global, i.e. unsharded) of a GLM-5.3-Flash
    text config."""
    kda_layers: tuple[int, ...]
    mla_layers: tuple[int, ...]
    conv_shape: tuple[int, int]  # (K - 1, 3 * H * d)
    state_shape: tuple[int, int, int]  # (H, d, d)
    tail_shape: tuple[int, int]  # (kpool, 2 * index_head_dim)
    conv_dtype: torch.dtype
    state_dtype: torch.dtype
    tail_dtype: torch.dtype
    mla_head_size: int  # padded latent width
    index_kpool: int
    index_head_dim: int

    @classmethod
    def from_config(
            cls,
            text_config,
            model_dtype: torch.dtype,
            ssm_dtype: torch.dtype = torch.float32) -> "Glm5NextCacheLayout":
        # transformers canonicalizes the MLA layer type to "indexed_attention"
        # (raw config.json says "deepseek_sparse_attention"): only test for
        # the linear-attention string.
        types = list(text_config.layer_types)
        kda = tuple(i for i, t in enumerate(types) if t == "linear_attention")
        mla = tuple(i for i, t in enumerate(types) if t != "linear_attention")
        H, d = text_config.linear_num_heads, text_config.linear_head_dim
        K = text_config.linear_conv_kernel_dim
        kp, ihd = text_config.index_kpool, text_config.index_head_dim
        assert text_config.qk_rope_head_dim == 0, "GLM-5.3-Flash MLA is NoPE"
        return cls(kda_layers=kda,
                   mla_layers=mla,
                   conv_shape=(K - 1, 3 * H * d),
                   state_shape=(H, d, d),
                   tail_shape=(kp, 2 * ihd),
                   conv_dtype=model_dtype,
                   state_dtype=ssm_dtype,
                   tail_dtype=model_dtype,
                   mla_head_size=common_utils.align_to(
                       text_config.kv_lora_rank, 128),
                   index_kpool=kp,
                   index_head_dim=ihd)

    @staticmethod
    def _bytes(shapes, dtypes) -> int:
        return sum(
            int(np.prod(s)) * torch.tensor([], dtype=d).element_size()
            for s, d in zip(shapes, dtypes))

    @property
    def kda_page_bytes(self) -> int:
        return self._bytes((self.conv_shape, self.state_shape),
                           (self.conv_dtype, self.state_dtype))

    @property
    def tail_page_bytes(self) -> int:
        return self._bytes((self.tail_shape, ), (self.tail_dtype, ))

    def pooled_shape(self, num_blocks: int,
                     block_size: int) -> tuple[int, int, int]:
        assert block_size % self.index_kpool == 0, (block_size,
                                                    self.index_kpool)
        return (num_blocks, block_size // self.index_kpool,
                self.index_head_dim)


def vllm_num_groups_per_type(type_counts: list[int]) -> list[int]:
    """KV cache groups vLLM forms per spec type in its uniform-page grouping
    (`kv_cache_planning._get_kv_cache_groups_uniform_page_size`): the group
    size is the smallest type's layer count, or the largest's when they are
    within 1.5x."""
    lo, hi = min(type_counts), max(type_counts)
    group_size = hi if hi < lo * 1.5 else lo
    return [cdiv(c, group_size) for c in type_counts]


def _model_dtype(runner) -> torch.dtype:
    dtype = runner.model_config.dtype
    return dtype if isinstance(dtype, torch.dtype) else torch.bfloat16


def get_glm5_next_kv_cache_spec(
        manager: "KVCacheManager") -> dict[str, KVCacheSpec]:
    """The flax_nnx path's per-layer KV cache specs for GLM-5.3-Flash."""
    runner = manager.runner
    cache_config = runner.cache_config
    text_config = runner.model_config.hf_text_config
    block_size = cache_config.block_size
    assert common_utils.get_mesh_shape_product(
        runner.mesh, ShardingAxisName.KV_CONTEXT) == 1, (
            "decode context parallelism is not supported for GLM-5.3-Flash")
    num_spec = (runner.speculative_config.num_speculative_tokens
                if runner.speculative_config else 0)
    if num_spec:
        raise NotImplementedError(
            "speculative decoding is not supported for GLM-5.3-Flash yet")

    layout = Glm5NextCacheLayout.from_config(text_config, _model_dtype(runner))
    latent_page = get_attention_page_size_bytes(runner.mesh, block_size, 1,
                                                layout.mla_head_size,
                                                runner.kv_cache_dtype, True,
                                                runner.attention_backend)
    pooled_page = (block_size // layout.index_kpool * layout.index_head_dim *
                   torch.tensor([], dtype=layout.tail_dtype).element_size())
    attn_page = latent_page + pooled_page
    kda_page, tail_page = layout.kda_page_bytes, layout.tail_page_bytes
    n_mla, n_kda = len(layout.mla_layers), len(layout.kda_layers)

    # One uniform page size for every spec: the per-KVCacheTensor footprint of
    # vLLM's grouping (one layer from each group), as for other hybrid models.
    g_attn, g_kda, g_tail = vllm_num_groups_per_type([n_mla, n_kda, n_mla])
    uniform = g_attn * attn_page + g_kda * kda_page + g_tail * tail_page
    manager._hybrid_uniform_page_size_bytes = int(uniform)
    cache_config.mamba_page_size_padded = int(uniform)
    logger.info(
        "GLM-5.3-Flash KV cache: %d MLA layers (latent %d B + pooled keys %d B "
        "per block), %d KDA layers (%d B per slot), %d indexer tails (%d B "
        "per slot); uniform page %d B.", n_mla, latent_page, pooled_page,
        n_kda, kda_page, n_mla, tail_page, uniform)
    # Attention gets the HBM the compact mamba pool leaves; the sizing works
    # on totals, so pass them as one "layer" each.
    manager._maybe_set_compact_mamba_num_blocks_override(
        attn_page_size_bytes=n_mla * attn_page,
        unpadded_mamba_page_size_bytes=n_kda * kda_page + n_mla * tail_page,
        num_attn_layers=1,
        num_mamba_layers=1)

    mamba_common = dict(block_size=cache_config.mamba_block_size or block_size,
                        page_size_padded=int(uniform),
                        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                        mamba_cache_mode=cache_config.mamba_cache_mode,
                        num_speculative_blocks=num_spec)
    specs: dict[str, KVCacheSpec] = {}
    for i in range(text_config.num_hidden_layers):
        if i in layout.kda_layers:
            specs[layer_name(i)] = MambaSpec(shapes=(layout.conv_shape,
                                                     layout.state_shape),
                                             dtypes=(layout.conv_dtype,
                                                     layout.state_dtype),
                                             **mamba_common)
        else:
            specs[layer_name(i)] = manager._create_attention_spec(
                block_size, 1, layout.mla_head_size)
            specs[indexer_tail_name(i)] = MambaSpec(
                shapes=(layout.tail_shape, ),
                dtypes=(layout.tail_dtype, ),
                **mamba_common)
    return specs


def initialize_glm5_next_kv_cache(manager: "KVCacheManager",
                                  kv_cache_config: KVCacheConfig,
                                  layer_name_to_spec: dict[str, KVCacheSpec],
                                  kv_caches: list, num_blocks_list: list,
                                  metadata: dict) -> None:
    """Allocates every GLM-5.3-Flash cache with its real shape and fills
    `runner.layer_name_to_kvcache_index`.

    kv_caches entries: KDA layer -> (conv_state, recurrent_state); MLA layer
    -> (latent cache, pooled indexer keys); indexer tail -> (tail, ).
    """
    runner = manager.runner
    mesh = runner.mesh
    text_config = runner.model_config.hf_text_config
    block_size = runner.cache_config.block_size
    divisor = common_utils.get_mesh_shape_product(mesh,
                                                  ShardingAxisName.ATTN_DATA)
    num_blocks = kv_cache_config.num_blocks
    override = runner.cache_config.num_gpu_blocks_override
    if override is not None:
        num_blocks = min(num_blocks, override)
    num_blocks = (num_blocks // divisor) * divisor
    if manager._mamba_num_blocks is not None:
        mamba_num_blocks = manager._mamba_num_blocks
    else:
        from tpu_inference.core.hybrid_coordinator import (
            mamba_blocks_per_request, mamba_pool_size)
        _, stride = mamba_blocks_per_request(runner.vllm_config,
                                             is_align_mode=False)
        mamba_num_blocks = mamba_pool_size(runner.max_num_reqs, stride,
                                           divisor)
    if manager.actual_mamba_num_blocks is None:
        manager.actual_mamba_num_blocks = mamba_num_blocks
    kv_cache_config.mamba_num_blocks = int(mamba_num_blocks)

    D, H = ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD
    conv_sharding = NamedSharding(mesh, PartitionSpec(D, None, H))
    state_sharding = NamedSharding(mesh, PartitionSpec(D, H, None, None))
    tail_sharding = NamedSharding(mesh, PartitionSpec(D, None, None))
    pooled_sharding = NamedSharding(mesh,
                                    PartitionSpec(ShardingAxisName.BATCH))

    layout = Glm5NextCacheLayout.from_config(text_config, _model_dtype(runner))

    def add(name, entry, n):
        kv_caches.append(entry)
        num_blocks_list.append(n)
        runner.layer_name_to_kvcache_index[name] = len(kv_caches) - 1

    for i in range(text_config.num_hidden_layers):
        name = layer_name(i)
        spec = layer_name_to_spec[name]
        if isinstance(spec, MambaSpec):  # KDA
            (cs, ss), (cd, sd) = spec.shapes, spec.dtypes
            entry = (create_mamba_cache((mamba_num_blocks, *cs), t2j_dtype(cd),
                                        conv_sharding),
                     create_mamba_cache((mamba_num_blocks, *ss), t2j_dtype(sd),
                                        state_sharding))
            add(name, entry, mamba_num_blocks)
            metadata["mamba"].count += 1
            if metadata["mamba"].shape is None:
                metadata["mamba"].shape = tuple(a.shape for a in entry)
                metadata["mamba"].dtype = tuple(a.dtype for a in entry)
                metadata["mamba"].sharding = tuple(a.sharding for a in entry)
            continue
        latent = create_kv_caches(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=1,
            head_size=spec.head_size,
            mesh=mesh,
            layer_names=[name],
            cache_dtype=t2j_dtype(spec.dtype),
            use_mla=True,
            attention_backend=runner.attention_backend)[0]
        tail_spec = layer_name_to_spec[indexer_tail_name(i)]
        pooled = create_mamba_cache(
            layout.pooled_shape(num_blocks, block_size),
            t2j_dtype(tail_spec.dtypes[0]), pooled_sharding)
        add(name, (latent, pooled), num_blocks)
        metadata["regular_attn"].count += 1
        if metadata["regular_attn"].shape is None:
            metadata["regular_attn"].shape = (latent.shape, pooled.shape)
            metadata["regular_attn"].dtype = (latent.dtype, pooled.dtype)
            metadata["regular_attn"].sharding = (latent.sharding,
                                                 pooled.sharding)
        tail = create_mamba_cache((mamba_num_blocks, *tail_spec.shapes[0]),
                                  t2j_dtype(tail_spec.dtypes[0]),
                                  tail_sharding)
        add(indexer_tail_name(i), (tail, ), mamba_num_blocks)
