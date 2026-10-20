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
"""GLM-5.3-Flash (HF `glm5_next`) in JAX.

The text model is a stack of 45 layers, each with mHC residual streams
(`layers/jax/mhc.py`) around an attention block and an MLP block:
  * attention: KDA linear attention (`layers/jax/attention/kda.py`) on 34
    layers, NoPE MLA with a DeepSeek-style sparse indexer on 11 layers;
  * MLP: a dense clamped-SwiGLU MLP on the first 3 layers, a MoE (288 routed
    experts, top-8, sigmoid scores steered by a correction bias, x2.5, plus a
    shared expert) on the rest.

Module and parameter names follow the checkpoint (`model.language_model.` ->
`model.`). Tensors the checkpoint stores in FP8 (MLA q_a / q_b / kv_a / o,
dense and expert MLPs) get the FP8 quant config; everything else is
unquantized.
"""
import functools
import re
from typing import Any, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.model_executor.models.utils import WeightsMapper

import tpu_inference.envs as envs
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.sparse_mla_indexer import run_sparse_mla_step
from tpu_inference.layers.jax import JaxModule, mhc
from tpu_inference.layers.jax.attention.kda import KimiDeltaAttention
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear, JaxLmHead
from tpu_inference.layers.jax.moe.moe import JaxRoutedExperts
from tpu_inference.layers.jax.norm import JaxLayerNorm, JaxRmsNorm
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.glm5_next_vision import Glm5NextVisionModel
from tpu_inference.models.jax.utils.multi_modal_utils import \
    merge_multimodal_embeddings
from tpu_inference.models.jax.utils.weight_utils import (
    JaxAutoWeightsLoader, LoadableWithIterator,
    load_nnx_param_from_reshaped_torch)
from tpu_inference.runner import glm5_next_kv_cache
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)
init_fn = nnx.initializers.uniform()

# GLM5_MOE_TOKEN_SPLITS applies to steps of at least this many tokens
MOE_SPLIT_MIN_TOKENS = 4096


def swiglu_clamped(gate: jax.Array, up: jax.Array, limit: float) -> jax.Array:
    """silu(min(gate, limit)) * clip(up, -limit, limit)."""
    return jax.nn.silu(jnp.minimum(gate, limit)) * jnp.clip(up, -limit, limit)


class Glm5NextMLP(JaxModule):
    """Clamped SwiGLU MLP (the dense layers and the shared expert)."""

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 swiglu_limit: float,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config=None,
                 prefix: str = ""):
        # Like DeepSeek-V3's dense MLP: tensor-parallel over ATTN_HEAD, which is
        # disjoint from the token axis (ATTN_DATA) under attention DP;
        # MLP_TENSOR overlaps it in the new model design.
        mlp = ShardingAxisName.ATTN_HEAD

        def linear(i, o, sharding, name):
            return JaxLinear(i,
                             o,
                             use_bias=False,
                             dtype=dtype,
                             kernel_init=nnx.with_partitioning(
                                 init_fn, sharding),
                             rngs=rng,
                             quant_config=quant_config,
                             prefix=f"{prefix}.{name}")

        self.gate_proj = linear(hidden_size, intermediate_size, (None, mlp),
                                "gate_proj")
        self.up_proj = linear(hidden_size, intermediate_size, (None, mlp),
                              "up_proj")
        self.down_proj = linear(intermediate_size, hidden_size, (mlp, None),
                                "down_proj")
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(
            swiglu_clamped(self.gate_proj(x), self.up_proj(x),
                           self.swiglu_limit))


class Glm5NextRouter(JaxModule):
    """`mlp.gate`: f32 router weight and the noaux_tc correction bias."""

    def __init__(self, hidden_size: int, num_experts: int, rng: nnx.Rngs):
        self.weight = nnx.Param(init_fn(rng.params(),
                                        (hidden_size, num_experts),
                                        jnp.float32),
                                out_sharding=(None, None),
                                eager_sharding=False)
        self.e_score_correction_bias = nnx.Param(jnp.zeros((num_experts, ),
                                                           jnp.float32),
                                                 out_sharding=(None, ),
                                                 eager_sharding=False)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Router logits [T, E] in f32 (sigmoid is applied by the experts)."""
        return jnp.einsum("td,de->te",
                          x.astype(jnp.float32),
                          self.weight[...],
                          precision=jax.lax.Precision.HIGHEST)


def _fused_reduce_axes(experts, shared, mesh: Mesh) -> tuple:
    """The mesh axes to reduce the routed and shared experts' partial sums
    over in one all-reduce, or () to keep their separate all-reduces: needs
    the TP GMM backend, no attention DP, blockwise-fp8 shared weights (whose
    matmul can skip its psum) and the same effective tensor-parallel axes."""
    from tpu_inference.layers.common.moe import MoEBackend
    from tpu_inference.layers.jax.quantization.fp8 import \
        Fp8BlockwiseLinearMethod

    def effective(axes):
        axes = (axes, ) if isinstance(axes, str) else axes
        return tuple(a for a in axes if get_mesh_shape_product(mesh, a) > 1)

    if (getattr(experts, "moe_backend", None) != MoEBackend.GMM_TP
            or get_mesh_shape_product(mesh, ShardingAxisName.ATTN_DATA) > 1
            or not isinstance(shared.down_proj.quant_method,
                              Fp8BlockwiseLinearMethod)):
        return ()
    gmm, mlp = (effective(ShardingAxisName.MLP_TENSOR),
                effective(ShardingAxisName.ATTN_HEAD))
    return gmm if gmm and set(gmm) == set(mlp) else ()


class Glm5NextMoE(JaxModule):

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 quant_config, prefix: str):
        assert config.scoring_func == "sigmoid" and config.n_group == 1, (
            config.scoring_func, config.n_group)
        # The GMM kernels' clamped SwiGLU ("silu_and_mul_with_clamp") has its
        # limit fixed at 10.0.
        assert config.swiglu_limit == 10.0, config.swiglu_limit
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = Glm5NextRouter(config.hidden_size, config.n_routed_experts,
                                   rng)
        self.experts = JaxRoutedExperts(
            dtype=dtype,
            num_local_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size_moe=config.moe_intermediate_size,
            hidden_act="silu",
            rngs=rng,
            mesh=mesh,
            top_k=config.num_experts_per_tok,
            scoring_func="sigmoid",
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts")
        self.experts.swiglu_limit = config.swiglu_limit
        self.shared_experts = Glm5NextMLP(config.hidden_size,
                                          config.moe_intermediate_size *
                                          config.n_shared_experts,
                                          config.swiglu_limit,
                                          dtype,
                                          rng,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.shared_experts")
        self.mesh = mesh
        # One all-reduce for the routed and shared experts together: both
        # return partial sums over the same tensor-parallel axes, which are
        # added and then reduced once (half the bytes of two all-reduces).
        self.reduce_axes = _fused_reduce_axes(self.experts,
                                              self.shared_experts, mesh)
        if self.reduce_axes:
            self.shared_experts.down_proj.quant_method.linear_config.defer_all_reduce = True

    def __call__(self,
                 x: jax.Array,
                 num_valid_tokens: Optional[jax.Array] = None) -> jax.Array:
        logits = self.gate(x)
        # GLM5_MOE_TOKEN_SPLITS > 1 runs a prefill-sized step's routed experts
        # as that many independent token slices, so one slice's SparseCore
        # combine can overlap the next one's GMMs on the TensorCore.
        T, n = x.shape[0], envs.GLM5_MOE_TOKEN_SPLITS
        if n < 2 or T < MOE_SPLIT_MIN_TOKENS or T % (16 * n):
            n = 1
        parts = []
        for c in range(n):
            lo = c * (T // n)
            nv = num_valid_tokens
            if nv is not None and n > 1:
                nv = jnp.clip(nv - lo, 0, T // n)
            routed, _ = self.experts(
                x[lo:lo + T // n],
                logits[lo:lo + T // n],
                e_score_correction_bias=self.gate.e_score_correction_bias[...],
                num_valid_tokens=nv,
                defer_all_reduce=bool(self.reduce_axes))
            parts.append(routed)
        routed = jnp.concatenate(parts) if n > 1 else parts[0]
        out = (routed * self.routed_scaling_factor).astype(
            x.dtype) + self.shared_experts(x)
        if self.reduce_axes:
            out = jax.shard_map(
                lambda a: jax.lax.psum(a, self.reduce_axes),
                mesh=self.mesh,
                in_specs=P(),
                out_specs=P(),
                check_vma=False,
            )(out)
        return out


class Glm5NextIndexer(JaxModule):
    """DSA indexer with k-pool compression (`self_attn.indexer`): per-token
    keys (LayerNorm'd) and pooling gates, per-head queries from the MLA query
    residual and per-head weights. Replicated over TP; unquantized."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, prefix: str):
        D, qr = config.hidden_size, config.q_lora_rank
        IH, ihd, kp = (config.index_n_heads, config.index_head_dim,
                       config.index_kpool)
        self.num_heads, self.head_dim = IH, ihd

        def linear(i, o, name):
            return JaxLinear(i,
                             o,
                             use_bias=False,
                             dtype=dtype,
                             kernel_init=nnx.with_partitioning(
                                 init_fn, (None, None)),
                             rngs=rng,
                             quant_config=None,
                             prefix=f"{prefix}.{name}")

        self.wq_b = JaxEinsum("TQ,QNH->TNH", (qr, IH, ihd),
                              dtype=dtype,
                              param_dtype=dtype,
                              kernel_init=nnx.with_partitioning(
                                  init_fn, (None, None, None)),
                              rngs=rng,
                              quant_config=None,
                              prefix=f"{prefix}.wq_b")
        self.wq_b.weight.set_metadata(
            "weight_loader",
            functools.partial(load_nnx_param_from_reshaped_torch,
                              reshape_dims=(IH, ihd, qr),
                              permute_dims=(2, 0, 1),
                              param_name=f"{prefix}.wq_b.weight"))
        self.wk = linear(D, ihd, "wk")
        self.k_norm = JaxLayerNorm(ihd,
                                   epsilon=1e-6,
                                   dtype=dtype,
                                   param_dtype=dtype,
                                   rngs=rng)
        self.weights_proj = linear(D, IH, "weights_proj")
        # [kpool, ihd] as stored (no transpose); the gate is stored [ihd, D].
        self.index_kpool_compress_ape = nnx.Param(
            init_fn(rng.params(), (kp, ihd), jnp.float32),
            out_sharding=(None, None),
            eager_sharding=False,
            weight_loader=functools.partial(
                load_nnx_param_from_reshaped_torch,
                permute_dims=(0, 1),
                param_name=f"{prefix}.index_kpool_compress_ape"))
        self.index_kpool_compress_gate = nnx.Param(init_fn(
            rng.params(), (D, ihd), dtype),
                                                   out_sharding=(None, None),
                                                   eager_sharding=False)

    def __call__(self, x: jax.Array, q_resid: jax.Array):
        """-> (keys [T, ihd], gates [T, ihd], queries [T, IH, ihd] f32,
        head weights [T, IH] f32, already scaled by IH^-0.5)."""
        f32 = jnp.float32
        k = self.k_norm(self.wk(x))
        gate = x @ self.index_kpool_compress_gate[...]
        q = self.wq_b(q_resid).astype(f32)
        # f32 head weights: bf16 flips near-tie pool rankings at long context
        w = jnp.einsum("td,dh->th",
                       x.astype(f32),
                       self.weights_proj.weight[...].astype(f32),
                       precision=jax.lax.Precision.HIGHEST)
        return k, gate, q, w * self.num_heads**-0.5


class Glm5NextMLA(JaxModule):
    """NoPE multi-head latent attention with the key/value up-projection
    absorbed (attention runs on the normalized latents, K = V = c) and the
    k-pool sparse indexer choosing each query's tokens
    (`layers/common/sparse_mla_indexer.py`)."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 quant_config, prefix: str):
        D = config.hidden_size
        H, qk, dv = (config.num_attention_heads, config.qk_nope_head_dim,
                     config.v_head_dim)
        qr, kvr = config.q_lora_rank, config.kv_lora_rank
        assert config.qk_rope_head_dim == 0
        self.num_heads, self.qk, self.dv, self.kvr = H, qk, dv, kvr
        self.scale = qk**-0.5
        self.mesh = mesh
        heads = ShardingAxisName.ATTN_HEAD

        def proj(einsum, shape, sharding, name, quant=True):
            return JaxEinsum(einsum,
                             shape,
                             dtype=dtype,
                             param_dtype=dtype,
                             kernel_init=nnx.with_partitioning(
                                 init_fn, sharding),
                             rngs=rng,
                             quant_config=quant_config if quant else None,
                             prefix=f"{prefix}.{name}")

        def norm(n, name):
            return JaxRmsNorm(n,
                              epsilon=config.rms_norm_eps,
                              dtype=dtype,
                              param_dtype=dtype,
                              scale_init=nnx.with_partitioning(
                                  init_fn, (None, )),
                              rngs=rng,
                              prefix=f"{prefix}.{name}")

        self.q_a_proj = proj("TD,DQ->TQ", (D, qr), (None, None), "q_a_proj")
        self.q_a_layernorm = norm(qr, "q_a_layernorm")
        self.q_b_proj = proj("TQ,QNH->TNH", (qr, H, qk), (None, heads, None),
                             "q_b_proj")
        # Unless a quant method (FP8) brought its own 2-D loader.
        if not hasattr(self.q_b_proj.weight, "weight_loader"):
            self.q_b_proj.weight.set_metadata(
                "weight_loader",
                functools.partial(load_nnx_param_from_reshaped_torch,
                                  reshape_dims=(H, qk, qr),
                                  permute_dims=(2, 0, 1),
                                  param_name=f"{prefix}.q_b_proj.weight"))
        self.kv_a_proj_with_mqa = proj("TD,DR->TR", (D, kvr), (None, None),
                                       "kv_a_proj_with_mqa")
        self.kv_a_layernorm = norm(kvr, "kv_a_layernorm")
        # kv_b [kvr, H, qk + dv] (unquantized): the key part is absorbed into
        # the queries, the value part applied to the attention output.
        self.kv_b_proj = proj("TR,RNH->TNH", (kvr, H, qk + dv),
                              (None, heads, None),
                              "kv_b_proj",
                              quant=False)
        self.kv_b_proj.weight.set_metadata(
            "weight_loader",
            functools.partial(load_nnx_param_from_reshaped_torch,
                              reshape_dims=(H, qk + dv, kvr),
                              permute_dims=(2, 0, 1),
                              param_name=f"{prefix}.kv_b_proj.weight"))
        self.o_proj = proj("TNH,NHD->TD", (H, dv, D), (heads, None, None),
                           "o_proj")
        self.index_topk = config.index_topk
        self.indexer = Glm5NextIndexer(config, dtype, rng, f"{prefix}.indexer")

    def project(self, x: jax.Array):
        """x [T, D] -> (q_lat [T, H, kvr], latents c [T, kvr], q_resid [T,
        q_lora])."""
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid)  # [T, H, qk]
        c = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        w_k = self.kv_b_proj.weight[...][..., :self.qk]  # [kvr, H, qk]
        q_lat = jnp.einsum("thq,rhq->thr", q, w_k)
        return q_lat, c, q_resid

    def output(self, o_lat: jax.Array) -> jax.Array:
        w_v = self.kv_b_proj.weight[...][..., self.qk:]  # [kvr, H, dv]
        return self.o_proj(jnp.einsum("thr,rhv->thv", o_lat, w_v))

    def __call__(self, kv_cache, x: jax.Array, attention_metadata):
        """kv_cache: ((latent pages, pooled keys), (indexer tail, ));
        attention_metadata: (this layer's, the indexer tail's). Returns (new
        kv_cache, y [T, D])."""
        md, tail_md = attention_metadata
        (latent, pooled), (tail, ) = kv_cache
        q_lat, c, q_resid = self.project(x)
        k, gate, qi, wts = self.indexer(x, q_resid)
        latent, pooled, tail, o_lat = run_sparse_mla_step(
            latent,
            pooled,
            tail,
            q_lat,
            c,
            k,
            gate,
            self.indexer.index_kpool_compress_ape[...],
            qi,
            wts,
            md.input_positions,
            md.block_tables,
            md.seq_lens,
            md.query_start_loc,
            md.request_distribution,
            tail_md.mamba_state_indices,
            mesh=self.mesh,
            index_topk=self.index_topk,
            scale=self.scale,
            kvr=self.kvr)
        return ((latent, pooled), (tail, )), self.output(o_lat)


def _layer_metadata(attention_metadata, name: str) -> AttentionMetadata:
    """A layer's metadata: hybrid models get one entry per layer
    (GroupedAttentionMetadata), single-group runs one shared object."""
    if isinstance(attention_metadata, dict):
        return attention_metadata[name]
    return attention_metadata


class Glm5NextDecoderLayer(JaxModule):
    """mHC site -> attention (KDA or MLA) -> mHC site -> MLP / MoE -> mHC
    residual update."""

    def __init__(self, config, layer_idx: int, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh, quant_config, prefix: str):
        self.layer_idx = layer_idx
        self.is_kda = config.layer_types[layer_idx] == "linear_attention"
        self.is_moe = config.mlp_layer_types[layer_idx] == "sparse"
        self.mesh = mesh
        D, hc = config.hidden_size, config.hc_mult
        M = (2 + hc) * hc
        self.mhc_cfg = mhc.MHCConfig(hc=hc,
                                     sinkhorn_iters=config.hc_sinkhorn_iters,
                                     hc_eps=config.hc_eps,
                                     rms_eps=config.rms_norm_eps)
        self.use_mhc_kernel = mhc.mhc_kernel_available(dtype)

        def norm(name):
            return JaxRmsNorm(D,
                              epsilon=config.rms_norm_eps,
                              dtype=dtype,
                              param_dtype=dtype,
                              scale_init=nnx.with_partitioning(
                                  init_fn, (None, )),
                              rngs=rng,
                              prefix=f"{prefix}.{name}")

        def hc_param(shape, name):
            return nnx.Param(init_fn(rng.params(), shape, jnp.float32),
                             out_sharding=(None, ) * len(shape),
                             eager_sharding=False,
                             weight_loader=functools.partial(
                                 load_nnx_param_from_reshaped_torch,
                                 permute_dims=tuple(range(len(shape))),
                                 param_name=f"{prefix}.{name}"))

        self.input_layernorm = norm("input_layernorm")
        self.post_attention_layernorm = norm("post_attention_layernorm")
        # mHC parameters in the checkpoint layout: fn [(2 + hc) * hc, hc * D].
        for site in ("attn", "ffn"):
            setattr(self, f"hc_{site}_fn",
                    hc_param((M, hc * D), f"hc_{site}_fn"))
            setattr(self, f"hc_{site}_base", hc_param((M, ),
                                                      f"hc_{site}_base"))
            setattr(self, f"hc_{site}_scale",
                    hc_param((3, ), f"hc_{site}_scale"))
        if self.is_kda:
            self.self_attn = KimiDeltaAttention(config,
                                                dtype,
                                                rng,
                                                mesh,
                                                prefix=f"{prefix}.self_attn")
        else:
            self.self_attn = Glm5NextMLA(config, dtype, rng, mesh,
                                         quant_config, f"{prefix}.self_attn")
        if self.is_moe:
            self.mlp = Glm5NextMoE(config, dtype, rng, mesh, quant_config,
                                   f"{prefix}.mlp")
        else:
            self.mlp = Glm5NextMLP(config.hidden_size,
                                   config.intermediate_size,
                                   config.swiglu_limit,
                                   dtype,
                                   rng,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")

    def _site(self, name):
        return (getattr(self, f"hc_{name}_fn")[...],
                getattr(self, f"hc_{name}_scale")[...],
                getattr(self, f"hc_{name}_base")[...])

    def __call__(self, kv_cache, streams: jax.Array,
                 attention_metadata: AttentionMetadata):
        """streams [T, hc, D] -> (new kv_cache, new streams)."""
        kw = dict(use_kernel=self.use_mhc_kernel, mesh=self.mesh)
        # named scopes label the ops for profiles
        attn = "mla" if isinstance(self.self_attn, Glm5NextMLA) else "kda"
        ffn = "moe" if isinstance(self.mlp, Glm5NextMoE) else "mlp"
        with jax.named_scope("mhc"):
            post, comb, h = mhc.mhc_pre(streams, *self._site("attn"),
                                        self.mhc_cfg, **kw)
        with jax.named_scope(attn):
            kv_cache, h = self.self_attn(kv_cache, self.input_layernorm(h),
                                         attention_metadata)
        with jax.named_scope("mhc"):
            streams, post, comb, h = mhc.mhc_post_pre(h, streams, post, comb,
                                                      *self._site("ffn"),
                                                      self.mhc_cfg, **kw)
        with jax.named_scope(ffn):
            h = self.post_attention_layernorm(h)
            if ffn == "moe":
                h = self.mlp(h, _num_valid_tokens(attention_metadata,
                                                  self.mesh))
            else:
                h = self.mlp(h)
        with jax.named_scope("mhc"):
            return kv_cache, mhc.mhc_post(h, streams, post, comb, **kw)


def _num_valid_tokens(attention_metadata, mesh: Mesh) -> Optional[jax.Array]:
    """Number of real (unpadded) tokens in the step, or None under attention
    DP, where each rank's padding is interleaved with the next rank's tokens.

    The runner pads a step to a token bucket (at least 16 tokens), and the
    padding rows would otherwise route to arbitrary experts: at batch 1 that is
    up to 16x the expert weights a decode step has to read."""
    if isinstance(attention_metadata, tuple):  # MLA: (latent, indexer tail)
        attention_metadata = attention_metadata[0]
    if get_mesh_shape_product(mesh, ShardingAxisName.ATTN_DATA) > 1:
        return None
    qsl = attention_metadata.query_start_loc
    # the runner repeats the end offset past the last request; max() also
    # covers zero padding
    return None if qsl is None else jnp.max(qsl)


class Glm5NextModel(JaxModule):

    def __init__(self,
                 vllm_config,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model"):
        model_config = vllm_config.model_config
        config = model_config.hf_text_config
        dtype = model_config.dtype
        self.config = config
        quant_config = vllm_config.quant_config
        self.embed_tokens = JaxEmbed(
            num_embeddings=model_config.get_vocab_size(),
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            prefix=f"{prefix}.embed_tokens")
        self.layers = nnx.List([
            Glm5NextDecoderLayer(config, i, dtype, rng, mesh, quant_config,
                                 f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = JaxRmsNorm(config.hidden_size,
                               epsilon=config.rms_norm_eps,
                               dtype=dtype,
                               param_dtype=dtype,
                               scale_init=nnx.with_partitioning(
                                   init_fn, (None, )),
                               rngs=rng,
                               prefix=f"{prefix}.norm")

    def __call__(self,
                 kv_caches,
                 input_ids,
                 attention_metadata,
                 layer_name_to_kv_cache: dict,
                 inputs_embeds=None):
        x = (self.embed_tokens(input_ids)
             if inputs_embeds is None else inputs_embeds)
        hc = self.config.hc_mult
        streams = jnp.broadcast_to(x[:, None, :],
                                   (x.shape[0], hc, x.shape[-1]))
        kv_caches = list(kv_caches)
        for i, layer in enumerate(self.layers):
            name = glm5_next_kv_cache.layer_name(i)
            idx = layer_name_to_kv_cache[name]
            md = _layer_metadata(attention_metadata, name)
            if layer.is_kda:
                kv_caches[idx], streams = layer(kv_caches[idx], streams, md)
                continue
            tail_name = glm5_next_kv_cache.indexer_tail_name(i)
            tidx = layer_name_to_kv_cache[tail_name]
            (kv_caches[idx], kv_caches[tidx]), streams = layer(
                (kv_caches[idx], kv_caches[tidx]), streams,
                (md, _layer_metadata(attention_metadata, tail_name)))
        return kv_caches, self.norm(mhc.mhc_head(streams))


class Glm5NextForCausalLM(JaxModule, LoadableWithIterator):
    """GLM-5.3-Flash text model. Checkpoint names come as
    `model.language_model.*` (multimodal checkpoint) or `model.*`."""

    def __init__(self, vllm_config, rng_key: jax.Array, mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self.mesh = mesh
        rng = nnx.Rngs(rng_key)
        model_config = vllm_config.model_config
        config = model_config.hf_text_config
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "pipeline parallelism is not supported for GLM-5.3-Flash")
        self.model = Glm5NextModel(vllm_config, rng, mesh, prefix="model")
        self.lm_head = JaxLmHead(hidden_size=config.hidden_size,
                                 vocab_size=model_config.get_vocab_size(),
                                 dtype=model_config.dtype,
                                 param_dtype=model_config.dtype,
                                 rngs=rng,
                                 prefix="lm_head")

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors=None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array], None]:
        kv_caches, x = self.model(kv_caches, input_ids, attention_metadata,
                                  dict(_layer_name_to_kv_cache), inputs_embeds)
        return kv_caches, x, [], None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, Any]]) -> set:
        # the vision tower is built by Glm5NextForConditionalGeneration only
        return self._load_weights(weights, skip_prefixes=["model.visual."])

    def _load_weights(self, weights, skip_prefixes) -> set:
        num_layers = self.model.config.num_hidden_layers
        mapper = WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "model."})

        def keep(name: str) -> bool:
            # MTP (NextN) layers sit after the decoder layers and are not
            # built yet.
            m = re.match(r"model\.layers\.(\d+)\.", name)
            return not (m and int(m.group(1)) >= num_layers)

        weights = ((n, w) for n, w in mapper.apply(weights) if keep(n))
        loader = JaxAutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)


class Glm5NextForConditionalGeneration(Glm5NextForCausalLM):
    """The checkpoint's architecture: the text model plus the vision tower
    (`model.visual`), which turns each image or video grid into text
    embeddings for its `<|image|>` tokens. Without vision inputs enabled
    (`language_model_only`, or no image/video allowed per prompt) the tower
    isn't built and its weights are skipped."""

    supports_multimodal = True

    def __init__(self, vllm_config, rng_key: jax.Array, mesh: Mesh) -> None:
        super().__init__(vllm_config, rng_key, mesh)
        hf_config = vllm_config.model_config.hf_config
        self.vision_config = getattr(hf_config, "vision_config", None)
        self.image_token_id = getattr(hf_config, "image_token_id", None)
        self.has_vision = (self.vision_config is not None
                           and _vision_enabled(vllm_config.model_config))
        if self.has_vision:
            self.model.visual = Glm5NextVisionModel(
                self.vision_config, vllm_config.model_config.dtype,
                nnx.Rngs(jax.random.fold_in(rng_key, 1)), mesh)

    def load_weights(self, weights: Iterable[Tuple[str, Any]]) -> set:
        if self.has_vision:
            return self._load_weights(weights, skip_prefixes=[])
        return super().load_weights(weights)

    # -- multimodal interface (models/common/interface.py) --

    def embed_input_ids(
            self,
            input_ids: jax.Array,
            multimodal_embeddings: Optional[jax.Array] = None,
            *,
            is_multimodal: Optional[jax.Array] = None) -> jax.Array:
        """Token embeddings with the vision embeddings written over the
        image tokens, in order (video frames use the image token too)."""
        del is_multimodal
        embeds = self.model.embed_tokens(input_ids)
        if multimodal_embeddings is not None and multimodal_embeddings.shape[
                0] > 0:
            embeds = merge_multimodal_embeddings(
                input_ids, embeds, multimodal_embeddings.astype(embeds.dtype),
                self.image_token_id)
        return embeds

    @functools.partial(jax.jit, static_argnames=("grid", ))
    def encode_grid(self, patches: jax.Array, grid: tuple) -> jax.Array:
        """One image or video: patches [t * h * w, 1176] -> its embeddings
        [t * h * w / 4, hidden] (one compile per grid)."""
        return self.model.visual(patches, grid)

    def embed_multimodal(self, **kwargs) -> Tuple[jax.Array, ...]:
        """One embedding array per image / video item, in the kwargs' order
        (`pixel_values` + `image_grid_thw`, `pixel_values_videos` +
        `video_grid_thw`)."""
        if not self.has_vision:
            return ()
        out = []
        for pv_key, grid_key in (("pixel_values", "image_grid_thw"),
                                 ("pixel_values_videos", "video_grid_thw")):
            pixels, grids = kwargs.get(pv_key), kwargs.get(grid_key)
            if pixels is None:
                continue
            pixels, grids = _as_patches(pixels), _as_grids(grids)
            if self.mesh is not None:  # host -> replicated over the mesh
                pixels = jax.device_put(
                    pixels, jax.sharding.NamedSharding(self.mesh, P()))
            start = 0
            for grid in grids:
                n = int(np.prod(grid))
                out.append(self.encode_grid(pixels[start:start + n], grid))
                start += n
            assert start == pixels.shape[0], (start, pixels.shape)
        return tuple(out)

    def precompile_vision_encoder(self, run_compilation_fn) -> None:
        """Compiles the encoder for the image sizes in
        `additional_config["vision_warmup_config"]["image_shapes"]` ([[H, W]]
        in pixels, multiples of 28); other sizes compile on first use."""
        if not self.has_vision:
            return
        extra = getattr(self.vllm_config, "additional_config", None) or {}
        shapes = extra.get("vision_warmup_config", {}).get("image_shapes", [])
        vc = self.vision_config
        unit = vc.patch_size * vc.spatial_merge_size
        width = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
        for hw in shapes:
            if (len(hw) != 2 or hw[0] % unit or hw[1] % unit):
                logger.warning("skipping vision warm-up shape %s", hw)
                continue
            grid = (1, hw[0] // vc.patch_size, hw[1] // vc.patch_size)
            patches = jnp.zeros((int(np.prod(grid)), width),
                                self.vllm_config.model_config.dtype)
            run_compilation_fn("vision_encoder",
                               functools.partial(self.encode_grid, grid=grid),
                               patches,
                               image_shape=hw)


def _vision_enabled(model_config) -> bool:
    mm = getattr(model_config, "multimodal_config", None)
    if mm is None:
        return False
    if getattr(mm, "language_model_only", False):
        return False
    return any(mm.get_limit_per_prompt(m) > 0 for m in ("image", "video"))


def _as_patches(x) -> jax.Array:
    if isinstance(x, (list, tuple)):
        x = torch.cat([torch.as_tensor(a) for a in x])
    if isinstance(x, torch.Tensor):
        x = x.to(torch.float32).numpy()
    return jnp.asarray(np.asarray(x).reshape(-1, np.asarray(x).shape[-1]))


def _as_grids(g) -> Tuple[Tuple[int, int, int], ...]:
    if isinstance(g, (list, tuple)) and g and isinstance(g[0], torch.Tensor):
        g = torch.cat([torch.as_tensor(a).reshape(-1, 3) for a in g])
    if isinstance(g, torch.Tensor):
        g = g.tolist()
    return tuple(tuple(int(v) for v in row) for row in np.reshape(g, (-1, 3)))
