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

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.nope_mla_attention import (dense_nope_mla,
                                                            write_latents)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.moe.moe import JaxRoutedExperts
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.models.jax.utils.weight_utils import \
    load_nnx_param_from_reshaped_torch

init_fn = nnx.initializers.uniform()


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
        mlp = ShardingAxisName.MLP_TENSOR

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
                          self.weight.value,
                          precision=jax.lax.Precision.HIGHEST)


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

    def __call__(self, x: jax.Array) -> jax.Array:
        routed, _ = self.experts(
            x,
            self.gate(x),
            e_score_correction_bias=self.gate.e_score_correction_bias.value)
        return (routed * self.routed_scaling_factor).astype(
            x.dtype) + self.shared_experts(x)


class Glm5NextMLA(JaxModule):
    """NoPE multi-head latent attention with the key/value up-projection
    absorbed: attention runs on the normalized latents (K = V = c).

    kv_cache: the MLA layer's (latent pages, pooled indexer keys); the pooled
    keys are used by the sparse indexer (M4), dense attention reads only the
    latents.
    """

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
        if self.q_b_proj.quant_method is None:  # FP8 brings its own loader
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

    def project(self, x: jax.Array):
        """x [T, D] -> (q_lat [T, H, kvr], latents c [T, kvr], q_resid [T,
        q_lora])."""
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid)  # [T, H, qk]
        c = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        w_k = self.kv_b_proj.weight.value[..., :self.qk]  # [kvr, H, qk]
        q_lat = jnp.einsum("thq,rhq->thr", q, w_k)
        return q_lat, c, q_resid

    def output(self, o_lat: jax.Array) -> jax.Array:
        w_v = self.kv_b_proj.weight.value[..., self.qk:]  # [kvr, H, dv]
        return self.o_proj(jnp.einsum("thr,rhv->thv", o_lat, w_v))

    def __call__(self, kv_cache, x: jax.Array,
                 attention_metadata: AttentionMetadata):
        """Dense causal attention (exact while a request's context fits in the
        indexer's top-k). Returns (new kv_cache, y [T, D])."""
        md = attention_metadata
        latent, pooled = kv_cache
        q_lat, c, _ = self.project(x)
        latent = write_latents(latent, c, md.input_positions, md.block_tables,
                               md.query_start_loc, md.request_distribution)
        o_lat = dense_nope_mla(q_lat,
                               latent,
                               md.input_positions,
                               md.block_tables,
                               md.seq_lens,
                               md.query_start_loc,
                               md.request_distribution,
                               scale=self.scale,
                               kvr=self.kvr)
        return (latent, pooled), self.output(o_lat)
