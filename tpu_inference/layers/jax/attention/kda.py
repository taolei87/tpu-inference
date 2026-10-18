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
"""KDA (Kimi Delta Attention) linear-attention layer, as in GLM-5.3-Flash.

    q, k, v = x W_q, x W_k, x W_v                (per-head, then conv + SiLU)
    g       = lb * sigmoid(exp(A_log) * (x W_fa W_fb + dt_bias))
              (or -exp(A_log) * softplus(...) without a lower bound)
    beta    = sigmoid(x W_b)
    o       = RMSNorm(KDA(q, k, v, g, beta)) * sigmoid(x W_ga W_gb)   (per head)
    y       = o W_o

Parameter names follow the checkpoint (`self_attn.{q,k,v,o,b,f_a,f_b,g_a,
g_b}_proj.weight`, `{q,k,v}_conv1d.weight`, `A_log`, `dt_bias`,
`o_norm.weight`); every tensor is unquantized. Heads are sharded on ATTN_HEAD;
the recurrent core is `layers/common/kda_attention.run_jax_kda_attention`.
"""
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.kda_attention import run_jax_kda_attention
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.models.jax.utils.weight_utils import \
    load_nnx_param_from_reshaped_torch

init_fn = nnx.initializers.uniform()


def _param(rngs: nnx.Rngs, shape, dtype, sharding, **loader_kwargs):
    kwargs = {}
    if loader_kwargs:
        kwargs["weight_loader"] = functools.partial(
            load_nnx_param_from_reshaped_torch, **loader_kwargs)
    return nnx.Param(init_fn(rngs.params(), shape, dtype),
                     out_sharding=sharding,
                     eager_sharding=False,
                     **kwargs)


class _Weight(JaxModule):
    """Holds one checkpoint tensor named `<module>.weight`."""

    def __init__(self, rngs: nnx.Rngs, shape, dtype, sharding,
                 **loader_kwargs):
        self.weight = _param(rngs, shape, dtype, sharding, **loader_kwargs)


class KimiDeltaAttention(JaxModule):

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = ""):
        """`config`: a text config with `hidden_size`, `linear_num_heads`,
        `linear_head_dim`, `linear_conv_kernel_dim`, `linear_lower_bound`
        (None: softplus gate) and `rms_norm_eps`."""
        D = config.hidden_size
        H, d = config.linear_num_heads, config.linear_head_dim
        K = config.linear_conv_kernel_dim
        self.num_heads, self.head_dim, self.kernel_size = H, d, K
        self.gate_lower_bound = config.linear_lower_bound
        self.eps = config.rms_norm_eps
        self.mesh = mesh
        heads = ShardingAxisName.ATTN_HEAD

        def proj(einsum, shape, sharding, name):
            return JaxEinsum(einsum,
                             shape,
                             dtype=dtype,
                             param_dtype=dtype,
                             kernel_init=nnx.with_partitioning(
                                 init_fn, sharding),
                             rngs=rng,
                             quant_config=None,
                             prefix=f"{prefix}.{name}")

        # [D, H, d]; the default loader maps the checkpoint's [H * d, D].
        self.q_proj = proj("TD,DNH->TNH", (D, H, d), (None, heads, None),
                           "q_proj")
        self.k_proj = proj("TD,DNH->TNH", (D, H, d), (None, heads, None),
                           "k_proj")
        self.v_proj = proj("TD,DNH->TNH", (D, H, d), (None, heads, None),
                           "v_proj")
        self.o_proj = proj("TNH,NHD->TD", (H, d, D), (heads, None, None),
                           "o_proj")
        self.b_proj = proj("TD,DN->TN", (D, H), (None, heads), "b_proj")
        # Low-rank gate projections: D -> d (replicated) -> H * d (by heads).
        self.f_a_proj = proj("TD,DK->TK", (D, d), (None, None), "f_a_proj")
        self.f_b_proj = proj("TK,KNH->TNH", (d, H, d), (None, heads, None),
                             "f_b_proj")
        self.g_a_proj = proj("TD,DK->TK", (D, d), (None, None), "g_a_proj")
        self.g_b_proj = proj("TK,KNH->TNH", (d, H, d), (None, heads, None),
                             "g_b_proj")
        for name in ("f_b_proj", "g_b_proj"):  # checkpoint [H * d, d]
            getattr(self, name).weight.set_metadata(
                "weight_loader",
                functools.partial(load_nnx_param_from_reshaped_torch,
                                  reshape_dims=(H, d, d),
                                  permute_dims=(2, 0, 1),
                                  param_name=f"{prefix}.{name}.weight"))
        self.A_log = _param(rng, (H, ), jnp.float32, (heads, ))
        self.dt_bias = _param(rng, (H, d),
                              jnp.float32, (heads, None),
                              reshape_dims=(H, d),
                              permute_dims=(0, 1),
                              param_name=f"{prefix}.dt_bias")
        # Depthwise causal conv taps per channel: checkpoint [H * d, 1, K].
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            setattr(
                self, name,
                _Weight(rng, (H, d, K),
                        dtype, (heads, None, None),
                        reshape_dims=(H, d, K),
                        permute_dims=(0, 1, 2),
                        param_name=f"{prefix}.{name}.weight"))
        self.o_norm = _Weight(rng, (d, ), dtype, (None, ))

    def __call__(
        self,
        kv_cache: Tuple[jax.Array, jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
        """x [T, D] (input-normed tokens of the ragged batch); kv_cache
        (conv_state [slots, K-1, 3C], recurrent_state [slots, H, d, d]).
        Returns (new kv_cache, y [T, D])."""
        md = attention_metadata
        f32 = jnp.float32
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # [T, H, d]
        f = self.f_b_proj(self.f_a_proj(x)).astype(f32) + self.dt_bias.value
        decay_rate = jnp.exp(self.A_log.value.astype(f32))[:, None]
        if self.gate_lower_bound is not None:
            g = self.gate_lower_bound * jax.nn.sigmoid(decay_rate * f)
        else:
            g = -decay_rate * jax.nn.softplus(f)
        beta = jax.nn.sigmoid(self.b_proj(x).astype(f32))  # [T, H]
        gate = self.g_b_proj(self.g_a_proj(x))  # [T, H, d]
        conv_state, recurrent_state = kv_cache
        new_cache, core = run_jax_kda_attention(q,
                                                k,
                                                v,
                                                g,
                                                beta,
                                                conv_state,
                                                recurrent_state,
                                                self.q_conv1d.weight.value,
                                                self.k_conv1d.weight.value,
                                                self.v_conv1d.weight.value,
                                                md.mamba_state_indices,
                                                md.query_start_loc,
                                                md.request_distribution,
                                                md.seq_lens,
                                                mesh=self.mesh)
        # Gated RMSNorm over each head (fp32, sigmoid gate).
        o = core * jax.lax.rsqrt(
            jnp.mean(core * core, -1, keepdims=True) + self.eps)
        o = o * self.o_norm.weight.value.astype(f32)
        o = o * jax.nn.sigmoid(gate.astype(f32))
        return new_cache, self.o_proj(o.astype(x.dtype))
