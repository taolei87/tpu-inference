# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM-5.3-Flash MTP (NextN) draft model: the checkpoint's layer
`model.layers.{num_hidden_layers}` (one layer).

    x = eh_proj(cat(enorm(embed(t + 1)), hnorm(h_t)))   (embed zeroed at
                                                         position 0, as vLLM)
    x += MLA(input_layernorm(x))     NoPE MLA with its own k-pool indexer
    x += MoE(post_attention_layernorm(x))
    h = shared_head.norm(x)          -> lm_head; also the next step's h_t

h_t is the target's post-final-norm hidden state at position t. There is no
mHC. The MLA has its own latent, pooled-key and indexer-tail caches, named
like a decoder layer at index num_hidden_layers.
"""

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from vllm.model_executor.models.utils import WeightsMapper

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxLinear, JaxLmHead
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.models.jax.glm5_next import (Glm5NextMLA, Glm5NextMoE,
                                                _layer_metadata,
                                                _num_valid_tokens)
from tpu_inference.models.jax.utils.weight_utils import (JaxAutoWeightsLoader,
                                                         LoadableWithIterator)
from tpu_inference.runner import glm5_next_kv_cache

init_fn = nnx.initializers.uniform()


class _Node(JaxModule):
    """A named level of the checkpoint's module tree."""


def mtp_text_config(vllm_config):
    """The text config the MTP layer is built from: the draft model's (vLLM
    renames the draft to glm5_next_mtp) or, without one, the target's."""
    spec = vllm_config.speculative_config
    cfg = (spec.draft_model_config.hf_config
           if spec is not None and spec.draft_model_config is not None else
           vllm_config.model_config.hf_config)
    return getattr(cfg, "text_config", cfg)


class Glm5NextMTPLayer(JaxModule):

    def __init__(self, config, layer_idx: int, dtype, rng: nnx.Rngs,
                 mesh: Mesh, quant_config, prefix: str):
        D = config.hidden_size
        self.layer_idx, self.mesh = layer_idx, mesh

        def norm(name):
            return JaxRmsNorm(D,
                              epsilon=config.rms_norm_eps,
                              dtype=dtype,
                              param_dtype=dtype,
                              scale_init=nnx.with_partitioning(
                                  init_fn, (None, )),
                              rngs=rng,
                              prefix=f"{prefix}.{name}")

        self.enorm = norm("enorm")
        self.hnorm = norm("hnorm")
        self.eh_proj = JaxLinear(2 * D,
                                 D,
                                 use_bias=False,
                                 dtype=dtype,
                                 kernel_init=nnx.with_partitioning(
                                     init_fn, (None, None)),
                                 rngs=rng,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.eh_proj")
        self.input_layernorm = norm("input_layernorm")
        self.post_attention_layernorm = norm("post_attention_layernorm")
        self.self_attn = Glm5NextMLA(config, dtype, rng, mesh, quant_config,
                                     f"{prefix}.self_attn")
        self.mlp = Glm5NextMoE(config, dtype, rng, mesh, quant_config,
                               f"{prefix}.mlp")
        self.shared_head = _Node()
        self.shared_head.norm = norm("shared_head.norm")

    def __call__(self, kv_cache, embeds: jax.Array, h_prev: jax.Array,
                 attention_metadata):
        """embeds [T, D] of the token after each position; h_prev [T, D] the
        target's (or the previous draft step's) hidden state there;
        attention_metadata (MLA's, indexer tail's). Returns (new kv_cache,
        h [T, D] after shared_head.norm)."""
        md = attention_metadata[0]
        embeds = jnp.where((md.input_positions == 0)[:, None], 0, embeds)
        x = self.eh_proj(
            jnp.concatenate(
                [self.enorm(embeds), self.hnorm(h_prev)], -1))
        kv_cache, a = self.self_attn(kv_cache, self.input_layernorm(x),
                                     attention_metadata)
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x),
                         _num_valid_tokens(md, self.mesh))
        return kv_cache, self.shared_head.norm(x)


class Glm5NextMTPModel(JaxModule, LoadableWithIterator):
    """The draft model for `method="mtp"` (Eagle3Proposer): __call__(kv_caches,
    input_ids, hidden_states, attention_metadata, layer_name_to_kvcache_index)
    -> (kv_caches, h, [h], None). The token embedding is shared with the
    target model by the proposer; lm_head is loaded from the checkpoint."""

    def __init__(self, vllm_config, rng_key: jax.Array, mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self.mesh = mesh
        rng = nnx.Rngs(rng_key)
        config = mtp_text_config(vllm_config)
        dtype = vllm_config.model_config.dtype
        self.config = config
        self.mtp_idx = config.num_hidden_layers
        self.model = _Node()
        self.model.embed_tokens = JaxEmbed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            prefix="model.embed_tokens")
        self.model.mtp = Glm5NextMTPLayer(config, self.mtp_idx, dtype, rng,
                                          mesh, vllm_config.quant_config,
                                          f"model.layers.{self.mtp_idx}")
        self.lm_head = JaxLmHead(hidden_size=config.hidden_size,
                                 vocab_size=config.vocab_size,
                                 dtype=dtype,
                                 param_dtype=dtype,
                                 rngs=rng,
                                 prefix="lm_head")

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        hidden_states: jax.Array,
        attention_metadata,
        layer_name_to_kvcache_index: Optional[Sequence[Tuple[str,
                                                             int]]] = None,
        *args,
    ):
        index = dict(layer_name_to_kvcache_index)
        name = glm5_next_kv_cache.layer_name(self.mtp_idx)
        tail_name = glm5_next_kv_cache.indexer_tail_name(self.mtp_idx)
        idx, tidx = index[name], index[tail_name]
        md = (_layer_metadata(attention_metadata, name),
              _layer_metadata(attention_metadata, tail_name))
        kv_caches = list(kv_caches)
        (kv_caches[idx], kv_caches[tidx]), h = self.model.mtp(
            (kv_caches[idx], kv_caches[tidx]),
            self.model.embed_tokens(input_ids), hidden_states, md)
        return kv_caches, h, [h], None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)

    @property
    def draft_layer_names(self) -> Tuple[str, str]:
        """The KV cache names of the MTP layer (its MLA, its indexer tail):
        the proposer takes the draft's block table from their group."""
        return (glm5_next_kv_cache.layer_name(self.mtp_idx),
                glm5_next_kv_cache.indexer_tail_name(self.mtp_idx))

    def load_weights(self, weights: Iterable[Tuple[str, Any]]) -> set:
        """The MTP layer (renamed to `model.mtp.`), the token embedding and
        lm_head; everything else in the checkpoint is skipped. (The embedding
        is loaded so every parameter is materialized; the proposer then finds
        it equal to the target's and shares that one.)"""
        mtp = f"model.layers.{self.mtp_idx}."
        mapper = WeightsMapper(orig_to_new_prefix={
            "model.language_model.": "model.",
        })

        def keep(item):
            name, w = item
            if name.startswith(mtp):
                return ("model.mtp." + name[len(mtp):], w)
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                return item
            return None

        weights = (kw for kw in map(keep, mapper.apply(weights)) if kw)
        return JaxAutoWeightsLoader(self).load_weights(weights)
