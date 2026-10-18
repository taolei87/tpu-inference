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
"""Tiny random GLM-5.3-Flash checkpoints in the real checkpoint's format, for parity tests.

The checkpoint is a directory with `config.json` (a full `Glm5NextConfig`: text + vision) and `model.safetensors`
whose tensor names, layouts and (optionally) FP8 block quantization follow `zai-org/GLM-5.3-Flash` exactly, so the
same directory loads through transformers (`from_pretrained`), the reference reader (`ref_checkpoint`) and the
vLLM / tpu-inference loaders.

    python -m tests.models.jax.glm5_next_ref.make_tiny_ckpt OUT_DIR [--fp8] [--bf16] [--seed N]
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os

import numpy as np

from tests.models.jax.glm5_next_ref import ref_checkpoint as RC

# Small but structurally complete: KDA and MLA layers, dense and sparse MLPs, an indexer with real k-pool settings,
# head dims that keep the per-head math non-trivial, an MTP layer.
TINY_TEXT = dict(
    vocab_size=512,
    hidden_size=64,
    intermediate_size=96,
    moe_intermediate_size=32,
    num_hidden_layers=6,
    layer_types=[
        "linear_attention", "linear_attention", "deepseek_sparse_attention",
        "linear_attention", "deepseek_sparse_attention", "linear_attention"
    ],
    mlp_layer_types=[
        "dense", "sparse", "sparse", "sparse", "sparse", "sparse"
    ],
    indexer_types=["full"] * 6,
    first_k_dense_replace=1,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    moe_router_dtype="float32",
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=48,
    kv_lora_rank=32,
    qk_nope_head_dim=16,
    qk_rope_head_dim=0,
    v_head_dim=16,
    head_dim=0,
    mla_use_nope=True,
    # 8 indexer heads: with very few heads many pools tie at exactly 0 after the ReLU and top-k tie-breaking
    # differs across frameworks.
    index_n_heads=8,
    index_head_dim=16,
    index_topk=64,
    index_kpool=4,
    index_kpool_always_select_tail=True,
    index_kpool_compress=True,
    linear_attn_config={
        "num_heads": 4,
        "head_dim": 16,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0
    },
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    mhc=True,
    rms_norm_eps=1e-5,
    swiglu_limit=10.0,
    max_position_embeddings=4096,
    tie_word_embeddings=False,
    num_nextn_predict_layers=1,
    pad_token_id=0,
    eos_token_id=1,
    index_share_for_mtp_iteration=True,
    indexer_rope_interleave=True,
    qk_head_dim=16,
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    initializer_range=0.02,
    output_router_logits=False,
    router_aux_loss_coef=0.001,
    use_cache=True,
)
TINY_VISION = dict(depth=1,
                   hidden_size=32,
                   num_heads=2,
                   intermediate_size=64,
                   out_hidden_size=64,
                   projection_intermediate_size=96,
                   patch_size=14,
                   image_size=56,
                   spatial_merge_size=2,
                   temporal_patch_size=2,
                   swiglu_limit=10.0,
                   attention_bias=True,
                   attention_dropout=0.0,
                   hidden_act="silu",
                   in_channels=3,
                   initializer_range=0.02,
                   rms_norm_eps=1e-5)
TINY_TOP = dict(image_token_id=500,
                video_token_id=501,
                image_start_token_id=502,
                image_end_token_id=503,
                video_start_token_id=504,
                video_end_token_id=505)

# Tensors kept in f32 when the checkpoint is written in bf16 (gate/state parameters, as in the real checkpoint).
_F32_SUFFIXES = ("A_log", "dt_bias", "e_score_correction_bias",
                 "index_kpool_compress_ape", "hc_attn_fn", "hc_attn_base",
                 "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale")


def tiny_text_dict(**text_overrides) -> dict:
    """The tiny text config as it appears in `config.json` (raw, like the real checkpoint's: MLA layers are
    "deepseek_sparse_attention", which transformers canonicalizes to "indexed_attention" on load)."""
    text = copy.deepcopy(TINY_TEXT)
    text.update(text_overrides)
    n = text["num_hidden_layers"]
    for k in ("layer_types", "mlp_layer_types", "indexer_types"):
        assert len(text[k]) == n, (k, len(text[k]), n)
    lt = text["layer_types"]
    text["linear_attn_config"] = dict(
        text["linear_attn_config"],
        kda_layers=[i for i, t in enumerate(lt) if t == "linear_attention"],
        full_attn_layers=[
            i for i, t in enumerate(lt) if t != "linear_attention"
        ])
    return text


def tiny_config(**text_overrides):
    """A transformers Glm5NextConfig (text + vision) for the tiny model."""
    from transformers import Glm5NextConfig
    return Glm5NextConfig(text_config=tiny_text_dict(**text_overrides),
                          vision_config=TINY_VISION,
                          **TINY_TOP)


def tiny_config_json(dtype: str, **text_overrides) -> dict:
    """`config.json` contents with the same keys as the real checkpoint's (quantization_config added by the
    writer for FP8)."""
    import transformers
    text = dict(tiny_text_dict(**text_overrides),
                model_type="glm5_next_text",
                dtype=dtype)
    return dict(architectures=["Glm5NextForConditionalGeneration"],
                model_type="glm5_next",
                text_config=text,
                vision_config=dict(TINY_VISION, model_type="glm5_next_vision"),
                tie_word_embeddings=False,
                transformers_version=transformers.__version__,
                **TINY_TOP)


def build_hf_model(cfg, seed: int = 0):
    """A randomly initialized transformers Glm5NextForConditionalGeneration (f32, eval) whose gates, router bias,
    k-pool parameters and mHC parameters are perturbed away from their (often degenerate) init values."""
    import torch
    from transformers import Glm5NextForConditionalGeneration
    torch.manual_seed(seed)
    model = Glm5NextForConditionalGeneration(cfg).float().eval()
    g = torch.Generator().manual_seed(seed + 1)

    def normal_(t, std, mean=0.0):
        t.copy_(torch.randn(t.shape, generator=g) * std + mean)

    with torch.no_grad():
        # e_score_correction_bias is a buffer, not a parameter: walk both.
        for name, t in itertools.chain(model.named_parameters(),
                                       model.named_buffers()):
            if name.endswith("e_score_correction_bias"):
                normal_(t, 0.1)
            elif name.endswith("A_log"):
                t.copy_(torch.rand(t.shape, generator=g) * 2.5 - 2.0)
            elif name.endswith("dt_bias"):
                normal_(t, 0.5)
            elif name.endswith("index_kpool_compress_ape"):
                normal_(t, 0.5)
            elif name.endswith("index_kpool_compress_gate"):
                normal_(t, 0.05)
            elif name.endswith("_hc.fn"):
                normal_(t, 0.05)
            elif name.endswith("_hc.base"):
                normal_(t, 0.5)
            elif name.endswith("_hc.scale"):
                normal_(t, 0.3, mean=1.0)
            elif name.endswith("norm.weight") or name.endswith(
                    "layernorm.weight"):
                normal_(t, 0.1, mean=1.0)
    return model


def _random_mtp_tensors(cfg, rng: np.random.Generator) -> dict:
    """Real-naming tensors of the MTP (NextN) layers (transformers does not instantiate them): a plain MLA + indexer
    + MoE layer plus eh_proj / enorm / hnorm / shared_head.norm. Returns {name: float32 numpy}."""
    t = cfg.text_config
    D, H, qk, dv, kvr, qr = (t.hidden_size, t.num_attention_heads,
                             t.qk_nope_head_dim, t.v_head_dim, t.kv_lora_rank,
                             t.q_lora_rank)
    IH, ihd, kp, E, mi = t.index_n_heads, t.index_head_dim, t.index_kpool, t.n_routed_experts, t.moe_intermediate_size

    def lin(o, i):
        return (rng.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    out = {}
    for j in range(t.num_nextn_predict_layers):
        p = f"{RC.PREFIX}layers.{t.num_hidden_layers + j}."
        a, ix, m = p + "self_attn.", p + "self_attn.indexer.", p + "mlp."
        out.update({
            p + "eh_proj.weight":
            lin(D, 2 * D),
            p + "enorm.weight":
            norm(D),
            p + "hnorm.weight":
            norm(D),
            p + "shared_head.norm.weight":
            norm(D),
            p + "input_layernorm.weight":
            norm(D),
            p + "post_attention_layernorm.weight":
            norm(D),
            a + "q_a_proj.weight":
            lin(qr, D),
            a + "q_a_layernorm.weight":
            norm(qr),
            a + "q_b_proj.weight":
            lin(H * qk, qr),
            a + "kv_a_proj_with_mqa.weight":
            lin(kvr, D),
            a + "kv_a_layernorm.weight":
            norm(kvr),
            a + "kv_b_proj.weight":
            lin(H * (qk + dv), kvr),
            a + "o_proj.weight":
            lin(D, H * dv),
            ix + "wq_b.weight":
            lin(IH * ihd, qr),
            ix + "wk.weight":
            lin(ihd, D),
            ix + "k_norm.weight":
            norm(ihd),
            ix + "k_norm.bias":
            (0.1 * rng.standard_normal(ihd)).astype(np.float32),
            ix + "weights_proj.weight":
            lin(IH, D),
            ix + "index_kpool_compress_ape": (0.5 * rng.standard_normal(
                (kp, ihd))).astype(np.float32),
            ix + "index_kpool_compress_gate": (0.05 * rng.standard_normal(
                (ihd, D))).astype(np.float32),
            m + "gate.weight":
            lin(E, D),
            m + "gate.e_score_correction_bias":
            (0.1 * rng.standard_normal(E)).astype(np.float32),
            m + "shared_experts.gate_proj.weight":
            lin(mi, D),
            m + "shared_experts.up_proj.weight":
            lin(mi, D),
            m + "shared_experts.down_proj.weight":
            lin(D, mi),
        })
        for e in range(E):
            out[f"{m}experts.{e}.gate_proj.weight"] = lin(mi, D)
            out[f"{m}experts.{e}.up_proj.weight"] = lin(mi, D)
            out[f"{m}experts.{e}.down_proj.weight"] = lin(D, mi)
    return out


def write_tiny_checkpoint(out_dir: str,
                          seed: int = 0,
                          dtype: str = "float32",
                          fp8: bool = False,
                          with_mtp: bool = True,
                          **text_overrides):
    """Writes `config.json` + `model.safetensors` in the real checkpoint format to `out_dir`.

    dtype: "float32" (exact parity tests) or "bfloat16" (like the real checkpoint; gate/state tensors stay f32).
    fp8: quantize the tensors the real checkpoint quantizes (block 128, e4m3, `*_scale_inv`), and write the matching
    `quantization_config`. Returns (hf_model, config); hf_model holds the unquantized f32 weights."""
    import torch
    from safetensors.torch import save_file
    cfg = tiny_config(**text_overrides)
    model = build_hf_model(cfg, seed)
    layer_types = list(cfg.text_config.layer_types)
    tensors = RC.hf_state_dict_to_real(
        {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }, layer_types)
    if with_mtp:
        rng = np.random.default_rng(seed + 2)
        tensors.update({
            k: torch.from_numpy(v)
            for k, v in _random_mtp_tensors(cfg, rng).items()
        })

    tdt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    out = {}
    for name, t in tensors.items():
        if fp8 and RC.fp8_quantized(name, layer_types):
            q, s = RC.quant_fp8_block(t.float().numpy())
            out[name] = q.contiguous()
            out[name + "_scale_inv"] = torch.from_numpy(s)
        elif name.endswith(_F32_SUFFIXES):
            out[name] = t.float().contiguous()
        else:
            out[name] = t.to(tdt).contiguous()

    os.makedirs(out_dir, exist_ok=True)
    save_file(out,
              os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    cfg_dict = tiny_config_json(dtype, **text_overrides)
    if fp8:
        cfg_dict["quantization_config"] = {
            "activation_scheme":
            "dynamic",
            "fmt":
            "e4m3",
            "quant_method":
            "fp8",
            "weight_block_size": [RC.BLOCK, RC.BLOCK],
            "modules_to_not_convert":
            RC.modules_to_not_convert(list(out), layer_types)
        }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2, sort_keys=True)
    return model, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--no-mtp", action="store_true")
    a = ap.parse_args()
    write_tiny_checkpoint(a.out_dir, a.seed,
                          "bfloat16" if a.bf16 else "float32", a.fp8,
                          not a.no_mtp)
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
