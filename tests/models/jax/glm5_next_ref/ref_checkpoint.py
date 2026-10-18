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
"""GLM-5.3-Flash checkpoint naming, FP8 rules and the reader that turns a checkpoint into `ref_model` params.

The real checkpoint (`zai-org/GLM-5.3-Flash`) stores torch-layout tensors ([out, in]) under these names:

  lm_head.weight, model.language_model.{embed_tokens.weight, norm.weight}
  model.language_model.layers.{i}.
    input_layernorm.weight, post_attention_layernorm.weight
    hc_{attn,ffn}_{fn,base,scale}                       (decoder layers only; not the MTP layer)
    self_attn. (KDA)  {q,k,v,b,f_a,f_b,g_a,g_b,o}_proj.weight, {q,k,v}_conv1d.weight [C,1,K], A_log, dt_bias,
                      o_norm.weight
    self_attn. (MLA)  q_a_proj, q_a_layernorm, q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj,
                      indexer.{wq_b.weight, wk.weight, k_norm.{weight,bias}, weights_proj.weight,
                               index_kpool_compress_ape [kp, ihd], index_kpool_compress_gate [ihd, D]}
    mlp. (dense)      {gate,up,down}_proj.weight
    mlp. (sparse)     gate.weight, gate.e_score_correction_bias, shared_experts.{gate,up,down}_proj.weight,
                      experts.{e}.{gate,up,down}_proj.weight
    (MTP layer, index = num_hidden_layers)  + eh_proj.weight, enorm.weight, hnorm.weight, shared_head.norm.weight
  model.visual.*                                         (vision tower)

FP8 tensors (e4m3, 128x128 blocks) carry `<name>_scale_inv` [ceil(O/128), ceil(I/128)] f32; `fp8_quantized()` is
the rule, checked against the real index in the tests.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

PREFIX = "model.language_model."
BLOCK = 128

# Linear weights that are FP8 in the real checkpoint (everything else is bf16/f32, see
# quantization_config.modules_to_not_convert). Matched against the name after PREFIX + "layers.{i}.".
_FP8_MLA = re.compile(
    r"self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)\.weight$")
_FP8_MLP = re.compile(
    r"mlp\.((shared_experts|experts\.\d+)\.)?(gate|up|down)_proj\.weight$")


def fp8_quantized(name: str, layer_types: list[str]) -> bool:
    """Whether tensor `name` (real naming) is stored as FP8 block-128 in the real checkpoint. `layer_types`:
    text_config.layer_types (the MTP layer, index len(layer_types), is an MLA + MoE layer)."""
    m = re.match(re.escape(PREFIX) + r"layers\.(\d+)\.(.*)$", name)
    if not m:
        return False
    i, rest = int(m.group(1)), m.group(2)
    is_mla = i >= len(layer_types) or layer_types[i] != "linear_attention"
    if _FP8_MLA.match(rest):
        return is_mla  # KDA o_proj is bf16
    return bool(_FP8_MLP.match(rest))


def modules_to_not_convert(names: list[str],
                           layer_types: list[str]) -> list[str]:
    """`quantization_config.modules_to_not_convert` for a checkpoint with these tensor names, in the real
    checkpoint's convention (module paths without the `language_model.` level and without `.weight`)."""
    out = []
    for n in names:
        if fp8_quantized(n, layer_types) or n.endswith("_scale_inv"):
            continue
        mod = n[:-len(".weight")] if n.endswith(".weight") else n
        out.append(mod.replace("model.language_model.", "model.", 1))
    return sorted(set(out))


# ----------------------------------------------------------------------------- FP8 block quantization
def quant_fp8_block(w: np.ndarray, block: int = BLOCK):
    """float [O, I] -> (torch float8_e4m3fn tensor [O, I], scale_inv f32 numpy [ceil(O/b), ceil(I/b)]), such that
    dequant_fp8_block(q.float(), s) ~= w (the checkpoint's `weight` / `weight_scale_inv` pair)."""
    import torch
    n_out, n_in = w.shape
    po, pi = (-n_out) % block, (-n_in) % block
    wp = np.pad(w.astype(np.float32), ((0, po), (0, pi)))
    blk = wp.reshape((n_out + po) // block, block, (n_in + pi) // block, block)
    amax = np.abs(blk).max(axis=(1, 3))
    scale = np.where(amax > 0, amax / 448.0,
                     1.0).astype(np.float32)  # e4m3 max = 448
    q = blk / scale[:, None, :, None]
    q = torch.from_numpy(np.ascontiguousarray(q.reshape(wp.shape))).to(
        torch.float8_e4m3fn)
    return q[:n_out, :n_in], scale


def dequant_fp8_block(w: np.ndarray,
                      scale_inv: np.ndarray,
                      block: int = BLOCK) -> np.ndarray:
    n_out, n_in = w.shape
    s = np.repeat(np.repeat(scale_inv.astype(np.float32), block, axis=0),
                  block,
                  axis=1)[:n_out, :n_in]
    return w.astype(np.float32) * s


# ----------------------------------------------------------------------------- HF module names -> real names
def hf_state_dict_to_real(sd: dict, layer_types: list[str]) -> dict:
    """A transformers `Glm5NextForConditionalGeneration` state_dict (torch tensors) -> {real checkpoint name:
    torch tensor}: the inverse of transformers' `conversion_mapping["glm5_next"]` (per-expert tensors, separate
    q/k/v convs, flat KDA gate and mHC names)."""
    out = {}
    for k, v in sd.items():
        m = re.match(re.escape(PREFIX) + r"layers\.(\d+)\.(.*)$", k)
        if not m:
            out[k] = v
            continue
        i, rest = int(m.group(1)), m.group(2)
        pfx = f"{PREFIX}layers.{i}."
        if rest == "self_attn.conv1d.weight":  # [3C, 1, K] -> q / k / v
            C = v.shape[0] // 3
            for j, c in enumerate("qkv"):
                out[f"{pfx}self_attn.{c}_conv1d.weight"] = v[j * C:(j + 1) *
                                                             C].clone()
        elif rest.startswith("self_attn.forget_gate."):
            out[pfx + "self_attn." + rest[len("self_attn.forget_gate."):]] = v
        elif re.match(r"(attn|ffn)_hc\.(fn|base|scale)$", rest):
            site, part = rest.split("_hc.")
            out[f"{pfx}hc_{site}_{part}"] = v
        elif rest == "mlp.experts.gate_up_proj":  # [E, 2mi, D] -> experts.e.{gate,up}_proj.weight [mi, D]
            mi = v.shape[1] // 2
            for e in range(v.shape[0]):
                out[f"{pfx}mlp.experts.{e}.gate_proj.weight"] = v[
                    e, :mi].clone()
                out[f"{pfx}mlp.experts.{e}.up_proj.weight"] = v[e, mi:].clone()
        elif rest == "mlp.experts.down_proj":  # [E, D, mi]
            for e in range(v.shape[0]):
                out[f"{pfx}mlp.experts.{e}.down_proj.weight"] = v[e].clone()
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------- reader
class CheckpointReader:
    """Reads a checkpoint directory (`model.safetensors.index.json` + shards, or one `model.safetensors`) by real
    tensor name; FP8 tensors are dequantized with their block scales."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            from safetensors import safe_open
            with safe_open(os.path.join(model_dir, "model.safetensors"),
                           framework="pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._open = {}

    def names(self):
        return list(self.weight_map)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def raw(self, name: str):
        from safetensors import safe_open
        fname = self.weight_map[name]
        if fname not in self._open:
            self._open[fname] = safe_open(os.path.join(self.model_dir, fname),
                                          framework="pt")
        return self._open[fname].get_tensor(name)

    def get(self, name: str) -> np.ndarray:
        """float32 numpy array (FP8 dequantized)."""
        import torch
        t = self.raw(name)
        if t.dtype == torch.float8_e4m3fn:
            return dequant_fp8_block(
                t.float().numpy(),
                self.raw(name + "_scale_inv").float().numpy())
        return t.float().numpy()


# ----------------------------------------------------------------------------- real names -> ref_model params
def _lin(r: CheckpointReader, name: str):
    return np.ascontiguousarray(r.get(name).T)  # torch [out, in] -> [in, out]


def _load_mla(r: CheckpointReader, a: str):
    p = {
        "q_a": _lin(r, a + "q_a_proj.weight"),
        "q_a_norm": r.get(a + "q_a_layernorm.weight"),
        "q_b": _lin(r, a + "q_b_proj.weight"),
        "kv_a": _lin(r, a + "kv_a_proj_with_mqa.weight"),
        "kv_a_norm": r.get(a + "kv_a_layernorm.weight"),
        "kv_b": _lin(r, a + "kv_b_proj.weight"),
        "o": _lin(r, a + "o_proj.weight")
    }
    ix = a + "indexer."
    if r.has(ix + "wk.weight"):
        p["indexer"] = {
            "wq_b": _lin(r, ix + "wq_b.weight"),
            "wk": _lin(r, ix + "wk.weight"),
            "k_norm_w": r.get(ix + "k_norm.weight"),
            "k_norm_b": r.get(ix + "k_norm.bias"),
            "weights_proj": _lin(r, ix + "weights_proj.weight"),
            "ape": r.get(ix + "index_kpool_compress_ape"),
            "gate": _lin(r, ix + "index_kpool_compress_gate")
        }
    return p


def _load_kda(r: CheckpointReader, a: str):
    return {
        "q": _lin(r, a + "q_proj.weight"),
        "k": _lin(r, a + "k_proj.weight"),
        "v": _lin(r, a + "v_proj.weight"),
        "conv_q": r.get(a + "q_conv1d.weight")[:, 0, :],
        "conv_k": r.get(a + "k_conv1d.weight")[:, 0, :],
        "conv_v": r.get(a + "v_conv1d.weight")[:, 0, :],
        "f_a": _lin(r, a + "f_a_proj.weight"),
        "f_b": _lin(r, a + "f_b_proj.weight"),
        "dt_bias": r.get(a + "dt_bias"),
        "A_log": r.get(a + "A_log"),
        "b": _lin(r, a + "b_proj.weight"),
        "g_a": _lin(r, a + "g_a_proj.weight"),
        "g_b": _lin(r, a + "g_b_proj.weight"),
        "o_norm": r.get(a + "o_norm.weight"),
        "o": _lin(r, a + "o_proj.weight")
    }


def _load_mlp(r: CheckpointReader, m: str):
    return {
        "gate": _lin(r, m + "gate_proj.weight"),
        "up": _lin(r, m + "up_proj.weight"),
        "down": _lin(r, m + "down_proj.weight")
    }


def _load_moe(r: CheckpointReader, m: str, n_experts: int):
    gu, dn = [], []
    for e in range(n_experts):
        x = f"{m}experts.{e}."
        gu.append(
            np.concatenate([
                _lin(r, x + "gate_proj.weight"),
                _lin(r, x + "up_proj.weight")
            ],
                           axis=1))
        dn.append(_lin(r, x + "down_proj.weight"))
    return {
        "router_w": _lin(r, m + "gate.weight"),
        "router_bias": r.get(m + "gate.e_score_correction_bias"),
        "gate_up": np.stack(gu),
        "down": np.stack(dn),
        "shared": _load_mlp(r, m + "shared_experts.")
    }


def load_layer(r: CheckpointReader, i: int, layer_type: str, mlp_type: str,
               n_experts: int):
    pfx = f"{PREFIX}layers.{i}."
    p = {
        "ln1": r.get(pfx + "input_layernorm.weight"),
        "ln2": r.get(pfx + "post_attention_layernorm.weight"),
        "hc_attn": {
            "fn": _lin(r, pfx + "hc_attn_fn"),
            "base": r.get(pfx + "hc_attn_base"),
            "scale": r.get(pfx + "hc_attn_scale")
        },
        "hc_ffn": {
            "fn": _lin(r, pfx + "hc_ffn_fn"),
            "base": r.get(pfx + "hc_ffn_base"),
            "scale": r.get(pfx + "hc_ffn_scale")
        }
    }
    a = pfx + "self_attn."
    p["attn"] = _load_kda(
        r, a) if layer_type == "linear_attention" else _load_mla(r, a)
    p["mlp"] = _load_moe(r, pfx + "mlp.",
                         n_experts) if mlp_type == "sparse" else _load_mlp(
                             r, pfx + "mlp.")
    return p


def load_mtp_layer(r: CheckpointReader, i: int, n_experts: int):
    pfx = f"{PREFIX}layers.{i}."
    return {
        "ln1": r.get(pfx + "input_layernorm.weight"),
        "ln2": r.get(pfx + "post_attention_layernorm.weight"),
        "attn": _load_mla(r, pfx + "self_attn."),
        "mlp": _load_moe(r, pfx + "mlp.", n_experts),
        "enorm": r.get(pfx + "enorm.weight"),
        "hnorm": r.get(pfx + "hnorm.weight"),
        "eh_proj": _lin(r, pfx + "eh_proj.weight"),
        "head_norm": r.get(pfx + "shared_head.norm.weight")
    }


def load_ref_params(model_dir: str,
                    text_cfg,
                    n_layers: int | None = None,
                    with_mtp: bool = False) -> dict:
    """Checkpoint directory -> nested dict of float32 numpy arrays in the `ref_model` layout. `text_cfg`: the
    transformers Glm5NextTextConfig. `n_layers`: load only the first n decoder layers."""
    r = CheckpointReader(model_dir)
    L = text_cfg.num_hidden_layers if n_layers is None else n_layers
    params = {
        "embed":
        r.get(PREFIX + "embed_tokens.weight"),
        "norm":
        r.get(PREFIX + "norm.weight"),
        "lm_head":
        _lin(r, "lm_head.weight"),
        "layers": [
            load_layer(r, i, text_cfg.layer_types[i],
                       text_cfg.mlp_layer_types[i], text_cfg.n_routed_experts)
            for i in range(L)
        ]
    }
    if with_mtp:
        params["mtp"] = [
            load_mtp_layer(r, text_cfg.num_hidden_layers + j,
                           text_cfg.n_routed_experts)
            for j in range(text_cfg.num_nextn_predict_layers)
        ]
    return params
