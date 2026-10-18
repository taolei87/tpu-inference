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
"""M0 checks for the GLM-5.3-Flash parity oracle (`glm5_next_ref`).

* The vendored functional JAX model matches transformers' `Glm5NextTextModel` on tiny random checkpoints
  (dense and sparse attention, chunked KDA, decode).
* Tiny checkpoints use the real checkpoint's tensor names, layouts and FP8 rules (checked against a template
  table derived from the real `model.safetensors.index.json`), reload through transformers, and load through the
  reference reader.
* The real `config.json` parses in transformers and vLLM.
"""
import json
import os
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.glm5_next")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tests.models.jax.glm5_next_ref import make_tiny_ckpt as MT  # noqa: E402
from tests.models.jax.glm5_next_ref import ref_checkpoint as RC  # noqa: E402
from tests.models.jax.glm5_next_ref import ref_model as M  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "glm5_next_ref", "data")


@pytest.fixture(autouse=True)
def _highest_precision():
    with jax.default_matmul_precision("highest"):
        yield


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def _hf_hidden(hf_model, ids):
    with torch.no_grad():
        return hf_model.model.language_model(
            input_ids=torch.as_tensor(ids),
            use_cache=False).last_hidden_state.numpy()


def _ref(ckpt_dir, cfg):
    params = jax.tree.map(
        jnp.asarray,
        RC.load_ref_params(ckpt_dir, cfg.text_config, with_mtp=True))
    return params, M.Cfg.from_hf(cfg.text_config)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("glm5_next_tiny"))
    hf_model, cfg = MT.write_tiny_checkpoint(d, seed=0)
    params, rcfg = _ref(d, cfg)
    return d, hf_model, cfg, params, rcfg


# ----------------------------------------------------------------------------- reference model vs transformers
def test_prefill_matches_hf(tiny):
    _, hf_model, _, params, rcfg = tiny
    ids = np.random.default_rng(0).integers(0, rcfg.vocab, size=(2, 37))
    out, _ = M.forward(params, jnp.asarray(ids), rcfg)
    assert _rel(out, _hf_hidden(hf_model, ids)) < 2e-3


def test_kda_chunk_boundary_matches_hf(tmp_path):
    """130 tokens cross two KDA chunks (64); index_topk=256 keeps MLA dense so HF and ref agree exactly."""
    hf_model, cfg = MT.write_tiny_checkpoint(str(tmp_path),
                                             seed=1,
                                             index_topk=256)
    params, rcfg = _ref(str(tmp_path), cfg)
    ids = np.random.default_rng(1).integers(0, rcfg.vocab, size=(1, 130))
    out, _ = M.forward(params, jnp.asarray(ids), rcfg)
    assert _rel(out, _hf_hidden(hf_model, ids)) < 2e-3


def test_decode_matches_prefill(tiny):
    """prefill(T-1) + one recurrent decode step == prefill(T); a 3-token chunked continuation == prefill(T)."""
    _, _, _, params, rcfg = tiny
    T = 20
    ids = np.random.default_rng(2).integers(0, rcfg.vocab, size=(1, T))
    full, _ = M.forward(params, jnp.asarray(ids), rcfg)
    _, caches = M.forward(params, jnp.asarray(ids[:, :T - 1]), rcfg, cap=T + 4)
    step, _ = M.forward(params,
                        jnp.asarray(ids[:, T - 1:]),
                        rcfg,
                        caches=caches,
                        pos0=T - 1,
                        use_recurrent=True)
    assert _rel(step[:, -1], full[:, -1]) < 2e-3
    _, caches = M.forward(params, jnp.asarray(ids[:, :T - 3]), rcfg, cap=T)
    step3, _ = M.forward(params,
                         jnp.asarray(ids[:, T - 3:]),
                         rcfg,
                         caches=caches,
                         pos0=T - 3)
    assert _rel(step3, full[:, T - 3:]) < 2e-3


def test_sparse_indexer_matches_hf(tmp_path):
    """index_topk=16 (4 pools) over 45 tokens: DSA attends to a strict subset. Prefill, query blocking and a decode
    step all match HF; dense attention must not (so the test exercises sparsity)."""
    hf_model, cfg = MT.write_tiny_checkpoint(str(tmp_path),
                                             seed=3,
                                             index_topk=16)
    params, rcfg = _ref(str(tmp_path), cfg)
    T = 45
    ids = np.random.default_rng(3).integers(0, rcfg.vocab, size=(1, T))
    ref = _hf_hidden(hf_model, ids)
    out, _ = M.forward(params, jnp.asarray(ids), rcfg, sparse="always")
    assert _rel(out, ref) < 2e-3
    import dataclasses
    out_blk, _ = M.forward(params,
                           jnp.asarray(ids),
                           dataclasses.replace(rcfg, q_block=7),
                           sparse="always")
    assert _rel(out_blk, ref) < 2e-3
    dense, _ = M.forward(params, jnp.asarray(ids), rcfg, sparse=None)
    assert _rel(dense, ref) > 1e-3
    _, caches = M.forward(params,
                          jnp.asarray(ids[:, :T - 1]),
                          rcfg,
                          sparse="always",
                          cap=T + 7)
    step, _ = M.forward(params,
                        jnp.asarray(ids[:, T - 1:]),
                        rcfg,
                        caches=caches,
                        pos0=T - 1,
                        use_recurrent=True,
                        sparse="always")
    assert _rel(step[:, -1], ref[:, -1]) < 2e-3


def test_mtp_layer_runs(tiny):
    _, _, _, params, rcfg = tiny
    ids = np.random.default_rng(4).integers(0, rcfg.vocab, size=(1, 9))
    h, _ = M.forward(params, jnp.asarray(ids), rcfg)
    nxt = params["embed"][jnp.asarray(np.roll(ids, -1, axis=1))]
    out, cache = M.mtp_layer(params["mtp"][0], nxt, h, rcfg)
    assert out.shape == h.shape and bool(jnp.all(jnp.isfinite(out)))
    assert set(cache) == {"c", "pk", "tk", "tg"}


# ----------------------------------------------------------------------------- checkpoint format
def _template(name):
    t = re.sub(r"experts\.\d+\.", "experts.E.", name)
    m = re.match(r"(model\.language_model\.layers\.)(\d+)\.(.*)", t)
    if m:
        return m.group(1) + "L." + m.group(3), int(m.group(2))
    return re.sub(r"blocks\.\d+\.", "blocks.B.", t), None


def _real_templates():
    with open(os.path.join(DATA, "real_weight_templates.json")) as f:
        return json.load(f)["templates"]


def test_fp8_rule_matches_real_checkpoint():
    """`fp8_quantized` reproduces which tensors carry *_scale_inv in the real checkpoint."""
    with open(os.path.join(DATA, "real_config.json")) as f:
        layer_types = json.load(f)["text_config"]["layer_types"]
    for tpl, e in _real_templates().items():
        for layer in e["layers"] or [None]:
            name = tpl if layer is None else tpl.replace(
                ".layers.L.", f".layers.{layer}.").replace(
                    "experts.E.", "experts.0.")
            want = layer in e["fp8_layers"] if "fp8_layers" in e else e["fp8"]
            assert RC.fp8_quantized(name, layer_types) == want, name


def test_tiny_checkpoint_names_match_real(tiny):
    """Every tensor of the tiny checkpoint has a counterpart template in the real checkpoint, and every real
    text-model template appears in the tiny checkpoint (the tiny vision tower is a subset of the real one)."""
    d, _, cfg, _, _ = tiny
    real = _real_templates()
    names = [
        n for n in RC.CheckpointReader(d).names()
        if not n.endswith("_scale_inv")
    ]
    tiny_tpl = {_template(n)[0] for n in names}
    assert tiny_tpl <= set(real), sorted(tiny_tpl - set(real))
    real_text = {
        t
        for t in real
        if t.startswith("model.language_model.") or t == "lm_head.weight"
    }
    assert real_text <= tiny_tpl, sorted(real_text - tiny_tpl)
    # per-layer-kind structure: a template appears on a tiny layer iff it appears on a real layer of the same kind
    tc = cfg.text_config
    real_cfg = json.load(open(os.path.join(DATA,
                                           "real_config.json")))["text_config"]

    def kind(i, n_layers, layer_types, mlp_types):
        if i >= n_layers:
            return ("mtp", )
        return (layer_types[i] == "linear_attention", mlp_types[i])

    real_kinds, tiny_kinds = {}, {}
    for tpl, e in real.items():
        for i in e["layers"]:
            real_kinds.setdefault(
                kind(i, real_cfg["num_hidden_layers"], real_cfg["layer_types"],
                     real_cfg["mlp_layer_types"]), set()).add(tpl)
    for n in names:
        tpl, i = _template(n)
        if i is not None:
            tiny_kinds.setdefault(
                kind(i, tc.num_hidden_layers, tc.layer_types,
                     tc.mlp_layer_types), set()).add(tpl)
    for k, tpls in tiny_kinds.items():
        assert tpls == real_kinds[k], (k, tpls ^ real_kinds[k])


def test_tiny_config_json_matches_real_keys(tiny):
    """The tiny config.json has exactly the real one's keys (minus quantization_config, which only the FP8 variant
    writes) and the raw layer-type strings."""
    d = tiny[0]
    real = json.load(open(os.path.join(DATA, "real_config.json")))
    mine = json.load(open(os.path.join(d, "config.json")))
    assert set(mine) == set(real) - {"quantization_config"}
    assert set(mine["text_config"]) == set(real["text_config"])
    assert set(mine["vision_config"]) == set(real["vision_config"])
    assert set(mine["text_config"]["linear_attn_config"]) == set(
        real["text_config"]["linear_attn_config"])
    assert set(mine["text_config"]["layer_types"]) == set(
        real["text_config"]["layer_types"])


def test_tiny_checkpoint_reloads_in_transformers(tiny):
    """transformers maps the real names onto its modules (conversion_mapping) with nothing missing; only the MTP
    layer, which transformers does not build, is left over."""
    from transformers import Glm5NextForConditionalGeneration
    d, hf_model, cfg, _, _ = tiny
    hf2, info = Glm5NextForConditionalGeneration.from_pretrained(
        d, dtype=torch.float32, output_loading_info=True)
    assert not info["missing_keys"], info["missing_keys"]
    mtp = f"layers.{cfg.text_config.num_hidden_layers}."
    assert all(mtp in k
               for k in info["unexpected_keys"]), info["unexpected_keys"]
    ids = np.random.default_rng(5).integers(0,
                                            cfg.text_config.vocab_size,
                                            size=(1, 11))
    np.testing.assert_array_equal(_hf_hidden(hf2.eval(), ids),
                                  _hf_hidden(hf_model, ids))


def test_tiny_checkpoint_bf16_dtypes(tmp_path):
    MT.write_tiny_checkpoint(str(tmp_path), seed=0, dtype="bfloat16")
    r = RC.CheckpointReader(str(tmp_path))
    for n in r.names():
        want = torch.float32 if n.endswith(
            MT._F32_SUFFIXES) else torch.bfloat16
        assert r.raw(n).dtype == want, n


def test_tiny_checkpoint_fp8(tmp_path, tiny):
    """FP8 checkpoint: exactly the real FP8 tensors are e4m3 with block scales, the quantization_config matches the
    real one's form, and the dequantized model stays close to the f32 one."""
    d32, _, cfg, params32, rcfg = tiny
    MT.write_tiny_checkpoint(str(tmp_path), seed=0, fp8=True)
    r = RC.CheckpointReader(str(tmp_path))
    layer_types = list(cfg.text_config.layer_types)
    names = r.names()
    for n in names:
        if n.endswith("_scale_inv"):
            continue
        is_fp8 = r.raw(n).dtype == torch.float8_e4m3fn
        assert is_fp8 == RC.fp8_quantized(n, layer_types), n
        assert is_fp8 == ((n + "_scale_inv") in names), n
    with open(os.path.join(str(tmp_path), "config.json")) as f:
        qc = json.load(f)["quantization_config"]
    real_qc = json.load(open(os.path.join(
        DATA, "real_config.json")))["quantization_config"]
    assert {k: v for k, v in qc.items() if k != "modules_to_not_convert"} == \
        {k: v for k, v in real_qc.items() if k != "modules_to_not_convert"}
    assert "model.layers.0.hc_attn_fn" in qc["modules_to_not_convert"]
    assert "model.layers.2.self_attn.kv_b_proj" in qc["modules_to_not_convert"]
    assert "model.layers.2.self_attn.q_a_proj" not in qc[
        "modules_to_not_convert"]
    params8 = jax.tree.map(jnp.asarray,
                           RC.load_ref_params(str(tmp_path), cfg.text_config))
    ids = np.random.default_rng(6).integers(0, rcfg.vocab, size=(1, 16))
    out8, _ = M.forward(params8, jnp.asarray(ids), rcfg)
    out32, _ = M.forward(params32, jnp.asarray(ids), rcfg)
    assert 0 < _rel(out8, out32) < 0.1


def test_fp8_block_roundtrip():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((300, 200)).astype(np.float32)
    q, s = RC.quant_fp8_block(w)
    assert s.shape == (3, 2)
    back = RC.dequant_fp8_block(q.float().numpy(), s)
    assert np.abs(back - w).max() <= 0.07 * np.abs(w).max()


# ----------------------------------------------------------------------------- configs
def test_real_config_parses():
    from transformers import Glm5NextConfig
    with open(os.path.join(DATA, "real_config.json")) as f:
        c = Glm5NextConfig.from_dict(json.load(f))
    t = c.text_config
    rc = M.Cfg.from_hf(t)
    assert (rc.hidden, rc.n_layers, rc.vocab) == (4096, 45, 154880)
    assert (rc.kda_heads, rc.kda_hd, rc.conv_k, rc.gate_lb) == (64, 128, 4,
                                                                -5.0)
    assert (rc.n_heads, rc.q_lora, rc.kv_lora, rc.qk_nope,
            rc.v_hd) == (64, 1536, 512, 256, 256)
    assert (rc.idx_heads, rc.idx_hd, rc.idx_kpool, rc.idx_topk) == (32, 128, 4,
                                                                    2048)
    assert (rc.hc, rc.hc_iters) == (4, 20)
    assert (rc.n_exp, rc.topk, rc.n_shared, rc.moe_inter,
            rc.inter) == (288, 8, 1, 2048, 12288)
    assert rc.layer_types.count(
        "linear_attention") == 34 and rc.mlp_types[:4] == ("dense", ) * 3 + (
            "sparse", )
    assert t.qk_rope_head_dim == 0 and t.num_nextn_predict_layers == 1


@pytest.mark.parametrize("which", ["tiny", "real"])
def test_vllm_model_config(which, tiny, tmp_path):
    """vLLM resolves the architecture and the text config the tpu-inference model will read."""
    pytest.importorskip("vllm")
    from vllm.config import ModelConfig
    if which == "tiny":
        d = tiny[0]
    else:
        d = str(tmp_path)
        with open(os.path.join(DATA, "real_config.json")) as f, open(
                os.path.join(d, "config.json"), "w") as g:
            g.write(f.read())
    mc = ModelConfig(model=d,
                     tokenizer=None,
                     skip_tokenizer_init=True,
                     dtype="bfloat16",
                     max_model_len=4096)
    assert mc.architecture == "Glm5NextForConditionalGeneration"
    assert type(mc.hf_text_config).__name__ == "Glm5NextTextConfig"
    assert mc.is_hybrid and mc.use_mla and mc.is_multimodal_model
    want = json.load(open(os.path.join(d, "config.json")))["text_config"]
    assert mc.hf_text_config.num_hidden_layers == want["num_hidden_layers"]
    assert mc.hf_text_config.linear_num_heads == want["linear_attn_config"][
        "num_heads"]
