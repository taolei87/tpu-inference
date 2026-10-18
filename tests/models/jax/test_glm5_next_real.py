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
"""The real GLM-5.3-Flash checkpoint truncated to its first decoder layers
(real FP8 weights, all routed experts), built and loaded through the production
model loader, against the f32 reference on the exactly dequantized weights.

Needs a TPU host and the checkpoint: set GLM5_NEXT_REAL_CKPT to a directory
with `config.json`, `model.safetensors.index.json` and the tokenizer files. Only
the shards holding the kept layers have to be present. GLM5_NEXT_REAL_LAYERS
sets the number of kept layers (default 5: dense-MLP KDA layers, the first MLA
layer and the first MoE layers)."""

import json
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from vllm.config import set_current_vllm_config

from tests.models.jax import test_glm5_next as T
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.models.common import model_loader
from tpu_inference.models.jax.glm5_next import Glm5NextForConditionalGeneration
from tpu_inference.utils import to_jax_dtype

CKPT = os.environ.get("GLM5_NEXT_REAL_CKPT")
N_LAYERS = int(os.environ.get("GLM5_NEXT_REAL_LAYERS", "5"))
TP = 4

pytestmark = [
    pytest.mark.skipif(not CKPT, reason="GLM5_NEXT_REAL_CKPT not set"),
    pytest.mark.skipif(jax.default_backend() != "tpu"
                       or len(jax.devices()) < TP,
                       reason=f"needs {TP} TPU chips"),
]

PROMPTS = [
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
]


@pytest.fixture(autouse=True)
def _new_model_design(monkeypatch):
    monkeypatch.setenv("NEW_MODEL_DESIGN", "1")


def _truncated_view(src: str, dst: str, n: int) -> str:
    """A checkpoint directory holding the first n decoder layers: an edited
    config, an index restricted to their tensors, links to the shards."""
    cfg = json.load(open(os.path.join(src, "config.json")))
    text = cfg["text_config"]
    for k in ("layer_types", "mlp_layer_types", "indexer_types"):
        text[k] = text[k][:n]
    text["num_hidden_layers"] = n
    lac = text["linear_attn_config"]
    for k in ("kda_layers", "full_attn_layers"):
        lac[k] = [i for i in lac[k] if i < n]
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"))

    def keep(name):
        m = re.search(r"\.layers\.(\d+)\.", name)
        return not name.startswith("model.visual.") and not (m and int(
            m.group(1)) >= n)

    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    weight_map = {k: f for k, f in idx["weight_map"].items() if keep(k)}
    json.dump({
        "metadata": idx.get("metadata", {}),
        "weight_map": weight_map
    }, open(os.path.join(dst, "model.safetensors.index.json"), "w"))
    for f in set(weight_map.values()) | {
            f
            for f in os.listdir(src) if f.startswith("tokenizer")
    }:
        os.symlink(os.path.join(src, f), os.path.join(dst, f))
    return dst


@pytest.mark.parametrize("latent_dtype", ["bfloat16", "float8_e4m3fn"])
def test_truncated_real_checkpoint(tmp_path, latent_dtype):
    d = _truncated_view(CKPT, str(tmp_path), N_LAYERS)
    vllm_config = T._vllm_config(d, "bfloat16")
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config)
    vllm_config.model_config.dtype = to_jax_dtype(
        vllm_config.model_config.dtype)
    mesh = Mesh(
        np.array(jax.devices()[:TP]).reshape(1, 1, 1, 1, TP, 1, 1),
        MESH_AXIS_NAMES)
    with set_current_vllm_config(vllm_config):
        model = model_loader._get_nnx_model(Glm5NextForConditionalGeneration,
                                            vllm_config, jax.random.key(0),
                                            mesh)
    text = vllm_config.model_config.hf_text_config

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(d)
    ids = {
        i: np.asarray(tok(p)["input_ids"], np.int64)
        for i, p in enumerate(PROMPTS)
    }
    # The oracle's dense MoE gathers per-token expert weights: run it on the
    # host.
    with jax.default_matmul_precision("highest"), jax.default_device(
            jax.devices("cpu")[0]):
        ref = T._reference(d, ids)

    caches = T._Caches(text,
                       dtype=jnp.bfloat16,
                       latent_dtype=jnp.dtype(latent_dtype))
    a, b = len(ids[0]), len(ids[1])
    # chunked prefill, then a decode and the prefill remainder
    s1 = T._run(model, mesh, caches, [(0, 0, a - 3), (1, 0, b - 1)], ids)
    s2 = T._run(model, mesh, caches, [(1, b - 1, 1), (0, a - 3, 3)], ids)
    for r in ids:
        logits = np.concatenate([s1[r][1], s2[r][1]]).astype(np.float32)
        p = jax.nn.log_softmax(ref[r][1].astype(np.float32), -1)
        q = jax.nn.log_softmax(logits, -1)
        kl = float(jnp.mean(jnp.sum(jnp.exp(p) * (p - q), -1)))
        top1 = (logits.argmax(-1) == ref[r][1].argmax(-1)).mean()
        # measured on v6e (default weight requantization): KL 0.0007 at 5
        # layers, 0.0023 at 12; top-1 >= 0.94
        print(f"REALCKPT latent={latent_dtype} req={r} kl={kl:.5f} "
              f"top1={top1:.3f}")
        assert kl < 0.004 * max(1, N_LAYERS / 12), (r, kl)
        assert top1 >= 0.9, (r, top1)
        assert logits[-1].argmax() == ref[r][1][-1].argmax(), r
