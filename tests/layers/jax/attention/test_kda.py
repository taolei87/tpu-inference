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
"""The KDA layer (`layers/jax/attention/kda.py`) loaded from a tiny GLM-5.3-
Flash checkpoint vs the reference `kda_attention`, per request of a ragged
batch."""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from flax import nnx  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

pytest.importorskip("transformers.models.glm5_next")

from tests.models.jax.glm5_next_ref import make_tiny_ckpt as MT  # noqa: E402
from tests.models.jax.glm5_next_ref import ref_checkpoint as RC  # noqa: E402
from tests.models.jax.glm5_next_ref import ref_model as R  # noqa: E402
from tpu_inference.layers.common.attention_metadata import \
    AttentionMetadata  # noqa: E402
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES  # noqa: E402
from tpu_inference.layers.jax.attention.kda import \
    KimiDeltaAttention  # noqa: E402
from tpu_inference.models.jax.utils.weight_utils import \
    JaxAutoWeightsLoader  # noqa: E402

LAYER = 1  # a KDA layer of the tiny model


@pytest.fixture(autouse=True)
def _highest_precision():
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("glm5_next_kda"))
    _, cfg = MT.write_tiny_checkpoint(d, seed=7)
    return d, cfg.text_config


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def _layer(tiny, tp):
    d, text = tiny
    if len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} devices")
    mesh = Mesh(
        np.array(jax.devices()[:tp]).reshape(1, 1, 1, 1, tp, 1, 1),
        MESH_AXIS_NAMES)
    with jax.set_mesh(mesh):
        layer = KimiDeltaAttention(text,
                                   jnp.float32,
                                   nnx.Rngs(0),
                                   mesh,
                                   prefix=f"layers.{LAYER}.self_attn")
    r = RC.CheckpointReader(d)
    pfx = f"{RC.PREFIX}layers.{LAYER}.self_attn."
    weights = [(n[len(pfx):], r.raw(n)) for n in r.names()
               if n.startswith(pfx)]
    with jax.set_mesh(mesh):  # as models/common/model_loader.py does
        loaded = JaxAutoWeightsLoader(layer).load_weights(iter(weights))
    assert loaded == {n for n, _ in layer.named_parameters()}
    return layer, mesh


def _batch(rng, text, lens, has_state, max_reqs=6, num_slots=8, pad=3):
    H, dh = text.linear_num_heads, text.linear_head_dim
    C, K = H * dh, text.linear_conv_kernel_dim
    T = sum(lens) + pad
    x = rng.standard_normal((T, text.hidden_size)).astype(np.float32)
    slots = rng.permutation(np.arange(1,
                                      num_slots))[:max_reqs].astype(np.int32)
    conv = rng.standard_normal((num_slots, K - 1, 3 * C)).astype(np.float32)
    rec = (0.3 * rng.standard_normal(
        (num_slots, H, dh, dh))).astype(np.float32)
    qsl = np.full(max_reqs + 1, sum(lens), np.int32)
    qsl[:len(lens) + 1] = np.concatenate([[0], np.cumsum(lens)])
    n_dec = sum(1 for n, s in zip(lens, has_state) if n == 1 and s)
    seq_lens = np.zeros(max_reqs, np.int32)
    seq_lens[:len(lens)] = [
        n + (5 if s else 0) for n, s in zip(lens, has_state)
    ]
    md = AttentionMetadata(input_positions=jnp.zeros((T, ), jnp.int32),
                           seq_lens=jnp.asarray(seq_lens),
                           query_start_loc=jnp.asarray(qsl),
                           request_distribution=jnp.asarray(
                               [n_dec, len(lens), len(lens)], jnp.int32),
                           mamba_state_indices=jnp.asarray(slots))
    return x, conv, rec, slots, qsl, md


def _interleave(x, tp, axis):
    """Global q | k | v channel layout -> per-shard [q_s | k_s | v_s]."""
    per = [np.split(p, tp, axis=axis) for p in np.split(x, 3, axis=axis)]
    return np.concatenate([per[j][s] for s in range(tp) for j in range(3)],
                          axis=axis)


@pytest.mark.parametrize("tp", [1, 2])
def test_kda_layer_matches_reference(tiny, tp):
    d, text = tiny
    layer, mesh = _layer(tiny, tp)
    ref_p = jax.tree.map(
        jnp.asarray,
        RC.load_layer(RC.CheckpointReader(d), LAYER, "linear_attention",
                      "sparse", text.n_routed_experts)["attn"])
    rcfg = R.Cfg.from_hf(text)
    lens, has_state = [1, 1, 23, 70], [True, True, False, True]
    x, conv, rec, slots, qsl, md = _batch(np.random.default_rng(0), text, lens,
                                          has_state)
    with jax.set_mesh(mesh):
        (new_conv, new_rec), y = jax.jit(lambda mod, c, x, m: mod(c, x, m))(
            layer, (jnp.asarray(_interleave(conv, tp, 2)), jnp.asarray(rec)),
            jnp.asarray(x), md)
    new_conv = np.asarray(new_conv)
    y, new_rec = np.asarray(y), np.asarray(new_rec)
    for i, (n, s) in enumerate(zip(lens, has_state)):
        a, e, slot = qsl[i], qsl[i + 1], slots[i]
        cache = ({
            "conv": jnp.asarray(conv[slot][None]),
            "state": jnp.asarray(rec[slot][None])
        } if s else None)
        ry, rc = R.kda_attention(ref_p, jnp.asarray(x[a:e][None]), rcfg, cache)
        assert _rel(y[a:e], ry[0]) < 1e-4, i
        assert _rel(new_rec[slot], rc["state"][0]) < 1e-4, i
        assert _rel(new_conv[slot],
                    _interleave(np.asarray(rc["conv"][0]), tp, 1)) < 1e-5, i
