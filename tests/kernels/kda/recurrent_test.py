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
"""The recurrent KDA kernel against a plain JAX loop (interpret mode on CPU)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.kda.recurrent import kda_recurrent


def _reference(s0, q, k, g, v, beta, state, write, ok):
    R, H, dk, G = q.shape
    state, out = np.array(state), np.zeros(v.shape, np.float32)
    for r in range(R):
        for h in range(H):
            S = np.array(s0[r, h], np.float64)
            for t in range(G):
                if not ok[r, t]:
                    out[r, h, t] = (q[r, h, :, t] @ S)
                    continue
                S = S * np.exp(g[r, h, :, t])[:, None]
                kt = k[r, h, :, t]
                delta = (v[r, h, t] - kt @ S) * beta[r, h, 0, t]
                S = S + np.outer(kt, delta)
                out[r, h, t] = q[r, h, :, t] @ S
                state[write[r, t], h] = S
    return state, out


@pytest.mark.parametrize("DK", [16, 128])
@pytest.mark.parametrize("G", [1, 4])
def test_matches_loop(G, DK):
    rng = np.random.default_rng(G)
    R, H, dk, dv, N = 3, 2, DK, DK, 2 + 3 * G
    f = np.float32

    def unit(x):
        return (x / np.linalg.norm(x, axis=2, keepdims=True)).astype(f)

    s0 = (0.3 * rng.standard_normal((R, H, dk, dv))).astype(f)
    q = unit(rng.standard_normal((R, H, dk, G))) * dk**-0.5
    k = unit(rng.standard_normal((R, H, dk, G)))
    g = (-5.0 / (1 + np.exp(-rng.standard_normal((R, H, dk, G))))).astype(f)
    v = rng.standard_normal((R, H, G, dv)).astype(f)
    beta = (1 / (1 + np.exp(-rng.standard_normal((R, H, 1, G))))).astype(f)
    state = rng.standard_normal((N, H, dk, dv)).astype(f)
    write = (1 + np.arange(R * G).reshape(R, G)).astype(np.int32)
    ok = np.ones((R, G), np.int32)
    ok[1, G - 1] = 0  # a window shorter than G
    ok[2] = 0  # a padding request
    want_state, want_out = _reference(s0, q, k, g, v, beta, state, write, ok)
    got_state, got_out = kda_recurrent(
        *map(jnp.asarray, (s0, q, k, g, v, beta, state, write, ok)),
        interpret=jax.default_backend() != "tpu")
    np.testing.assert_allclose(np.asarray(got_state),
                               want_state,
                               rtol=1e-5,
                               atol=1e-6)
    valid = ok.astype(bool)
    np.testing.assert_allclose(np.asarray(got_out)[valid.nonzero()[0], :,
                                                   valid.nonzero()[1]],
                               want_out[valid.nonzero()[0], :,
                                        valid.nonzero()[1]],
                               rtol=1e-5,
                               atol=1e-6)
