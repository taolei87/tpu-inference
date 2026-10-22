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
"""The KDA chunk-scan kernel against a plain loop (interpret mode on CPU)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.kda.chunk_scan import kda_chunk_scan


@pytest.mark.parametrize("dk", [16, 128])
def test_matches_loop(dk):
    rng = np.random.default_rng(dk)
    n, H, c, dv = 6, 2, 64, dk
    f = np.float32
    r = lambda *s: (0.3 * rng.standard_normal(s)).astype(f)  # noqa: E731
    u, w, q, attn = r(n, H, c, dv), r(n, H, c, dk), r(n, H, c,
                                                      dk), r(n, H, c, c)
    k_dec = r(n, H, dk, c)
    decay = rng.random((n, H, dk, 1)).astype(f)
    # chunks 0-2: sequence 0; 3-4: sequence 1; 5: padding (dummy sequence 2)
    seq = np.array([0, 0, 0, 1, 1, 2], np.int32)
    reset = np.array([1, 0, 0, 1, 0, 1], np.int32)
    init = r(3, H, dk, dv)
    out_ref, final_ref = np.zeros((n, H, c, dv)), np.zeros((3, H, dk, dv))
    for h in range(H):
        S = None
        for i in range(n):
            if reset[i]:
                S = init[seq[i], h].astype(np.float64)
            v_new = u[i, h] - w[i, h] @ S
            out_ref[i, h] = q[i, h] @ S + attn[i, h] @ v_new
            S = S * decay[i, h] + k_dec[i, h] @ v_new
            final_ref[seq[i], h] = S
    out, final = kda_chunk_scan(*map(
        jnp.asarray, (u, w, attn, q, k_dec, decay, seq, reset, init)),
                                interpret=jax.default_backend() != "tpu")
    for got, want in ((out, out_ref), (final, final_ref)):
        err = np.abs(np.asarray(got) - want).max() / np.abs(want).max()
        assert err < 1e-5, err
