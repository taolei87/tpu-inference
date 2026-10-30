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
"""Which grammar-bitmask row goes to which request: vLLM orders the rows by
structured_output_request_ids (scheduling order; under DP the ranks' ids
concatenated), not by sorted id. Runs without the full runner."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tpu_inference.runner.structured_decoding_manager import \
    StructuredDecodingManager


def _runner(req_ids, vocab=64):
    words = vocab // 32
    return SimpleNamespace(mesh=Mesh(np.array(jax.devices()[:1]), ("model", )),
                           vocab_size=vocab,
                           input_batch=SimpleNamespace(req_id_to_index={
                               r: i
                               for i, r in enumerate(req_ids)
                           }),
                           grammar_bitmask_cpu=np.zeros((8, words), np.int32),
                           require_structured_out_cpu=np.zeros((8, 1),
                                                               np.bool_),
                           structured_decode_arange=np.arange(32,
                                                              dtype=np.int32))


def test_bitmask_rows_follow_the_request_order_not_sorted_ids():
    # batch: b at index 0, a at 1, c at 2; rows ordered c, (gone), b
    runner = _runner(["b", "a", "c"])
    mgr = StructuredDecodingManager(runner)
    only_low = np.array([-1, 0], np.int32)  # tokens 0-31 allowed
    only_high = np.array([0, -1], np.int32)  # tokens 32-63 allowed
    out = SimpleNamespace(structured_output_request_ids=["c", "gone", "b"],
                          grammar_bitmask=np.stack([
                              only_high,
                              np.array([7, 7], np.int32), only_low
                          ]))
    logits = jnp.ones((3, 64), jnp.float32)
    require, bitmask, arange = mgr.prepare_structured_decoding_input(
        logits, out)
    np.testing.assert_array_equal(
        np.asarray(require)[:, 0], [True, False, True])
    masked = np.asarray(
        mgr.structured_decode_fn(require, bitmask, logits, arange))
    assert (masked[0, :32] == 1).all() and (masked[0, 32:] == -np.inf).all()
    assert (masked[1] == 1).all()  # a: no structured output
    assert (masked[2, :32] == -np.inf).all() and (masked[2, 32:] == 1).all()


def test_dict_form_maps_ids_to_rows():
    """Older vLLM: a dict from request id to row."""
    runner = _runner(["a", "b"])
    mgr = StructuredDecodingManager(runner)
    rows = np.array([[1, 0], [2, 0]], np.int32)
    out = SimpleNamespace(structured_output_request_ids={
        "b": 0,
        "a": 1
    },
                          grammar_bitmask=rows)
    _, bitmask, _ = mgr.prepare_structured_decoding_input(
        jnp.ones((2, 64)), out)
    np.testing.assert_array_equal(np.asarray(bitmask)[:, 0], [2, 1])
