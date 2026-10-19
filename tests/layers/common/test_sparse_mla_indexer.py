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
"""The streaming pool selection of `sparse_nope_mla` (chunks of pools, a
running top-K, as many chunks as the queries can see) against scoring every
pool a request can hold at once."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common.sparse_mla_indexer import sparse_nope_mla

PAGE, KP, PPS, MAX_REQS = 16, 4, 12, 3  # 48 pools per request
H, KVR, IH, IHD = 2, 32, 4, 16


def _inputs(seed, ties):
    rng = np.random.default_rng(seed)
    num_pages = MAX_REQS * PPS + 1
    latent = rng.standard_normal((num_pages, PAGE, 1, KVR), np.float32)
    pooled = rng.standard_normal((num_pages, PAGE // KP, IHD), np.float32)
    if ties:
        # many pools score exactly 0 after the ReLU: ties decide the top-K
        pooled[rng.random(pooled.shape[:2]) < 0.6] = -1.0
    bt = rng.permutation(np.arange(1, num_pages)).astype(np.int32)
    # a decode at 150, a prefill chunk 40..63 and a short request's first 9
    step = [(0, 150, 1), (1, 40, 24), (2, 0, 9)]
    T = sum(n for *_, n in step) + 6  # padding tokens
    pos = np.zeros(T, np.int32)
    qsl = np.zeros(MAX_REQS + 1, np.int32)
    seq_lens = np.zeros(MAX_REQS, np.int32)
    t = 0
    for i, (_, p0, n) in enumerate(step):
        pos[t:t + n] = np.arange(p0, p0 + n)
        t += n
        qsl[i + 1] = t
        seq_lens[i] = p0 + n
    q_lat = rng.standard_normal((T, H, KVR), np.float32)
    qi = rng.standard_normal((T, IH, IHD), np.float32)
    if ties:
        qi = np.abs(qi)
    wts = rng.random((T, IH), np.float32) * IH**-0.5
    return (jnp.asarray(q_lat), jnp.asarray(qi), jnp.asarray(wts),
            jnp.asarray(latent), jnp.asarray(pooled), jnp.asarray(pos),
            jnp.asarray(bt), jnp.asarray(seq_lens), jnp.asarray(qsl),
            jnp.asarray([1, len(step), len(step)], jnp.int32))


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("pool_chunk", [4, 8, 20])
def test_streaming_selection_matches_one_shot(ties, pool_chunk):
    args = _inputs(0, ties)
    kw = dict(index_topk=32, scale=KVR**-0.5, kvr=KVR)  # K = 8 pools
    ref = sparse_nope_mla(*args, **kw, pool_chunk=None)
    out = jax.jit(lambda *a: sparse_nope_mla(*a, **kw, pool_chunk=pool_chunk))(
        *args)
    np.testing.assert_allclose(np.asarray(out),
                               np.asarray(ref),
                               rtol=1e-6,
                               atol=1e-6)


def test_streaming_reads_only_visible_chunks():
    """Pools past the queries' positions never change the result, even when
    they would win the top-K."""
    args = list(_inputs(1, False))
    kw = dict(index_topk=32, scale=KVR**-0.5, kvr=KVR, pool_chunk=4)
    out = sparse_nope_mla(*args, **kw)
    pooled = np.asarray(args[4]).copy()
    bt = np.asarray(args[6]).reshape(MAX_REQS, PPS)
    # request 2 sees positions < 9: poison every page past its first
    pooled[bt[2, 1:]] = 1e3
    args[4] = jnp.asarray(pooled)
    poisoned = sparse_nope_mla(*args, **kw)
    t0, t1 = 25, 34  # request 2's tokens
    np.testing.assert_allclose(np.asarray(poisoned)[t0:t1],
                               np.asarray(out)[t0:t1],
                               rtol=1e-6,
                               atol=1e-6)


@pytest.mark.parametrize("pool_chunk", [None, 8])
def test_shared_key_gather_matches_per_query(pool_chunk):
    """Blocks of one request gather the pooled keys once (select_block 1:
    every block); blocks spanning requests gather them per query (one block
    for the whole step). Same selection."""
    args = _inputs(2, True)
    kw = dict(index_topk=32, scale=KVR**-0.5, kvr=KVR, pool_chunk=pool_chunk)
    shared = sparse_nope_mla(*args, **kw, select_block=1)
    per_query = sparse_nope_mla(*args, **kw, select_block=64)
    np.testing.assert_allclose(np.asarray(shared),
                               np.asarray(per_query),
                               rtol=1e-6,
                               atol=1e-6)


@pytest.mark.parametrize("packing", [2, 4])
def test_packed_latent_cache_layout(packing):
    """The same latents stored two / four tokens per row (as the TPU MLA
    cache packs bf16 / fp8) give the same attention as packing 1."""
    args = list(_inputs(3, False))
    kw = dict(index_topk=32, scale=KVR**-0.5, kvr=KVR, pool_chunk=8)
    flat = sparse_nope_mla(*args, **kw)
    lat = np.asarray(args[3])  # [pages, PAGE, 1, KVR]
    args[3] = jnp.asarray(
        lat.reshape(lat.shape[0], PAGE // packing, packing, KVR))
    packed = sparse_nope_mla(*args, **kw)
    np.testing.assert_allclose(np.asarray(packed),
                               np.asarray(flat),
                               rtol=1e-6,
                               atol=1e-6)


def test_row_view_and_native_gathers_agree(monkeypatch):
    """Large steps gather the latents through a row view of the cache, small
    ones from the cache layout: the same attention."""
    from tpu_inference.layers.common import sparse_mla_indexer as SM
    args = _inputs(4, False)
    kw = dict(index_topk=32, scale=KVR**-0.5, kvr=KVR, pool_chunk=8)
    monkeypatch.setattr(SM, "ROW_VIEW_MIN_TOKENS", 1)
    rows = np.asarray(SM.sparse_nope_mla(*args, **kw))
    monkeypatch.setattr(SM, "ROW_VIEW_MIN_TOKENS", 10**9)
    native = np.asarray(SM.sparse_nope_mla(*args, **kw))
    np.testing.assert_allclose(rows, native, rtol=1e-6, atol=1e-6)
