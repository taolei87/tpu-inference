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
"""The k-pool sparse indexer and sparse NoPE MLA of GLM-5.3-Flash, in JAX.

The indexer keys k [ihd] of every kpool consecutive tokens are pooled into one
key with a per-dimension softmax over the pool's tokens:

    pooled = sum_i softmax_i(gate_i + ape_i) * k_i          (i < kpool)

A query scores every complete pool visible from its position,

    score = sum_h w_h * relu(q_h . pooled / sqrt(ihd)),

attends to the tokens of its top `index_topk // kpool` pools plus the tokens
of its own incomplete pool, and attention runs on their latents (K = V = c).
While a request's context fits in `index_topk` this selects every visible
token, i.e. it is exact dense attention.

Caches (see `runner/glm5_next_kv_cache.py`):
  latent  [num_pages, page_size // packing, packing, width]  (MLA layout)
  pooled  [num_pages, page_size // kpool, ihd]  pool p of a request lives at
          row (p * kpool % page_size) // kpool of its page.
  tail    [num_slots, kpool, 2 * ihd]  raw (k | gate) of the request's
          incomplete pool by position % kpool, in the compact slot pool.

These are the correctness implementations: queries run in blocks of
`query_block`, each gathering its request's pooled keys.
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.nope_mla_attention import (_cache_dims,
                                                            _token_requests,
                                                            write_latents)
from tpu_inference.layers.common.sharding import ShardingAxisName


def top_k(x, k, chunk=None):
    """`lax.top_k` over the last axis with a chunked first stage (the top k of
    every chunk, then of the candidates): the same values and ids, without
    one sort over a very long axis."""
    n = x.shape[-1]
    c = chunk or max(4 * k, 256)
    if n <= 2 * c or 2 * k > c:
        return lax.top_k(x, k)
    pad = (-n) % c
    if pad:
        x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)],
                    constant_values=-jnp.inf)
    xc = x.reshape(x.shape[:-1] + (-1, c))
    v, i = lax.top_k(xc, k)
    i = i + (jnp.arange(xc.shape[-2], dtype=i.dtype) * c)[:, None]
    v = v.reshape(x.shape[:-1] + (-1, ))
    i = i.reshape(x.shape[:-1] + (-1, ))
    v, j = lax.top_k(v, k)
    return v, jnp.take_along_axis(i, j, -1)


def _step_info(positions, block_tables, seq_lens, query_start_loc,
               distribution, page_size):
    T = positions.shape[0]
    max_reqs = query_start_loc.shape[0] - 1
    pages_per_seq = block_tables.shape[0] // max_reqs
    req, valid = _token_requests(query_start_loc, distribution, T)
    qlen = query_start_loc[1:] - query_start_loc[:-1]
    step_start = (seq_lens - qlen)[req]  # first position of this step
    return req, valid, step_start, pages_per_seq


def indexer_write(pooled, tail, k, gate, ape, positions, block_tables,
                  seq_lens, query_start_loc, distribution, tail_slots):
    """Pools the k-pools completed by this step's tokens into `pooled` and
    keeps the raw (k | gate) of every request's incomplete last pool in
    `tail`.

    k, gate [T, ihd]; ape [kpool, ihd]; tail_slots [max_reqs] (the compact
    slot of each request). A pool may span earlier steps (read from the
    tail) and this step.
    """
    T, ihd = k.shape
    kp = ape.shape[0]
    num_pages, rows, _ = pooled.shape
    page_size = rows * kp
    req, valid, step_start, pps = _step_info(positions, block_tables, seq_lens,
                                             query_start_loc, distribution,
                                             page_size)
    f32 = jnp.float32
    slot = tail_slots[req]
    # The pool window of every token: positions base + i, i < kpool.
    base = (positions // kp) * kp
    wpos = base[:, None] + jnp.arange(kp)  # [T, kp]
    from_step = wpos >= step_start[:, None]
    src = jnp.clip(
        jnp.arange(T)[:, None] - (positions[:, None] - wpos), 0, T - 1)
    kg = jnp.concatenate([k, gate], -1).astype(f32)  # [T, 2 ihd]
    win = jnp.where(from_step[..., None], kg[src],
                    tail[slot].astype(f32))  # [T, kp, 2 ihd]
    wk, wg = win[..., :ihd], win[..., ihd:]
    probs = jax.nn.softmax(wg + ape.astype(f32)[None], axis=1)
    pool = (probs * wk).sum(1)  # [T, ihd]
    completes = valid & (positions % kp == kp - 1)
    page = block_tables[req * pps + positions // page_size]
    page = jnp.where(completes, page, num_pages)  # dropped
    row = (base % page_size) // kp
    pooled = pooled.at[page, row].set(pool.astype(pooled.dtype), mode="drop")
    # Tokens of the request's final incomplete pool go to its tail.
    end = seq_lens[req]
    in_tail = valid & (positions >= end - end % kp)
    tail_slot = jnp.where(in_tail, slot, tail.shape[0])
    tail = tail.at[tail_slot, positions % kp].set(kg.astype(tail.dtype),
                                                  mode="drop")
    return pooled, tail


def sparse_nope_mla(q_lat,
                    qi,
                    wts,
                    latent,
                    pooled,
                    positions,
                    block_tables,
                    seq_lens,
                    query_start_loc,
                    distribution,
                    *,
                    index_topk: int,
                    scale: float,
                    kvr: int,
                    query_block: int = 8):
    """Indexer selection + attention over the selected tokens' latents.

    q_lat [T, H, kvr]; qi [T, IH, ihd] and wts [T, IH] (f32; wts already
    scaled by IH^-0.5); the caches must already hold this step's latents and
    completed pools. Returns o_lat [T, H, kvr].
    """
    T, H, _ = q_lat.shape
    num_pages, page_size, packing, width = _cache_dims(latent)
    _, prows, ihd = pooled.shape
    kp = page_size // prows
    req, valid, _, pps = _step_info(positions, block_tables, seq_lens,
                                    query_start_loc, distribution, page_size)
    max_reqs = query_start_loc.shape[0] - 1
    bt = block_tables.reshape(max_reqs, pps)
    P = pps * prows  # pools a request can hold
    K = min(index_topk // kp, P)
    flat_latent = latent.reshape(num_pages, page_size, width)
    lo = jnp.finfo(jnp.float32).min
    f32 = jnp.float32

    def block(args):
        q_b, qi_b, w_b, req_b, pos_b = args
        pages = bt[req_b]  # [Tb, pps]
        pk = pooled[pages].reshape(-1, P, ihd).astype(f32)  # [Tb, P, ihd]
        sc = jax.nn.relu(jnp.einsum("thd,tpd->thp", qi_b, pk) * ihd**-0.5)
        sc = jnp.einsum("th,thp->tp", w_b, sc)
        visible = jnp.arange(P)[None] * kp + kp - 1 <= pos_b[:, None]
        sc = jnp.where(visible, sc, lo)
        vals, pools = top_k(sc, K)
        tok = pools[..., None] * kp + jnp.arange(kp)  # [Tb, K, kp]
        tok = jnp.where((vals > lo)[..., None], tok, -1).reshape(-1, K * kp)
        # plus the query's own incomplete pool: positions start .. pos
        start = (pos_b + 1) - (pos_b + 1) % kp
        off = jnp.arange(kp - 1)
        tail = jnp.where(start[:, None] + off <= pos_b[:, None],
                         start[:, None] + off, -1)
        tok = jnp.concatenate([tok, tail], -1)  # [Tb, W]
        ok = tok >= 0
        tok = jnp.maximum(tok, 0)
        page = jnp.take_along_axis(pages, tok // page_size, axis=1)
        keys = flat_latent[page, tok % page_size, :kvr].astype(f32)
        s = jnp.einsum("thr,twr->thw", q_b.astype(f32), keys) * scale
        s = jnp.where(ok[:, None], s, lo)
        return jnp.einsum("thw,twr->thr", jax.nn.softmax(s, -1), keys)

    pad = (-T) % query_block
    args = [
        jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
        for x in (q_lat, qi.astype(f32), wts.astype(f32), req, positions)
    ]
    blocks = tuple(x.reshape((-1, query_block) + x.shape[1:]) for x in args)
    out = lax.map(block, blocks).reshape(-1, H, kvr)[:T]
    return jnp.where(valid[:, None, None], out, 0.0).astype(q_lat.dtype)


def run_sparse_mla_step(latent, pooled, tail, q_lat, c, k, gate, ape, qi, wts,
                        positions, block_tables, seq_lens, query_start_loc,
                        distribution, tail_slots, *, mesh, index_topk: int,
                        scale: float, kvr: int):
    """One MLA layer's cache writes and sparse attention under the mesh.

    Requests are sharded on ATTN_DATA: with data-parallel attention the
    runner packs one segment of tokens and metadata per rank, and each rank
    owns its pages and slots (block tables and slots are rank-local), so every
    shard runs the single-rank ops on its own segment. Attention heads are
    sharded on ATTN_HEAD; the latent, pooled-key and tail caches are
    replicated over the head shards, and every shard writes the same values.

    Returns (latent, pooled, tail, o_lat [T, H, kvr]).
    """
    D, H = ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD
    B, CTX = ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT

    def local(latent, pooled, tail, q_lat, c, k, gate, ape, qi, wts, positions,
              block_tables, seq_lens, query_start_loc, distribution,
              tail_slots):
        latent = write_latents(latent, c, positions, block_tables,
                               query_start_loc, distribution)
        pooled, tail = indexer_write(pooled, tail, k, gate, ape, positions,
                                     block_tables, seq_lens, query_start_loc,
                                     distribution, tail_slots)
        o_lat = sparse_nope_mla(q_lat,
                                qi,
                                wts,
                                latent,
                                pooled,
                                positions,
                                block_tables,
                                seq_lens,
                                query_start_loc,
                                distribution,
                                index_topk=index_topk,
                                scale=scale,
                                kvr=kvr)
        return latent, pooled, tail, o_lat

    tok = P(D, None)
    fn = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(B, CTX), P(B), P(D), P(D, H, None), tok, tok, tok, P(),
                  P(D, None, None), tok, P(D), P(D), P(D), P(D), P(D), P(D)),
        out_specs=(P(B, CTX), P(B), P(D), P(D, H, None)),
        check_vma=False,
    )
    return fn(latent, pooled, tail, q_lat, c, k, gate, ape, qi, wts, positions,
              block_tables, seq_lens, query_start_loc, distribution,
              tail_slots)
