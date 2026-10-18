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
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.nope_mla_attention import (_cache_dims,
                                                            _token_requests,
                                                            write_latents)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product


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
                    select_block: int = 64,
                    attend_block: int = 32,
                    pool_chunk: int | None = 8192,
                    select_shard: tuple | None = None):
    """Indexer selection + attention over the selected tokens' latents.

    q_lat [T, H, kvr]; qi [T, IH, ihd] and wts [T, IH] (f32; wts already
    scaled by IH^-0.5); the caches must already hold this step's latents and
    completed pools. Returns o_lat [T, H, kvr].

    select_block / attend_block: queries per step of the selection / the
    attention (both run as a sequential map over query blocks).
    pool_chunk: score the pools in chunks of about this many, merging a
    running top-K, for only as many chunks as the block's queries can see:
    the cost follows the context, not max_model_len. None scores all the
    pools a request can hold at once (the reference path).
    select_shard: (axis name, size) inside a shard_map whose shards hold the
    same queries with different heads (the head shards): the shards split the
    queries for both the selection and the attention.
    """
    T, H, _ = q_lat.shape
    num_pages, page_size, packing, width = _cache_dims(latent)
    req, valid, _, pps = _step_info(positions, block_tables, seq_lens,
                                    query_start_loc, distribution, page_size)
    max_reqs = query_start_loc.shape[0] - 1
    bt = block_tables.reshape(max_reqs, pps)
    sel = functools.partial(select_pools,
                            pooled=pooled,
                            bt=bt,
                            page_size=page_size,
                            index_topk=index_topk,
                            pool_chunk=pool_chunk,
                            query_block=select_block)
    kp = page_size // pooled.shape[1]
    att = functools.partial(attend,
                            latent=latent,
                            bt=bt,
                            kp=kp,
                            scale=scale,
                            kvr=kvr,
                            query_block=attend_block)
    if select_shard is None:
        out = att(q_lat,
                  pools=sel(qi, wts, req, positions),
                  req=req,
                  positions=positions)
    else:
        # The head shards split the queries: each selects for, and attends
        # for, its slice of T / n queries across all heads (q_lat comes in
        # head-sharded and goes back that way through all-to-alls), so each
        # shard gathers only its queries' latents.
        axis, n = select_shard
        Ts = T // n
        i = lax.axis_index(axis)

        def part(x):
            return lax.dynamic_slice_in_dim(x, i * Ts, Ts, axis=0)

        pools = sel(part(qi), part(wts), part(req), part(positions))
        q_all = lax.all_to_all(q_lat, axis, 0, 1, tiled=True)  # [Ts, H*n]
        out = att(q_all, pools=pools, req=part(req), positions=part(positions))
        out = lax.all_to_all(out, axis, 1, 0, tiled=True)  # [T, H]
    return jnp.where(valid[:, None, None], out, 0.0).astype(q_lat.dtype)


def select_pools(qi, wts, req, positions, *, pooled, bt, page_size: int,
                 index_topk: int, pool_chunk, query_block: int):
    """The indexer's choice for every query: [T, K + 1] pool ids of its
    request (its top K = index_topk // kpool pools, then its own pool while
    incomplete), -1 where a slot is unused."""
    T = qi.shape[0]
    _, prows, ihd = pooled.shape
    kp = page_size // prows
    pps = bt.shape[1]
    P = pps * prows  # pools a request can hold
    K = min(index_topk // kp, P)
    lo = jnp.finfo(jnp.float32).min
    f32 = jnp.float32

    def scores(qi_b, w_b, pages, first_pool, pos_b, same):
        """same: every query of the block is of one request (pages[0]'s)
        -> its pooled keys are gathered once for the block."""
        n = pages.shape[1] * prows

        def shared():
            pk = pooled[pages[0]].reshape(n, ihd)
            return jnp.einsum("thd,pd->thp",
                              qi_b.astype(pk.dtype),
                              pk,
                              preferred_element_type=f32)

        def per_query():
            pk = pooled[pages].reshape(-1, n, ihd)  # [Tb, n, ihd]
            return jnp.einsum("thd,tpd->thp",
                              qi_b.astype(pk.dtype),
                              pk,
                              preferred_element_type=f32)

        sc = lax.cond(same, shared, per_query)
        sc = jnp.einsum("th,thp->tp", w_b, jax.nn.relu(sc * ihd**-0.5))
        ids = first_pool + jnp.arange(n)
        visible = ids[None] * kp + kp - 1 <= pos_b[:, None]
        return jnp.where(visible, sc, lo), ids

    stream = pool_chunk is not None and P > pool_chunk
    if stream:
        cpages = max(1, pool_chunk // prows)  # pages per chunk
        n_chunks = -(-pps // cpages)
        bt_pad = jnp.pad(bt, ((0, 0), (0, n_chunks * cpages - pps)))

    def select(qi_b, w_b, req_b, pos_b):
        same = jnp.all(req_b == req_b[0])
        if not stream:
            sc, _ = scores(qi_b, w_b, bt[req_b], 0, pos_b, same)
            return top_k(sc, K)
        pages = bt_pad[req_b]  # [Tb, n_chunks * cpages]
        Tb = pages.shape[0]

        def body(j, carry):
            vals, idx = carry
            pg = lax.dynamic_slice_in_dim(pages, j * cpages, cpages, axis=1)
            sc, ids = scores(qi_b, w_b, pg, j * cpages * prows, pos_b, same)
            # running entries first: ties keep the lowest pool id, as one
            # top_k over all the pools would
            cv = jnp.concatenate([vals, sc], -1)
            ci = jnp.concatenate([idx, jnp.broadcast_to(ids, sc.shape)], -1)
            v, i = lax.top_k(cv, K)
            # a flat 1-D gather (take_along_axis lowers to a slow batched
            # gather on TPU)
            rows = jnp.arange(Tb, dtype=i.dtype)[:, None] * cv.shape[-1]
            return v, ci.reshape(-1)[rows + i]

        # chunks holding at least one complete pool visible from the block
        visible_pools = (jnp.max(pos_b) + 1) // kp
        n = jnp.minimum(-(-visible_pools // (cpages * prows)), n_chunks)
        init = (jnp.full((Tb, K), lo, f32), jnp.zeros((Tb, K), jnp.int32))
        return lax.fori_loop(0, n, body, init)

    def block(args):
        qi_b, w_b, req_b, pos_b = args
        vals, pools = select(qi_b, w_b, req_b, pos_b)
        pools = jnp.where(vals > lo, pools, -1)  # [Tb, K]
        # plus the query's own pool while it is incomplete (its tokens up to
        # the query; a complete pool is selectable like any other)
        own = jnp.where(pos_b % kp != kp - 1, pos_b // kp, -1)
        return jnp.concatenate([pools, own[:, None]],
                               -1).astype(jnp.int32)  # [Tb, K + 1]

    query_block = min(query_block, T)  # small (decode) batches never pad
    pad = (-T) % query_block
    args = [
        jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
        for x in (qi.astype(f32), wts.astype(f32), req, positions)
    ]
    blocks = tuple(x.reshape((-1, query_block) + x.shape[1:]) for x in args)
    tok = lax.map(block, blocks)
    return tok.reshape(-1, tok.shape[-1])[:T]


# Steps of at least this many queries view the latent cache as rows of kpool
# tokens (on TPU the reshape copies the whole cache once per call: amortized
# over a prefill chunk, and the row gathers are faster); smaller (decode)
# steps gather from the cache in its own layout instead.
ROW_VIEW_MIN_TOKENS = 512


def attend(q_lat, *, latent, pools, req, positions, bt, kp: int, scale: float,
           kvr: int, query_block: int):
    """Softmax attention of every query over the latents of its selected
    pools' tokens (K = V = c; the own pool's tokens up to the query). pools
    [T, K + 1] from select_pools. Returns [T, H, kvr] f32. The latents are
    gathered as rows of kpool tokens, in the cache dtype."""
    T, H, _ = q_lat.shape
    num_pages, page_size, packing, width = _cache_dims(latent)
    ppp = page_size // kp  # pools per page
    pool_rows = (latent.reshape(num_pages * ppp, kp, width)
                 if T >= ROW_VIEW_MIN_TOKENS else None)
    bt_flat = bt.reshape(-1)
    pps = bt.shape[1]
    lo = jnp.finfo(jnp.float32).min
    f32 = jnp.float32

    def block(args):
        q_b, pool_b, req_b, pos_b = args
        Tb, n = pool_b.shape
        pl = jnp.maximum(pool_b, 0)
        page = bt_flat[req_b[:, None] * pps + pl // ppp]  # [Tb, n]
        if pool_rows is not None:
            keys = pool_rows[page * ppp + pl % ppp][..., :kvr]
        else:
            off = (pl % ppp)[..., None] * kp + jnp.arange(kp)  # [Tb, n, kp]
            keys = latent[page[..., None], off // packing, off % packing, :kvr]
        keys = keys.reshape(Tb, n * kp, kvr)
        tpos = pl[..., None] * kp + jnp.arange(kp)
        ok = (pool_b >= 0)[..., None] & (tpos <= pos_b[:, None, None])
        s = jnp.einsum("thr,twr->thw",
                       q_b.astype(keys.dtype),
                       keys,
                       preferred_element_type=f32) * scale
        s = jnp.where(ok.reshape(Tb, 1, n * kp), s, lo)
        p = jax.nn.softmax(s, -1)
        return jnp.einsum("thw,twr->thr",
                          p.astype(keys.dtype),
                          keys,
                          preferred_element_type=f32)

    query_block = min(query_block, T)
    pad = (-T) % query_block
    args = [
        jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
        for x in (q_lat, pools, req, positions)
    ]
    blocks = tuple(x.reshape((-1, query_block) + x.shape[1:]) for x in args)
    return lax.map(block, blocks).reshape(-1, H, kvr)[:T]


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
    n_head_shards = get_mesh_shape_product(mesh, H)

    def local(latent, pooled, tail, q_lat, c, k, gate, ape, qi, wts, positions,
              block_tables, seq_lens, query_start_loc, distribution,
              tail_slots):
        # the indexer's selection is the same for every head shard: split it
        # over them by query instead of repeating it
        split = n_head_shards > 1 and positions.shape[0] % n_head_shards == 0
        shard = (H, n_head_shards) if split else None
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
                                kvr=kvr,
                                select_shard=shard)
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
