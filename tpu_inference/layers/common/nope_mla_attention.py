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
"""NoPE MLA over a paged latent cache, in JAX (GLM-5.3-Flash).

With no rotary part and the key up-projection absorbed into the queries,
attention runs directly on the latents: keys and values are both the
normalized latent c [kv_lora_rank] of each token, queries are q_lat [H,
kv_lora_rank].

The cache uses the MLA kernels' paged layout
`[num_pages, page_size // packing, packing, align(kv_lora_rank, 128)]`
(`kernels/mla/v2/kernel.get_kv_cache_shape` with no rope part). Token `pos` of
request `r` lives in page `block_tables[r * pages_per_seq + pos // page_size]`.

These are the correctness implementations (CPU tests, reference): the dense
attention gathers each query's whole request, so its memory grows with the
context length.
"""
import jax
import jax.numpy as jnp


def _token_requests(query_start_loc, distribution, num_tokens):
    """Request index and validity of every packed token."""
    num_valid = distribution[2]
    mask = jnp.arange(query_start_loc.shape[0]) <= num_valid
    qsl = jnp.where(mask, query_start_loc, query_start_loc[num_valid])
    req = jnp.searchsorted(qsl, jnp.arange(num_tokens), side="right") - 1
    req = jnp.clip(req, 0, query_start_loc.shape[0] - 2)
    return req, jnp.arange(num_tokens) < qsl[num_valid]


def _cache_dims(cache):
    num_pages, rows, packing, width = cache.shape
    return num_pages, rows * packing, packing, width


def write_latents(cache, new_c, positions, block_tables, query_start_loc,
                  distribution):
    """Scatters new latents [T, kvr] into the paged cache at each token's
    position; padding tokens are dropped."""
    T, kvr = new_c.shape
    num_pages, page_size, packing, width = _cache_dims(cache)
    max_reqs = query_start_loc.shape[0] - 1
    pages_per_seq = block_tables.shape[0] // max_reqs
    req, valid = _token_requests(query_start_loc, distribution, T)
    page = block_tables[req * pages_per_seq + positions // page_size]
    page = jnp.where(valid, page, num_pages)  # out of bounds -> dropped
    off = positions % page_size
    row = jnp.zeros((T, width),
                    cache.dtype).at[:, :kvr].set(new_c.astype(cache.dtype))
    if packing == 1:
        return cache.at[page, off, 0].set(row, mode="drop")
    # Packed rows (bf16: 2, fp8: 4 tokens): write whole rows. A scatter into
    # the packing axis is a sub-word update on TPU. Every token assembles its
    # destination row: the slots this step writes come from its row-mates,
    # which are its neighbours in the packed stream (same request,
    # consecutive positions); the other slots keep the cache's values. Tokens
    # sharing a row write the same row.
    r = off // packing
    old = cache[jnp.minimum(page, num_pages - 1), r]  # [T, packing, width]
    s = jnp.arange(packing)
    src = jnp.arange(T)[:, None] - (off % packing)[:, None] + s  # [T, packing]
    srcc = jnp.clip(src, 0, T - 1)
    mine = ((src >= 0) & (src < T) & valid[srcc] & (req[srcc] == req[:, None])
            & (positions[srcc] == positions[:, None] -
               (off % packing)[:, None] + s))
    new_rows = jnp.where(mine[..., None], row[srcc], old)
    return cache.at[page, r].set(new_rows, mode="drop")


def dense_nope_mla(q_lat,
                   cache,
                   positions,
                   block_tables,
                   seq_lens,
                   query_start_loc,
                   distribution,
                   *,
                   scale: float,
                   kvr: int,
                   query_block: int = 16):
    """Causal attention of every query over its request's cached latents.

    q_lat [T, H, kvr]; the cache must already hold this step's latents
    (`write_latents`). Returns o_lat [T, H, kvr] in q_lat's dtype.
    """
    T, H, _ = q_lat.shape
    num_pages, page_size, packing, width = _cache_dims(cache)
    max_reqs = query_start_loc.shape[0] - 1
    pages_per_seq = block_tables.shape[0] // max_reqs
    req, valid = _token_requests(query_start_loc, distribution, T)
    L = pages_per_seq * page_size
    flat = cache.reshape(num_pages, page_size, width)[..., :kvr]
    lo = jnp.finfo(jnp.float32).min

    def block(args):
        q_b, req_b, pos_b = args  # [Tb, H, kvr], [Tb], [Tb]
        pages = block_tables.reshape(max_reqs, pages_per_seq)[req_b]
        keys = flat[pages].reshape(-1, L, kvr)  # [Tb, L, kvr]
        kpos = jnp.arange(L)
        mask = (kpos[None] <= pos_b[:, None]) & (kpos[None]
                                                 < seq_lens[req_b][:, None])
        s = jnp.einsum("thr,tlr->thl", q_b.astype(jnp.float32),
                       keys.astype(jnp.float32)) * scale
        s = jnp.where(mask[:, None], s, lo)
        p = jax.nn.softmax(s, -1)
        return jnp.einsum("thl,tlr->thr", p, keys.astype(jnp.float32))

    pad = (-T) % query_block
    padded = [
        jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
        for x in (q_lat, req, positions)
    ]
    blocks = [x.reshape((-1, query_block) + x.shape[1:]) for x in padded]
    out = jax.lax.map(block, tuple(blocks)).reshape(-1, H, kvr)[:T]
    return jnp.where(valid[:, None, None], out, 0.0).astype(q_lat.dtype)
