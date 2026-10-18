# Copyright (c) 2026 Abdur Rahim
# Copyright 2026 Google LLC (modifications)
#
# Licensed under the MIT License. Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
# following conditions: The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED. See SYNC.md for the full license text and provenance.
"""GLM-5.3-Flash (Glm5Next) text model in pure functional JAX: the parity oracle for tpu-inference.

Vendored and trimmed from kaggle-tpu-lab `glm53-flash/engine/glm53/model.py` (MIT, Copyright (c) 2026 Abdur Rahim;
see SYNC.md). Upstream validated this math against transformers `modeling_glm5_next.py` at rel err < 2e-3; the tests in
`tests/models/jax/test_glm5_next_ref.py` re-check the trimmed copy against transformers.

Trimmed relative to upstream: tensor/sequence sharding, the int8 latent cache, Pallas kernel switches, speculative-
decode histories / rollback, per-row ("rows") batching, the key-dedup ("union") fast path and the MTP layer's
sharded variants. What remains is the single-device math: one cache, all batch rows in lockstep.

Parameter layout (all Linear weights stored as [in, out], i.e. y = x @ W):
  embed [V, D], norm [D], lm_head [D, V]
  layers[i]:
    ln1 [D], ln2 [D]
    hc_attn / hc_ffn: fn [hc*D, (2+hc)*hc], base [(2+hc)*hc], scale [3]
    attn (KDA):  q,k,v [D, H*hd]; conv_q, conv_k, conv_v [H*hd, K]; f_a [D, hd]; f_b [hd, H*hd];
                 dt_bias [H*hd]; A_log [H]; b [D, H]; g_a [D, hd]; g_b [hd, H*hd]; o_norm [hd]; o [H*hd, D]
    attn (MLA):  q_a [D, qr]; q_a_norm [qr]; q_b [qr, H*qk]; kv_a [D, kvr]; kv_a_norm [kvr];
                 kv_b [kvr, H*(qk+v)]; o [H*v, D];
                 indexer {wq_b [qr, IH*ihd], wk [D, ihd], k_norm_w, k_norm_b [ihd], weights_proj [D, IH],
                          ape [kp, ihd], gate [D, ihd]}
    mlp (dense): gate [D, I], up [D, I], down [I, D]
    mlp (moe):   router_w [D, E], router_bias [E], gate_up [E, D, 2*mi], down [E, mi, D], shared {gate, up, down}
  mtp[j] (optional): enorm, hnorm [D], eh_proj [2D, D], ln1, ln2, attn (MLA), mlp (moe), head_norm [D]
"""
from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

Params = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Cfg:
    hidden: int
    vocab: int
    n_layers: int
    layer_types: tuple[str,
                       ...]  # "linear_attention" | "deepseek_sparse_attention"
    mlp_types: tuple[str, ...]  # "dense" | "sparse"
    eps: float
    # KDA
    kda_heads: int
    kda_hd: int
    conv_k: int
    gate_lb: float | None
    # MLA (NoPE)
    n_heads: int
    q_lora: int
    kv_lora: int
    qk_nope: int
    v_hd: int
    # indexer
    idx_heads: int
    idx_hd: int
    idx_topk: int
    idx_kpool: int
    # mHC
    hc: int
    hc_iters: int
    hc_eps: float
    # MLP / MoE
    inter: int
    moe_inter: int
    n_exp: int
    topk: int
    n_shared: int
    scaling: float
    norm_topk: bool
    swiglu_limit: float
    dtype: Any = jnp.float32
    q_block: int = 128  # queries per attention block (bounds the per-query temporaries)
    cache_cap: int = 0  # set inside mla_core: cache capacity

    @classmethod
    def from_hf(cls, c, dtype=jnp.float32) -> "Cfg":
        """`c`: a transformers Glm5NextTextConfig."""
        return cls(
            hidden=c.hidden_size,
            vocab=c.vocab_size,
            n_layers=c.num_hidden_layers,
            layer_types=tuple(c.layer_types),
            mlp_types=tuple(c.mlp_layer_types),
            eps=c.rms_norm_eps,
            kda_heads=c.linear_num_heads,
            kda_hd=c.linear_head_dim,
            conv_k=c.linear_conv_kernel_dim,
            gate_lb=c.linear_lower_bound,
            n_heads=c.num_attention_heads,
            q_lora=c.q_lora_rank,
            kv_lora=c.kv_lora_rank,
            qk_nope=c.qk_nope_head_dim,
            v_hd=c.v_head_dim,
            idx_heads=c.index_n_heads,
            idx_hd=c.index_head_dim,
            idx_topk=c.index_topk,
            idx_kpool=c.index_kpool,
            hc=c.hc_mult,
            hc_iters=c.hc_sinkhorn_iters,
            hc_eps=c.hc_eps,
            inter=c.intermediate_size,
            moe_inter=c.moe_intermediate_size,
            n_exp=c.n_routed_experts,
            topk=c.num_experts_per_tok,
            n_shared=c.n_shared_experts,
            scaling=c.routed_scaling_factor,
            norm_topk=c.norm_topk_prob,
            swiglu_limit=c.swiglu_limit,
            dtype=dtype,
        )


# ----------------------------------------------------------------------------- basics
def rmsnorm(x, w, eps):
    xf = x.astype(jnp.float32)
    xf = xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + eps)
    return (w.astype(jnp.float32) * xf).astype(x.dtype)


def unweighted_rmsnorm_f32(x, eps):
    return x * lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps)


def l2norm(x, eps=1e-6):
    return x / jnp.sqrt(jnp.sum(x * x, -1, keepdims=True) + eps)


def silu(x):
    return x * jax.nn.sigmoid(x)


def swiglu_clamped(gate, up, limit):
    gate = jnp.minimum(gate, limit)
    up = jnp.clip(up, -limit, limit)
    return silu(gate) * up


def mlp(p, x, cfg: Cfg):
    return swiglu_clamped(x @ p["gate"], x @ p["up"],
                          cfg.swiglu_limit) @ p["down"]


def layernorm(x, w, b, eps):
    xf = x.astype(jnp.float32)
    mu = xf.mean(-1, keepdims=True)
    var = jnp.mean((xf - mu)**2, -1, keepdims=True)
    return ((xf - mu) * lax.rsqrt(var + eps) * w.astype(jnp.float32) +
            b.astype(jnp.float32)).astype(x.dtype)


# ----------------------------------------------------------------------------- mHC
def hc_pre(p, streams, cfg: Cfg):
    """streams [B,T,hc,D] -> (post [B,T,hc], comb [B,T,hc,hc], collapsed [B,T,D])."""
    hc = cfg.hc
    B, T = streams.shape[:2]
    flat = unweighted_rmsnorm_f32(
        streams.reshape(B, T, -1).astype(jnp.float32), cfg.eps)
    mix = flat @ p["fn"].astype(jnp.float32)  # [B,T,(2+hc)*hc]
    pre_w, post_w, comb_w = mix[..., :hc], mix[..., hc:2 * hc], mix[...,
                                                                    2 * hc:]
    base, scale = p["base"].astype(jnp.float32), p["scale"].astype(jnp.float32)
    pre = jax.nn.sigmoid(pre_w * scale[0] + base[:hc]) + cfg.hc_eps
    post = 2.0 * jax.nn.sigmoid(post_w * scale[1] + base[hc:2 * hc])
    comb_logits = comb_w.reshape(
        B, T, hc, hc) * scale[2] + base[2 * hc:].reshape(hc, hc)
    comb = jax.nn.softmax(comb_logits, -1) + cfg.hc_eps
    comb = comb / (comb.sum(-2, keepdims=True) + cfg.hc_eps)  # columns
    for _ in range(cfg.hc_iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + cfg.hc_eps)  # rows
        comb = comb / (comb.sum(-2, keepdims=True) + cfg.hc_eps)  # columns
    collapsed = jnp.einsum("bth,bthd->btd", pre,
                           streams.astype(jnp.float32)).astype(streams.dtype)
    return post, comb, collapsed


def hc_post(post, comb, y, residual):
    """new_streams[k] = post[k] * y + sum_h comb[h, k] * residual[h]."""
    dt = residual.dtype
    return post.astype(dt)[..., None] * y[..., None, :] + jnp.einsum(
        "bthk,bthd->btkd", comb.astype(dt), residual)


# ----------------------------------------------------------------------------- KDA
def causal_conv1d(x, w, state=None):
    """Depthwise causal conv over time. x [B,T,C], w [C,K], state [B,K-1,C] or None (zeros).
    Returns (y [B,T,C] in x's dtype, new_state [B,K-1,C])."""
    B, T, C = x.shape
    K = w.shape[-1]
    if state is None:
        state = jnp.zeros((B, K - 1, C), x.dtype)
    xp = jnp.concatenate([state.astype(x.dtype), x], axis=1)  # [B, T+K-1, C]
    y = jnp.zeros((B, T, C), jnp.float32)
    for j in range(K):
        y = y + xp[:, j:j + T].astype(jnp.float32) * w[:, j].astype(
            jnp.float32)
    return y.astype(x.dtype), xp[:, -(K - 1):]


def kda_gates(p, x, cfg: Cfg):
    """Forget gate g [B,T,H,hd] (log-decay, <= 0) and beta [B,T,H] in fp32."""
    B, T, _ = x.shape
    f = (x @ p["f_a"]) @ p["f_b"]
    g = f.astype(jnp.float32) + p["dt_bias"].astype(jnp.float32)
    g = g.reshape(B, T, cfg.kda_heads, cfg.kda_hd)
    decay_rate = jnp.exp(p["A_log"].astype(jnp.float32))[None, None, :, None]
    if cfg.gate_lb is not None:
        g = cfg.gate_lb * jax.nn.sigmoid(decay_rate * g)
    else:
        g = -decay_rate * jax.nn.softplus(g)
    beta = jax.nn.sigmoid((x @ p["b"]).astype(jnp.float32))
    return g, beta


def kda_chunked(q, k, v, g, beta, state0=None, chunk=64):
    """Chunked KDA (HF `chunk_kimi_delta_attention`). q,k,v [B,T,H,d]; g [B,T,H,d]; beta [B,T,H].
    Returns out [B,T,H,d] (fp32) and final state [B,H,dk,dv] (fp32)."""
    B, T, H, dk = k.shape
    dv = v.shape[-1]
    f32 = jnp.float32
    q, k, v, g = [jnp.swapaxes(t.astype(f32), 1, 2)
                  for t in (q, k, v, g)]  # [B,H,T,*]
    beta = jnp.swapaxes(beta.astype(f32), 1, 2)  # [B,H,T]
    q = l2norm(q) * (dk**-0.5)
    k = l2norm(k)
    pad = (-T) % chunk
    Tp = T + pad
    if pad:
        q, k, v, g = [
            jnp.pad(t, ((0, 0), (0, 0), (0, pad), (0, 0)))
            for t in (q, k, v, g)
        ]
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad)))
    n = Tp // chunk
    q, k, v, g = [t.reshape(B, H, n, chunk, -1) for t in (q, k, v, g)]
    beta = beta.reshape(B, H, n, chunk)
    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    g = jnp.cumsum(g, axis=-2)  # per-channel cumsum in chunk
    ii = jnp.arange(chunk)
    strict_lower = ii[:, None] > ii[None, :]
    lower_incl = ii[:, None] >= ii[None, :]
    eye = jnp.eye(chunk, dtype=f32)

    def chunk_decay(g_i):
        # decay[i,j,d] = exp(g_i[d] - g_j[d]) for i >= j (<= 1); the upper triangle is never used, clamp it
        return jnp.exp(
            jnp.minimum(g_i[..., :, None, :] - g_i[..., None, :, :],
                        0.0))  # [B,H,c,c,d]

    def intra(xs):
        k_i, g_i, kb_i = xs  # [B,H,c,d]
        a = -jnp.einsum("bhid,bhjd,bhijd->bhij", kb_i, k_i, chunk_decay(g_i))
        return jnp.where(strict_lower, a, 0.0)

    def per_chunk(*ts):  # [B,H,n,c,*] -> [n,B,H,c,*]
        return tuple(jnp.moveaxis(t, 2, 0) for t in ts)

    attn = jnp.moveaxis(lax.map(intra, per_chunk(k, g, k_beta)), 0,
                        2)  # [B,H,n,c,c]
    # forward substitution computing (I - attn)^{-1} with unit diagonal
    L = eye - attn
    attn = jnp.vectorize(lambda m: jax.scipy.linalg.solve_triangular(
        m, eye, lower=True, unit_diagonal=True),
                         signature="(c,c)->(c,c)")(L)
    u = attn @ v_beta  # [B,H,n,c,dv]
    w = attn @ (k_beta * jnp.exp(g))  # [B,H,n,c,dk]
    S0 = jnp.zeros(
        (B, H, dk, dv), f32) if state0 is None else state0.astype(f32)

    def step(S, xs):
        q_i, k_i, g_i, u_i, w_i = xs
        attn_inter = (q_i * jnp.exp(g_i)) @ S
        attn_intra = jnp.einsum("bhid,bhjd,bhijd->bhij", q_i, k_i,
                                chunk_decay(g_i))
        attn_intra = jnp.where(lower_incl, attn_intra, 0.0)
        v_new = u_i - w_i @ S
        out = attn_inter + attn_intra @ v_new
        g_last = g_i[..., -1:, :]
        S = S * jnp.exp(g_last).swapaxes(-1, -2) + jnp.swapaxes(
            k_i * jnp.exp(g_last - g_i), -1, -2) @ v_new
        return S, out

    S, out = lax.scan(step, S0, per_chunk(q, k, g, u, w))
    out = jnp.moveaxis(out, 0, 2).reshape(B, H, Tp, dv)[:, :, :T]
    return jnp.swapaxes(out, 1, 2), S


def kda_state_step(S, k_i, v_i, g_i, b_i):
    """The KDA state update of one token (f32; k_i l2-normed): S [B,H,dk,dv], k/v/g [B,H,d], b [B,H]."""
    S = S * jnp.exp(g_i)[..., None]
    kv_mem = jnp.einsum("bhkv,bhk->bhv", S, k_i)
    delta = (v_i - kv_mem) * b_i[..., None]
    return S + k_i[..., None] * delta[..., None, :]


def kda_recurrent(q, k, v, g, beta, state):
    """Token-by-token recurrent form. Shapes as kda_chunked; state [B,H,dk,dv] fp32."""
    f32 = jnp.float32
    dk = k.shape[-1]
    q = l2norm(q.astype(f32)) * (dk**-0.5)
    k = l2norm(k.astype(f32))
    v, g, beta = v.astype(f32), g.astype(f32), beta.astype(f32)

    def step(S, xs):
        q_i, k_i, v_i, g_i, b_i = xs  # [B,H,d] / [B,H]
        S = kda_state_step(S, k_i, v_i, g_i, b_i)
        return S, jnp.einsum("bhkv,bhk->bhv", S, q_i)

    xs = tuple(jnp.moveaxis(t, 1, 0) for t in (q, k, v, g, beta))
    S, out = lax.scan(step, state.astype(f32), xs)
    return jnp.moveaxis(out, 0, 1), S


def kda_attention(p, x, cfg: Cfg, cache=None, use_recurrent=False):
    """x [B,T,D] (already input-normed). cache: None or {"conv": [B,K-1,3C], "state": [B,H,dk,dv]}.
    Returns (y [B,T,D], new_cache)."""
    B, T, _ = x.shape
    H, hd = cfg.kda_heads, cfg.kda_hd
    C = H * hd
    qkv_raw = jnp.concatenate([x @ p["q"], x @ p["k"], x @ p["v"]],
                              -1)  # [B,T,3C]
    conv_w = jnp.concatenate([p["conv_q"], p["conv_k"], p["conv_v"]],
                             0)  # [3C,K]
    g, beta = kda_gates(p, x, cfg)
    gate = ((x @ p["g_a"]) @ p["g_b"]).reshape(B, T, H, hd)
    qkv, new_conv = causal_conv1d(qkv_raw, conv_w,
                                  None if cache is None else cache["conv"])
    qkv = silu(qkv)
    q, k, v = [
        t.reshape(B, T, H, hd)
        for t in (qkv[..., :C], qkv[..., C:2 * C], qkv[..., 2 * C:])
    ]
    state = None if cache is None else cache["state"]
    if use_recurrent:
        assert state is not None
        core, new_state = kda_recurrent(q, k, v, g, beta, state)
    else:
        core, new_state = kda_chunked(q, k, v, g, beta, state)
    # gated RMSNorm over hd (fp32, sigmoid gate) + output projection
    o = core.astype(jnp.float32)
    o = o * lax.rsqrt(jnp.mean(o * o, -1, keepdims=True) +
                      cfg.eps) * p["o_norm"].astype(jnp.float32)
    o = o * jax.nn.sigmoid(gate.astype(jnp.float32))
    y = o.astype(x.dtype).reshape(B, T, C) @ p["o"]
    return y, {"conv": new_conv, "state": new_state}


# ----------------------------------------------------------------------------- DSA indexer (k-pool)
# Cache layout: slot t of the MLA/indexer caches holds position t.
def cache_write(cache, new, pos0, valid=None):
    """Write new [B,T,R] at positions pos0.. into cache [B,S,R]. `valid` [T] bool (optional) keeps the cache
    untouched for rows that are False."""
    new = new.astype(cache.dtype)
    if valid is None:
        return lax.dynamic_update_slice_in_dim(cache, new, pos0, axis=1)
    B, T = new.shape[:2]
    S = cache.shape[1]
    slot = jnp.where(valid, pos0 + jnp.arange(T), S +
                     jnp.arange(T))  # invalid rows go out of bounds (dropped)
    return cache.at[:, slot].set(new, mode="drop", unique_indices=True)


def mla_cache_shapes(cfg: Cfg, B, S, has_indexer):
    """Cache shapes of one MLA layer for capacity S: "c" latent [B,S,kvr]; with an indexer "pk" pooled indexer keys
    [B,S/kp,ihd] (one row per complete k-pool) and the raw keys/gates of the incomplete tail pool "tk"/"tg"
    [B,kp,ihd]."""
    kp = cfg.idx_kpool
    out = {"c": (B, S, cfg.kv_lora)}
    if has_indexer:
        out.update(pk=(B, S // kp, cfg.idx_hd),
                   tk=(B, kp, cfg.idx_hd),
                   tg=(B, kp, cfg.idx_hd))
    return out


def pool_new_keys(ix, tail_k, tail_g, new_k, new_g, pos0, cfg: Cfg, length):
    """k-pool compression of the NEW tokens (the first `length` of new_k/new_g, at positions pos0..) together with
    the raw keys of the incomplete pool they may extend (tail_k/tail_g [B,kp,ihd]: row i = position (pos0//kp)*kp + i
    for i < pos0 % kp). -> (pooled [B,NP,ihd] f32, complete [NP] bool, pool0 = id of pooled[0], new tails)."""
    B, T, ihd = new_k.shape
    kp = cfg.idx_kpool
    f32 = jnp.float32
    a = pos0 % kp  # tokens of the tail pool already seen
    NP = (T + 2 * kp - 2) // kp + 1  # pools covering a + T rows (+1 spare)
    L = NP * kp

    def fill(tail, new):
        buf = jnp.zeros((B, L, ihd), f32).at[:, :kp].set(tail.astype(f32))
        return lax.dynamic_update_slice_in_dim(buf, new.astype(f32), a, axis=1)

    kb, gb = fill(tail_k, new_k), fill(tail_g, new_g)
    present = jnp.arange(L) < a + length  # rows that hold real tokens
    logits = jnp.where(
        present[None, :, None], gb + jnp.tile(ix["ape"].astype(f32),
                                              (L // kp, 1))[None], -jnp.inf)
    probs = jnp.nan_to_num(
        jax.nn.softmax(logits.reshape(B, NP, kp, ihd), axis=2))
    pooled = (probs * kb.reshape(B, NP, kp, ihd)).sum(2)  # [B,NP,ihd]
    complete = present.reshape(NP, kp).all(-1)
    start = (
        (a + length) // kp) * kp  # first row of the (incomplete) last pool
    tail_k2 = lax.dynamic_slice_in_dim(kb, start, kp, axis=1)
    tail_g2 = lax.dynamic_slice_in_dim(gb, start, kp, axis=1)
    return pooled, complete, pos0 // kp, tail_k2, tail_g2


def top_k(x, k, chunk=None):
    """`lax.top_k` over the last axis with a chunked first stage (same values and ids as lax.top_k)."""
    n = x.shape[-1]
    c = chunk or max(4 * k, 256)
    if n <= 2 * c or 2 * k > c:
        return lax.top_k(x, k)
    pad = (-n) % c
    if pad:
        x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)],
                    constant_values=-jnp.inf)
    xc = x.reshape(x.shape[:-1] + (-1, c))
    v, i = lax.top_k(xc, k)  # [..., n/c, k]
    i = i + (jnp.arange(xc.shape[-2], dtype=i.dtype) * c)[:, None]
    v, i = v.reshape(x.shape[:-1] + (-1, )), i.reshape(x.shape[:-1] + (-1, ))
    v, j = lax.top_k(v, k)
    return v, jnp.take_along_axis(i, j, -1)


def indexer_select(scores, qpos, cfg: Cfg):
    """scores [B,Tq,P] f32; qpos [Tq] query positions -> (pools [B,Tq,K] pool ids, valid [B,Tq,K]): the top-K
    complete pools visible from each query."""
    P = scores.shape[-1]
    kp = cfg.idx_kpool
    lo = jnp.finfo(jnp.float32).min
    pool_end = jnp.arange(P) * kp + kp - 1
    visible = pool_end[None, :] <= qpos[:, None]  # [Tq,P]
    scores = jnp.where(visible[None], scores, lo)
    K = min(cfg.idx_topk // kp, P)
    vals, sel = top_k(scores, K)
    return sel, vals > lo


def select_tokens(pools, valid, qpos, cfg: Cfg):
    """Selected pools -> token indices [B,Tq,W] int32 (-1 = invalid): the pools' tokens plus the incomplete tail
    pool of the visible prefix (positions 0..qpos). W = idx_topk + kp - 1 padded to a multiple of 16."""
    B, Tq, K = pools.shape
    kp, S = cfg.idx_kpool, cfg.cache_cap
    tok = (pools[..., None] * kp + jnp.arange(kp)).reshape(B, Tq, K * kp)
    tok = jnp.where(jnp.repeat(valid, kp, axis=-1), tok, -1)
    tok = jnp.pad(tok, ((0, 0), (0, 0), (0, cfg.idx_topk - K * kp)),
                  constant_values=-1)
    vis_count = qpos + 1
    tail_count = vis_count % kp
    tail_start = vis_count - tail_count
    off = jnp.arange(kp - 1)
    tail = tail_start[:, None] + off[None, :]  # [Tq,kp-1]
    tail_ok = (off[None, :] < tail_count[:, None]) & (tail < S)
    tail = jnp.where(tail_ok, tail, -1)
    tok = jnp.concatenate(
        [tok, jnp.broadcast_to(tail[None], (B, Tq, kp - 1))], -1)
    W = tok.shape[-1]
    return jnp.pad(tok, ((0, 0), (0, 0), (0, (-W) % 16)),
                   constant_values=-1).astype(jnp.int32)


def _attend(q, keys, mask, cfg: Cfg, per_query_keys):
    """Softmax attention of q [B,Tq,H,r] over keys ([B,Tq,W,r] if per_query_keys else [B,S,r]) restricted to
    `mask` ([B,Tq,W] or [Tq,S]); keys double as values (NoPE MLA, absorbed)."""
    lo = jnp.finfo(jnp.float32).min
    scale = cfg.qk_nope**-0.5
    if per_query_keys:
        s = jnp.einsum("bthr,btwr->bhtw", q, keys).astype(jnp.float32) * scale
        s = jnp.where(mask[:, None], s, lo)
        pv = "bhtw,btwr->bthr"
    else:
        s = jnp.einsum("bthr,bsr->bhts", q, keys).astype(jnp.float32) * scale
        mask = mask if mask.ndim == 3 else mask[None]
        s = jnp.where(mask[:, None], s, lo)
        pv = "bhts,bsr->bthr"
    probs = jax.nn.softmax(s, -1).astype(keys.dtype)
    return jnp.einsum(pv, probs, keys)


# ----------------------------------------------------------------------------- MLA (NoPE)
def mla_project(p, x, cfg: Cfg):
    """Position-free part of an MLA layer: x [B,T,D] -> dict(q_lat [B,T,H,kvr] absorbed queries, new_c [B,T,kvr];
    with an indexer also nk / ng [B,T,ihd] (its keys / gates), qi [B,T,IH,ihd] f32, wts [B,T,IH] f32)."""
    B, T, _ = x.shape
    H, qk, dv, kvr = cfg.n_heads, cfg.qk_nope, cfg.v_hd, cfg.kv_lora
    q_resid = rmsnorm(x @ p["q_a"], p["q_a_norm"], cfg.eps)
    q = (q_resid @ p["q_b"]).reshape(B, T, H, qk)
    new_c = rmsnorm(x @ p["kv_a"], p["kv_a_norm"], cfg.eps)  # [B,T,kvr]
    Wk = p["kv_b"].reshape(kvr, H, qk + dv)[:, :, :qk]
    out = {"q_lat": jnp.einsum("bthq,rhq->bthr", q, Wk), "new_c": new_c}
    if "indexer" in p:
        ix = p["indexer"]
        IH, ihd = cfg.idx_heads, cfg.idx_hd
        out["nk"] = layernorm(x @ ix["wk"], ix["k_norm_w"], ix["k_norm_b"],
                              1e-6)
        out["ng"] = x @ ix["gate"]
        out["qi"] = (q_resid @ ix["wq_b"]).reshape(B, T, IH,
                                                   ihd).astype(jnp.float32)
        out["wts"] = (x.astype(jnp.float32) @ ix["weights_proj"].astype(
            jnp.float32)) * (IH**-0.5)  # [B,T,IH]
    return out


def _mla_write(p, proj, cfg: Cfg, cache, pos0, length):
    """The cache update: the new latents at pos0; with an indexer its pooled keys and tails."""
    new_cache = {"c": cache_write(cache["c"], proj["new_c"], pos0)}
    if "indexer" in p:
        pooled, complete, pool0, tk2, tg2 = pool_new_keys(
            p["indexer"], cache["tk"], cache["tg"], proj["nk"], proj["ng"],
            pos0, cfg, length)
        new_cache["pk"] = cache_write(cache["pk"],
                                      pooled,
                                      pool0,
                                      valid=complete)
        new_cache["tk"], new_cache["tg"] = tk2.astype(
            cache["tk"].dtype), tg2.astype(cache["tg"].dtype)
    return new_cache


def _indexer_scores(qi, wts, pool_keys, cfg: Cfg):
    """qi [B,Tq,IH,ihd], wts [B,Tq,IH], pool_keys [B,P,ihd] f32 -> [B,Tq,P]."""
    sc = jax.nn.relu(
        jnp.einsum("bthd,bpd->bthp", qi, pool_keys) * (cfg.idx_hd**-0.5))
    return jnp.einsum("bth,bthp->btp", wts, sc)


def _per_query_keys(c_all, pools, pvalid, qpos, cfg: Cfg):
    """The latents of every query's selected tokens: (keys [B,Tq,W,kvr], valid [B,Tq,W])."""
    tok = select_tokens(pools, pvalid, qpos, cfg)  # [B,Tq,W]
    valid = (tok >= 0) & (tok <= qpos[None, :, None])
    slot = jnp.where(valid, tok, 0)
    c_sel = jnp.take_along_axis(c_all[:, None],
                                slot[..., None],
                                axis=2,
                                mode="promise_in_bounds")
    return c_sel, valid


def mla_core(p,
             proj,
             cfg: Cfg,
             cache=None,
             pos0=0,
             sparse=None,
             cap=None,
             length=None):
    """Position-dependent part of an MLA layer: cache update, DSA selection and attention -> (o_lat [B,T,H,kvr],
    new_cache). sparse: None -> dense causal; "auto" -> DSA when capacity > idx_topk; "always"."""
    q_lat, new_c = proj["q_lat"], proj["new_c"]
    B, T, H, kvr = q_lat.shape
    kp = cfg.idx_kpool
    has_ix = "indexer" in p
    if cache is None:
        S = T if cap is None else cap
        S_alloc = -(-S // kp) * kp  # a whole number of pools
        cache = {
            k: jnp.zeros(sh, new_c.dtype)
            for k, sh in mla_cache_shapes(cfg, B, S_alloc, has_ix).items()
        }
        if S_alloc != S:
            cache["c"] = cache["c"][:, :S]
    S = cache["c"].shape[1]
    cfg = dataclasses.replace(cfg, cache_cap=S)
    new_cache = _mla_write(p, proj, cfg, cache, pos0,
                           T if length is None else length)
    c_all = new_cache["c"]
    qpos = pos0 + jnp.arange(T)
    use_sparse = sparse == "always" or (sparse == "auto" and S > cfg.idx_topk)
    if use_sparse:
        assert has_ix, "sparse attention needs an indexer"
        pool_keys = new_cache["pk"].astype(
            jnp.float32
        )  # only complete pools are written; later ones are masked

    def block(q_lat_b, qi_b, wts_b, qpos_b):
        if use_sparse:
            sc = _indexer_scores(qi_b, wts_b, pool_keys, cfg)  # [B,Tq,P]
            pools, pvalid = indexer_select(sc, qpos_b, cfg)
            c_sel, valid = _per_query_keys(c_all, pools, pvalid, qpos_b, cfg)
            return _attend(q_lat_b, c_sel, valid, cfg, True)
        return _attend(q_lat_b, c_all,
                       jnp.arange(S)[None, :] <= qpos_b[:, None], cfg, False)

    qi = proj.get("qi") if use_sparse else None
    wts = proj.get("wts") if use_sparse else None
    Tq = min(cfg.q_block, T)
    nb = -(-T // Tq)
    if nb == 1:
        o_lat = block(q_lat, qi, wts, qpos)
    else:
        Tpad = nb * Tq - T

        def blk(a):  # [B,T,...] -> [nb,B,Tq,...] (queries padded to nb*Tq)
            if Tpad:
                a = jnp.pad(a, ((0, 0), (0, Tpad)) + ((0, 0), ) * (a.ndim - 2))
            return jnp.moveaxis(a.reshape((B, nb, Tq) + a.shape[2:]), 1, 0)

        qpos_blk = jnp.pad(qpos, (0, Tpad)).reshape(nb, Tq)
        xs = (blk(q_lat), blk(qi) if use_sparse else None,
              blk(wts) if use_sparse else None, qpos_blk)
        o_lat = lax.map(lambda t: block(*t), xs)  # [nb,B,Tq,H,kvr]
        o_lat = jnp.moveaxis(o_lat, 0, 1).reshape(B, nb * Tq, H, kvr)[:, :T]
    return o_lat, new_cache


def mla_out(p, o_lat, cfg: Cfg):
    """o_lat [B,T,H,kvr] -> value up-projection + output projection [B,T,D]."""
    B, T, H, kvr = o_lat.shape
    qk, dv = cfg.qk_nope, cfg.v_hd
    Wv = p["kv_b"].reshape(kvr, cfg.n_heads, qk + dv)[:, :, qk:]
    o = jnp.einsum("bthr,rhv->bthv", o_lat, Wv).reshape(B, T, H * dv)
    return o @ p["o"]


def mla_attention(p,
                  x,
                  cfg: Cfg,
                  cache=None,
                  pos0=0,
                  sparse=None,
                  cap=None,
                  length=None):
    """Causal NoPE MLA with absorbed kv_b and a fixed-capacity latent cache. x [B,T,D]. cache: None (prefill;
    allocates `cap` slots, or exactly T when cap is None) or {"c", "pk", "tk", "tg"}; new tokens are written at pos0
    (may be traced). Queries are processed in blocks of cfg.q_block."""
    proj = mla_project(p, x, cfg)
    o_lat, new_cache = mla_core(p, proj, cfg, cache, pos0, sparse, cap, length)
    return mla_out(p, o_lat, cfg), new_cache


# ----------------------------------------------------------------------------- MoE
def moe_route(p, x, cfg: Cfg):
    """x [N,D] -> (topk_idx [N,k] int32, topk_w [N,k] fp32). Sigmoid scores; the bias only steers the choice."""
    logits = x.astype(jnp.float32) @ p["router_w"].astype(jnp.float32)
    scores = jax.nn.sigmoid(logits)
    choice = scores + p["router_bias"].astype(jnp.float32)
    _, idx = lax.top_k(choice, cfg.topk)
    w = jnp.take_along_axis(scores, idx, -1)
    if cfg.norm_topk:
        w = w / (w.sum(-1, keepdims=True) + 1e-20)
    return idx.astype(jnp.int32), w * cfg.scaling


def moe_dense(p, x, cfg: Cfg):
    """MoE block for x [B,T,D] with on-device expert tables p["gate_up"] [E,D,2mi], p["down"] [E,mi,D]."""
    B, T, D = x.shape
    xf = x.reshape(-1, D)
    idx, w = moe_route(p, xf, cfg)
    gu_w, dn_w = p["gate_up"][idx], p["down"][idx]  # [N,k,D,2mi], [N,k,mi,D]
    gu = jnp.einsum("nd,nkdm->nkm", xf, gu_w)
    mi = gu.shape[-1] // 2
    h = swiglu_clamped(gu[..., :mi], gu[..., mi:], cfg.swiglu_limit)
    y = jnp.einsum("nkm,nkmd->nkd", h, dn_w)
    y = jnp.einsum("nkd,nk->nd", y.astype(jnp.float32), w).astype(xf.dtype)
    y = y + mlp(p["shared"], xf, cfg)
    return y.reshape(B, T, D)


# ----------------------------------------------------------------------------- layer / model
def decoder_layer(p,
                  streams,
                  cfg: Cfg,
                  ltype,
                  mtype,
                  cache=None,
                  pos0=0,
                  use_recurrent=False,
                  sparse="auto",
                  cap=None,
                  length=None):
    post, comb, h = hc_pre(p["hc_attn"], streams, cfg)
    h = rmsnorm(h, p["ln1"], cfg.eps)
    if ltype == "linear_attention":
        h, new_cache = kda_attention(p["attn"], h, cfg, cache, use_recurrent)
    else:
        h, new_cache = mla_attention(p["attn"], h, cfg, cache, pos0, sparse,
                                     cap, length)
    streams = hc_post(post, comb, h, streams)
    post, comb, h = hc_pre(p["hc_ffn"], streams, cfg)
    h = rmsnorm(h, p["ln2"], cfg.eps)
    h = moe_dense(p["mlp"], h, cfg) if mtype == "sparse" else mlp(
        p["mlp"], h, cfg)
    return hc_post(post, comb, h, streams), new_cache


def forward(params,
            tokens,
            cfg: Cfg,
            caches=None,
            pos0=0,
            use_recurrent=False,
            sparse="auto",
            cap=None,
            n_layers=None):
    """tokens [B,T] -> (hidden [B,T,D] after the final norm, new_caches list). Prefill: caches=None.
    `n_layers` (optional) runs only the first n layers (truncated-model parity tests)."""
    x = params["embed"][tokens].astype(cfg.dtype)
    streams = jnp.broadcast_to(x[:, :, None, :],
                               x.shape[:2] + (cfg.hc, x.shape[-1]))
    new_caches = []
    for i in range(cfg.n_layers if n_layers is None else n_layers):
        c = None if caches is None else caches[i]
        streams, nc = decoder_layer(params["layers"][i],
                                    streams,
                                    cfg,
                                    cfg.layer_types[i],
                                    cfg.mlp_types[i],
                                    c,
                                    pos0,
                                    use_recurrent,
                                    sparse,
                                    cap=cap)
        new_caches.append(nc)
    h = rmsnorm(streams.mean(2).astype(cfg.dtype), params["norm"],
                cfg.eps)  # HF Glm5NextTextHyperHead: plain mean
    return h, new_caches


def logits(params, h):
    return h @ params["lm_head"]


def mtp_layer(p,
              embeds,
              h_prev,
              cfg: Cfg,
              cache=None,
              pos0=0,
              sparse="auto",
              cap=None):
    """GLM-5 MTP (NextN) block: x = eh_proj(cat(rmsnorm(embeds, enorm), rmsnorm(h_prev, hnorm))) — embeds [B,T,D] of
    the token AFTER each position, h_prev [B,T,D] the main model's post-norm hidden at that position — then one plain
    pre-norm MLA + MoE block (no hyper-connections) and the shared-head norm. Returns (h [B,T,D] for the shared
    lm_head, new attention cache)."""
    x = jnp.concatenate([
        rmsnorm(embeds.astype(cfg.dtype), p["enorm"], cfg.eps),
        rmsnorm(h_prev.astype(cfg.dtype), p["hnorm"], cfg.eps)
    ], -1) @ p["eh_proj"]
    h = rmsnorm(x, p["ln1"], cfg.eps)
    a, new_cache = mla_attention(p["attn"], h, cfg, cache, pos0, sparse, cap)
    x = x + a
    h = rmsnorm(x, p["ln2"], cfg.eps)
    x = x + moe_dense(p["mlp"], h, cfg)
    return rmsnorm(x, p["head_norm"], cfg.eps), new_cache
