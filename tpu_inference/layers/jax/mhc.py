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
"""Manifold-constrained hyper-connections (mHC) for token-major residual
streams [T, hc, D].

One site (before attention, before the MLP) mixes the hc residual streams:

    flat  = RMS(streams.reshape(T, hc * D))            (no weight)
    mix   = flat @ fn.T                                [T, (2 + hc) * hc]
    pre   = sigmoid(mix[:, :hc] * scale[0] + base[:hc]) + eps
    post  = 2 * sigmoid(mix[:, hc:2hc] * scale[1] + base[hc:2hc])
    comb  = Sinkhorn(softmax(mix[:, 2hc:] * scale[2] + base[2hc:]) + eps)
    x     = sum_h pre_h * streams_h                    -> the block's input
    new_k = post_k * block(x) + sum_h comb[h, k] * streams_h

`fn` is stored as in the checkpoint, [(2 + hc) * hc, hc * D]. On TPU (bf16
streams) the sites run as the Pallas kernels of
`kernels/experimental/deepseek_v4/mhc`, which compute the same gates; elsewhere
(or with `use_kernel=False`) as the XLA functions below. `post` is returned as
[T, hc] and `comb` as [T, hc, hc] f32 in both cases.
"""
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.sharding import ShardingAxisName


class MHCConfig(NamedTuple):
    hc: int
    sinkhorn_iters: int
    hc_eps: float
    rms_eps: float
    post_mult: float = 2.0


def mhc_kernel_available(dtype) -> bool:
    return jax.default_backend() == "tpu" and dtype == jnp.bfloat16


# ----------------------------------------------------------------------------- XLA
def mhc_pre_xla(streams, fn, scale, base, cfg: MHCConfig):
    """streams [T, hc, D] -> (post [T, hc] f32, comb [T, hc, hc] f32,
    collapsed [T, D] in streams' dtype)."""
    T, hc, D = streams.shape
    f32 = jnp.float32
    x = streams.reshape(T, hc * D).astype(f32)
    flat = x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + cfg.rms_eps)
    mix = jnp.einsum("tk,mk->tm",
                     flat,
                     fn.astype(f32),
                     precision=jax.lax.Precision.HIGHEST)
    scale, base = scale.astype(f32), base.astype(f32)
    pre = jax.nn.sigmoid(mix[:, :hc] * scale[0] + base[:hc]) + cfg.hc_eps
    post = cfg.post_mult * jax.nn.sigmoid(mix[:, hc:2 * hc] * scale[1] +
                                          base[hc:2 * hc])
    logits = (mix[:, 2 * hc:].reshape(T, hc, hc) * scale[2] +
              base[2 * hc:].reshape(hc, hc))
    comb = jax.nn.softmax(logits, -1) + cfg.hc_eps
    comb = comb / (comb.sum(-2, keepdims=True) + cfg.hc_eps)  # columns
    for _ in range(cfg.sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + cfg.hc_eps)  # rows
        comb = comb / (comb.sum(-2, keepdims=True) + cfg.hc_eps)  # columns
    collapsed = jnp.einsum("th,thd->td", pre,
                           streams.astype(f32)).astype(streams.dtype)
    return post, comb, collapsed


def mhc_post_xla(y, streams, post, comb):
    """new_streams[k] = post[k] * y + sum_h comb[h, k] * streams[h]."""
    dt = streams.dtype
    return (post.astype(dt)[..., None] * y[:, None, :] +
            jnp.einsum("thk,thd->tkd", comb.astype(dt), streams))


# ----------------------------------------------------------------------------- kernels
def _token_sharded(fn, mesh, n_token_args, n_token_outs):
    """Mosaic custom calls are not auto-partitioned: run them per shard of the
    attention-DP token axis (parameters replicated)."""
    if mesh is None:
        return fn
    data = P(ShardingAxisName.ATTN_DATA)

    def wrapped(*args):
        in_specs = (data, ) * n_token_args + (P(), ) * (len(args) -
                                                        n_token_args)
        out = (data, ) * n_token_outs
        return jax.shard_map(fn,
                             mesh=mesh,
                             in_specs=in_specs,
                             out_specs=out if n_token_outs > 1 else data,
                             check_vma=False)(*args)

    return wrapped


def _kernel_pre(streams, fn, scale, base, cfg: MHCConfig, mesh):
    from tpu_inference.kernels.experimental.deepseek_v4.mhc import pre_kernel
    run = functools.partial(pre_kernel.mhc_pre,
                            rms_eps=cfg.rms_eps,
                            hc_pre_eps=cfg.hc_eps,
                            hc_sinkhorn_eps=cfg.hc_eps,
                            hc_post_mult_value=cfg.post_mult,
                            sinkhorn_repeat=cfg.sinkhorn_iters)
    post, comb, x = _token_sharded(run, mesh, 1, 3)(streams,
                                                    fn.astype(jnp.float32),
                                                    scale.astype(jnp.float32),
                                                    base.astype(jnp.float32))
    return post[..., 0], comb, x


def _kernel_post(y, streams, post, comb, mesh):
    from tpu_inference.kernels.experimental.deepseek_v4.mhc import post_kernel
    return _token_sharded(post_kernel.mhc_post, mesh, 4,
                          1)(y.astype(streams.dtype), streams, post[..., None],
                             comb)


def _kernel_post_pre(y, streams, post, comb, fn, scale, base, cfg: MHCConfig,
                     mesh):
    from tpu_inference.kernels.experimental.deepseek_v4.mhc import \
        fused_post_pre_kernel
    run = functools.partial(fused_post_pre_kernel.mhc_fused_post_pre,
                            rms_eps=cfg.rms_eps,
                            hc_pre_eps=cfg.hc_eps,
                            hc_sinkhorn_eps=cfg.hc_eps,
                            hc_post_mult_value=cfg.post_mult,
                            sinkhorn_repeat=cfg.sinkhorn_iters)
    new, post2, comb2, x = _token_sharded(run, mesh, 4,
                                          4)(y.astype(streams.dtype), streams,
                                             post[..., None], comb,
                                             fn.astype(jnp.float32),
                                             scale.astype(jnp.float32),
                                             base.astype(jnp.float32))
    return new, post2[..., 0], comb2, x


# ----------------------------------------------------------------------------- dispatch
def mhc_pre(streams,
            fn,
            scale,
            base,
            cfg: MHCConfig,
            *,
            use_kernel: bool,
            mesh=None):
    """The first site of a layer: (post, comb, block input)."""
    if use_kernel:
        return _kernel_pre(streams, fn, scale, base, cfg, mesh)
    return mhc_pre_xla(streams, fn, scale, base, cfg)


def mhc_post(y, streams, post, comb, *, use_kernel: bool, mesh=None):
    """The last site of the model: the residual update only."""
    if use_kernel:
        return _kernel_post(y, streams, post, comb, mesh)
    return mhc_post_xla(y, streams, post, comb)


def mhc_post_pre(y,
                 streams,
                 post,
                 comb,
                 fn,
                 scale,
                 base,
                 cfg: MHCConfig,
                 *,
                 use_kernel: bool,
                 mesh=None):
    """A block's residual update fused with the next site's pre: (new
    streams, post, comb, next block input)."""
    if use_kernel:
        return _kernel_post_pre(y, streams, post, comb, fn, scale, base, cfg,
                                mesh)
    new = mhc_post_xla(y, streams, post, comb)
    return (new, ) + mhc_pre_xla(new, fn, scale, base, cfg)


def mhc_head(streams):
    """Final collapse of the streams before the model's last norm. GLM-5.3-
    Flash uses an unweighted mean (DeepSeek-V4 a learned weighting)."""
    return streams.mean(axis=1)
