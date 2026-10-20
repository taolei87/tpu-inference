# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM-5.3-Flash vision tower (HF `Glm5NextVisionModel`, weights
`model.visual.*`, bf16).

patches [N, 3 * 2 * 14 * 14] (the processor's block-major order: the 4
patches of a 2x2 merge block are consecutive) -> patch embedding (the Conv3d
with stride == kernel is a linear map over the 1176 inputs) -> 24 pre-norm
blocks (RMSNorm; attention with q/k RMSNorm over head_dim and 2-D rotary over
the (h, w) patch positions, full attention within each frame; clamped SwiGLU
MLP; all with biases) -> post RMSNorm -> the 2x2 downsample Conv2d (a linear
map over (c, i, j)) -> merger (proj, LayerNorm, GELU, clamped SwiGLU) ->
[N / 4, out_hidden_size], one text embedding per merge block, in the order of
the prompt's image tokens.

Linear weights are stored [in, out] (the loader transposes HF's [out, in]);
the two convs keep their HF shapes. All replicated: the tower is ~1.1 GB.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.attention_interface import \
    sharded_flash_attention
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule

init_fn = nnx.initializers.uniform()
_FLASH_BLOCK = 128  # flash attention pads the sequence to this multiple
Grid = Tuple[int, int, int]  # (t, h, w) in patches


class _Node(JaxModule):
    """A named level of the HF module tree (e.g. `attn.qkv`) holding params."""


# Big weights are split over the tensor-parallel axes (columns of the
# input-side projections, rows of the output-side ones) so the ~1.1 GB tower
# doesn't take its full size from every chip's KV cache; XLA inserts the
# collectives. Norms and the small patch embedding stay replicated.
TP = ShardingAxisName.ATTN_HEAD
COLS, ROWS = (None, TP), (TP, None)


def _param(rng, shape, dtype, sharding=None):
    return nnx.Param(init_fn(rng.params(), shape, dtype),
                     out_sharding=sharding or (None, ) * len(shape),
                     eager_sharding=False)


def _rms(x, w, eps):
    xf = x.astype(jnp.float32)
    xf = xf * jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + eps)
    return (xf * w.astype(jnp.float32)).astype(x.dtype)


def _swiglu(g, u, limit):
    return jax.nn.silu(jnp.minimum(g, limit)) * jnp.clip(u, -limit, limit)


def _dense(x, w, b=None):
    y = jnp.dot(x, w[...].astype(x.dtype), preferred_element_type=jnp.float32)
    return y if b is None else y + b[...].astype(jnp.float32)


def position_ids(grid: Grid, merge: int) -> np.ndarray:
    """[t * h * w, 2] (row, col) of every patch, in the block-major order."""
    t, h, w = grid
    hh, ww = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    shape = (h // merge, merge, w // merge, merge)
    hh = hh.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    ww = ww.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    return np.tile(np.stack([hh, ww], -1), (t, 1)).astype(np.int32)


def rotary(pos: jax.Array, head_dim: int, theta: float = 10000.0):
    """cos, sin [N, head_dim] for (row, col) positions [N, 2]: 2-D rotary,
    head_dim // 4 frequencies per axis, (row freqs, col freqs) repeated twice
    (rotate-half layout)."""
    d = head_dim // 2
    inv = 1.0 / (theta**(np.arange(0, d, 2, dtype=np.float32) / d))
    f = (pos.astype(jnp.float32)[:, :, None] * inv[None, None, :]).reshape(
        pos.shape[0], -1)
    emb = jnp.concatenate([f, f], -1)
    return jnp.cos(emb), jnp.sin(emb)


def bucket_inputs(grid: Grid, merge: int, min_bucket: int = 256):
    """(n, padded n, positions [padded, 2], frame segments [padded]) for one
    grid: the patch count padded to a power of two (at least min_bucket), so
    the encoder compiles once per bucket rather than per image size. Padding
    rows are segment -1: they attend only to each other and are sliced off."""
    t, h, w = grid
    n = t * h * w
    padded = max(min_bucket, 1 << (n - 1).bit_length())
    pos = np.zeros((padded, 2), np.int32)
    pos[:n] = position_ids(grid, merge)
    seg = np.full(padded, -1, np.int32)
    seg[:n] = np.repeat(np.arange(t, dtype=np.int32), h * w)
    return n, padded, pos, seg


def _rotate_half(x):
    a, b = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-b, a], -1)


class Glm5NextVisionAttention(JaxModule):

    def __init__(self, dim: int, num_heads: int, eps: float, dtype, rng):
        self.num_heads, self.head_dim, self.eps = num_heads, dim // num_heads, eps
        self.qkv = _Node()
        self.qkv.weight = _param(rng, (dim, 3 * dim), dtype, COLS)
        self.qkv.bias = _param(rng, (3 * dim, ), dtype)
        self.q_norm = _Node()
        self.q_norm.weight = _param(rng, (self.head_dim, ), dtype)
        self.k_norm = _Node()
        self.k_norm.weight = _param(rng, (self.head_dim, ), dtype)
        self.proj = _Node()
        self.proj.weight = _param(rng, (dim, dim), dtype, ROWS)
        self.proj.bias = _param(rng, (dim, ), dtype)

    def __call__(self, x, cos, sin, segments, attend):
        N, C = x.shape
        H, hd = self.num_heads, self.head_dim
        qkv = _dense(x, self.qkv.weight, self.qkv.bias).astype(x.dtype)
        qkv = qkv.reshape(N, 3, H, hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = _rms(q, self.q_norm.weight[...], self.eps)
        k = _rms(k, self.k_norm.weight[...], self.eps)
        c, s = cos[:, None, :], sin[:, None, :]
        q = (q.astype(jnp.float32) * c +
             _rotate_half(q.astype(jnp.float32)) * s).astype(x.dtype)
        k = (k.astype(jnp.float32) * c +
             _rotate_half(k.astype(jnp.float32)) * s).astype(x.dtype)
        o = attend(q, k, v, segments).reshape(N, C)
        return _dense(o.astype(x.dtype), self.proj.weight, self.proj.bias)


class Glm5NextVisionMLP(JaxModule):

    def __init__(self, dim: int, hidden: int, limit: float, dtype, rng,
                 bias: bool):
        self.limit = limit
        for name, shape in (("gate_proj", (dim, hidden)),
                            ("up_proj", (dim, hidden)), ("down_proj", (hidden,
                                                                       dim))):
            m = _Node()
            m.weight = _param(rng, shape, dtype,
                              ROWS if name == "down_proj" else COLS)
            if bias:
                m.bias = _param(rng, (shape[1], ), dtype)
            setattr(self, name, m)
        self.bias = bias

    def __call__(self, x):
        b = (lambda m: m.bias) if self.bias else (lambda m: None)
        g = _dense(x, self.gate_proj.weight, b(self.gate_proj))
        u = _dense(x, self.up_proj.weight, b(self.up_proj))
        h = _swiglu(g, u, self.limit).astype(x.dtype)
        return _dense(h, self.down_proj.weight, b(self.down_proj))


class Glm5NextVisionBlock(JaxModule):

    def __init__(self, cfg, dtype, rng):
        dim, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.eps = eps
        self.norm1 = _Node()
        self.norm1.weight = _param(rng, (dim, ), dtype)
        self.norm2 = _Node()
        self.norm2.weight = _param(rng, (dim, ), dtype)
        self.attn = Glm5NextVisionAttention(dim, cfg.num_heads, eps, dtype,
                                            rng)
        self.mlp = Glm5NextVisionMLP(dim, cfg.intermediate_size,
                                     cfg.swiglu_limit, dtype, rng, True)

    def __call__(self, x, cos, sin, segments, attend):
        h = self.attn(_rms(x, self.norm1.weight[...], self.eps), cos, sin,
                      segments, attend)
        x = x + h.astype(x.dtype)
        h = self.mlp(_rms(x, self.norm2.weight[...], self.eps))
        return x + h.astype(x.dtype)


def _masked_attention(q, k, v, segments):
    """Reference attention (any backend): q, k, v [N, H, d]; segments [N]."""
    scale = q.shape[-1]**-0.5
    s = jnp.einsum("nhd,mhd->hnm", q, k,
                   preferred_element_type=jnp.float32) * scale
    s = jnp.where((segments[:, None] == segments[None, :])[None], s, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("hnm,mhd->nhd",
                      p.astype(v.dtype),
                      v,
                      preferred_element_type=jnp.float32)


class Glm5NextVisionModel(JaxModule):
    """`model.visual`: patches of one image (or video) grid -> its text
    embeddings."""

    def __init__(self, vision_config, dtype, rng: nnx.Rngs, mesh: Mesh):
        cfg = vision_config
        self.cfg, self.mesh, self.dtype = cfg, mesh, dtype
        dim, out = cfg.hidden_size, cfg.out_hidden_size
        m, p, tp = cfg.spatial_merge_size, cfg.patch_size, cfg.temporal_patch_size
        self.merge = m
        self.patch_embed = _Node()
        self.patch_embed.proj = _Node()
        self.patch_embed.proj.weight = _param(rng,
                                              (dim, cfg.in_channels, tp, p, p),
                                              dtype)
        self.patch_embed.proj.bias = _param(rng, (dim, ), dtype)
        self.blocks = nnx.List(
            [Glm5NextVisionBlock(cfg, dtype, rng) for _ in range(cfg.depth)])
        self.post_layernorm = _Node()
        self.post_layernorm.weight = _param(rng, (dim, ), dtype)
        self.downsample = _Node()
        self.downsample.weight = _param(rng, (out, dim, m, m), dtype,
                                        (TP, None, None, None))
        self.downsample.bias = _param(rng, (out, ), dtype)
        self.merger = Glm5NextVisionMLP(out, cfg.projection_intermediate_size,
                                        cfg.swiglu_limit, dtype, rng, False)
        self.merger.proj = _Node()
        self.merger.proj.weight = _param(rng, (out, out), dtype, COLS)
        self.merger.post_projection_norm = _Node()
        self.merger.post_projection_norm.weight = _param(rng, (out, ), dtype)
        self.merger.post_projection_norm.bias = _param(rng, (out, ), dtype)

    def num_tokens(self, grid: Grid) -> int:
        t, h, w = grid
        return t * h * w // self.merge**2

    def _attend_fn(self, n: int):
        """Attention over [n, H, d] with per-frame segments (-1: padding). On
        TPU the flash attention kernel (heads over the model axis, memory
        linear in n); elsewhere the reference masked softmax."""
        if jax.devices()[0].platform != "tpu":
            return _masked_attention
        from tpu_inference.kernels.flash_attention.kernel import SegmentIds
        hd = self.cfg.hidden_size // self.cfg.num_heads
        flash = sharded_flash_attention(self.mesh,
                                        causal=False,
                                        sm_scale=hd**-0.5,
                                        vmem_limit_bytes=128 * 1024 * 1024)
        padded = -(-n // _FLASH_BLOCK) * _FLASH_BLOCK

        def attend(q, k, v, segments):
            pad = ((0, padded - n), (0, 0), (0, 0))
            # [n, H, d] -> [1, H, padded, d]; padding is segment 0 (alone)
            q, k, v = (jnp.pad(a, pad).transpose(1, 0, 2)[None]
                       for a in (q, k, v))
            seg = jnp.pad(segments + 1, (0, padded - n))[None]
            o = flash(q, k, v, SegmentIds(q=seg, kv=seg))
            return o[0].transpose(1, 0, 2)[:n]

        return attend

    def encode(self, patches: jax.Array, positions: jax.Array,
               segments: jax.Array) -> jax.Array:
        """patches [N, 1176], (row, col) positions [N, 2] and frame segments
        [N] (block-major order; N a multiple of 4) -> [N / 4, out_hidden].
        Every shape is data, so one compile serves every grid of N patches."""
        cfg, dt = self.cfg, self.dtype
        n = patches.shape[0]
        hd = cfg.hidden_size // cfg.num_heads
        cos, sin = rotary(positions, hd)
        attend = self._attend_fn(n)
        pw = self.patch_embed.proj.weight[...]
        x = _dense(patches.astype(dt),
                   pw.reshape(pw.shape[0], -1).T,
                   self.patch_embed.proj.bias).astype(dt)
        for blk in self.blocks:
            x = blk(x, cos, sin, segments, attend)
        x = _rms(x, self.post_layernorm.weight[...], cfg.rms_norm_eps)
        m, C = self.merge, x.shape[-1]
        # 2x2 merge blocks are consecutive: (c, i, j) like the conv weight
        x = x.reshape(n // m**2, m, m,
                      C).transpose(0, 3, 1, 2).reshape(n // m**2, C * m * m)
        dw = self.downsample.weight[...]
        y = _dense(x, dw.reshape(dw.shape[0], -1).T, self.downsample.bias)
        y = _dense(y.astype(dt), self.merger.proj.weight)
        ln = self.merger.post_projection_norm
        mu = y.mean(-1, keepdims=True)
        var = jnp.square(y - mu).mean(-1, keepdims=True)
        y = (y - mu) * jax.lax.rsqrt(var + 1e-5) * ln.weight[...].astype(
            jnp.float32) + ln.bias[...].astype(jnp.float32)
        y = jax.nn.gelu(y, approximate=False).astype(dt)
        return self.merger(y).astype(dt)

    def __call__(self, patches: jax.Array, grid: Grid) -> jax.Array:
        """patches [t * h * w, 1176] -> [num_tokens(grid), out_hidden]."""
        t, h, w = grid
        pos = position_ids(grid, self.merge)
        seg = np.repeat(np.arange(t, dtype=np.int32), h * w)
        return self.encode(patches, jnp.asarray(pos), jnp.asarray(seg))
