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
"""The chunked KDA state recurrence in one Pallas kernel.

The chunked KDA form splits into per-chunk terms that don't depend on the
state (computed batched over all chunks in XLA: `_kda_chunk_prep`) and the
sequential recurrence over a stream of chunks:

    S <- init[seq]                      at a sequence's first chunk
    v_new = u - w S
    out   = q_dec S + attn v_new
    S     <- diag(decay) S + k_dec v_new

The grid is (head, chunk) with the chunks innermost and sequential; S lives
in VMEM across them. Each sequence's initial state block is fetched by its
index map, and its final state is the output block of the sequence id,
written back when the next chunk belongs to another sequence.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(seq_ref, reset_ref, u_ref, w_ref, attn_ref, q_ref, k_ref,
            decay_ref, init_ref, out_ref, final_ref, s_ref):
    i = pl.program_id(1)

    @pl.when(reset_ref[i] > 0)
    def _load():
        s_ref[...] = init_ref[0, 0]

    S = s_ref[...]  # [dk, dv] f32
    dot = functools.partial(jnp.dot,
                            preferred_element_type=jnp.float32,
                            precision=jax.lax.Precision.HIGHEST)
    v_new = u_ref[0, 0] - dot(w_ref[0, 0], S)  # [c, dv]
    out_ref[0, 0] = dot(q_ref[0, 0], S) + dot(attn_ref[0, 0], v_new)
    S = S * decay_ref[0, 0] + dot(k_ref[0, 0], v_new)
    s_ref[...] = S
    final_ref[0, 0] = S


@functools.partial(jax.jit, static_argnames=("interpret", ))
def kda_chunk_scan(u,
                   w,
                   attn,
                   q_dec,
                   k_dec,
                   decay,
                   seq,
                   reset,
                   init,
                   interpret=False):
    """u [n, H, c, dv], w and q_dec [n, H, c, dk], attn [n, H, c, c], k_dec
    [n, H, dk, c], decay [n, H, dk, 1] (all f32): the per-chunk terms of a
    stream of n chunks. seq [n] int32: each chunk's sequence id in
    [0, num_seqs); reset [n] int32: 1 at each sequence's first chunk; init
    [num_seqs, H, dk, dv] f32: the sequences' initial states. Chunks of a
    sequence must be consecutive. Returns (out [n, H, c, dv] f32, final
    [num_seqs, H, dk, dv] f32: each sequence's state after its last chunk;
    sequences without chunks keep garbage)."""
    n, H, c, dv = u.shape
    dk = w.shape[-1]
    num_seqs = init.shape[0]
    chunk = lambda *shape: pl.BlockSpec(  # noqa: E731
        (1, 1) + shape, lambda h, i, *_: (i, h) + (0, ) * len(shape))
    per_seq = pl.BlockSpec((1, 1, dk, dv), lambda h, i, seq_ref, _:
                           (seq_ref[i], h, 0, 0))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(H, n),
        in_specs=[
            chunk(c, dv),  # u
            chunk(c, dk),  # w
            chunk(c, c),  # attn
            chunk(c, dk),  # q_dec
            chunk(dk, c),  # k_dec
            chunk(dk, 1),  # decay
            per_seq,  # init
        ],
        out_specs=[chunk(c, dv), per_seq],
        scratch_shapes=[pltpu.VMEM((dk, dv), jnp.float32)],
    )
    return pl.pallas_call(
        _kernel,
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((n, H, c, dv), jnp.float32),
            jax.ShapeDtypeStruct((num_seqs, H, dk, dv), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(seq, reset, u, w, attn, q_dec, k_dec, decay, init)
