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
"""Recurrent KDA steps in one Pallas kernel: decode and speculative windows.

For each request r and head h the kernel takes the state before its window
S [dk, dv] (f32, in VMEM), runs up to G recurrent steps

    S <- diag(exp(g_t)) S ;  S <- S + k_t (beta_t (v_t - k_t^T S))^T ;
    o_t = q_t^T S

and writes the state after every valid step straight to its slot of the
(aliased) state array by DMA. Positions with ok = 0 leave S unchanged and
write nothing. q, k and g come as columns ([dk, G]) so every step is
elementwise work on whole [dk, dv] tiles plus sublane reductions, without
transposes.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(write_ref, ok_ref, s0_ref, q_ref, k_ref, g_ref, v_ref, b_ref,
            state_in, out_ref, state_out, s_buf, sems, *, G: int):
    del state_in  # aliased with state_out
    r, h = pl.program_id(0), pl.program_id(1)
    S = s0_ref[0, 0]  # [dk, dv] f32
    copies = []
    for t in range(G):
        kt = k_ref[0, 0, :, t:t + 1]  # [dk, 1]
        S_dec = S * jnp.exp(g_ref[0, 0, :, t:t + 1])
        kv = jnp.sum(kt * S_dec, axis=0, keepdims=True)  # [1, dv]
        delta = (v_ref[0, 0, t:t + 1, :] - kv) * b_ref[0, 0, :, t:t + 1]
        S_new = S_dec + kt * delta
        ok = ok_ref[r, t] > 0
        S = jnp.where(ok, S_new, S)
        out_ref[0, 0, t:t + 1, :] = jnp.sum(q_ref[0, 0, :, t:t + 1] * S,
                                            axis=0,
                                            keepdims=True)
        s_buf[t] = S
        cp = pltpu.make_async_copy(s_buf.at[t], state_out.at[write_ref[r, t],
                                                             h], sems.at[t])
        copies.append((ok, cp))

        @pl.when(ok)
        def _start(cp=cp):
            cp.start()

    for ok, cp in copies:

        @pl.when(ok)
        def _wait(cp=cp):
            cp.wait()


@functools.partial(jax.jit,
                   donate_argnames=("state", ),
                   static_argnames=("interpret", ))
def kda_recurrent(s0, q, k, g, v, beta, state, write, ok, interpret=False):
    """s0 [R, H, dk, dv] f32: each request's state before its window.
    q, k, g [R, H, dk, G] (q l2-normalized and scaled, k l2-normalized, g
    the log decays, all f32); v [R, H, G, dv] f32; beta [R, H, 1, G] f32;
    state [N, H, dk, dv] f32; write [R, G] int32 (slot of the state after
    each position), ok [R, G] int32 (valid positions; the others are no-ops
    and write nothing). Returns (state, out [R, H, G, dv] f32)."""
    R, H, dk, G = q.shape
    dv = v.shape[-1]
    blk = lambda *shape: pl.BlockSpec(  # noqa: E731
        (1, 1) + shape, lambda r, h, *_: (r, h) + (0, ) * len(shape))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(R, H),
        in_specs=[
            blk(dk, dv),  # s0
            blk(dk, G),  # q
            blk(dk, G),  # k
            blk(dk, G),  # g
            blk(G, dv),  # v
            blk(1, G),  # beta
            pl.BlockSpec(memory_space=pltpu.HBM),  # state
        ],
        out_specs=[
            blk(G, dv),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        scratch_shapes=[
            pltpu.VMEM((G, dk, dv), jnp.float32),
            pltpu.SemaphoreType.DMA((G, )),
        ],
    )
    out, state = pl.pallas_call(
        functools.partial(_kernel, G=G),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((R, H, G, dv), jnp.float32),
            jax.ShapeDtypeStruct(state.shape, state.dtype),
        ],
        # operands: write, ok, s0, q, k, g, v, beta, state -> state is 8
        input_output_aliases={8: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary")),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(write, ok, s0, q, k, g, v, beta, state)
    return state, out
