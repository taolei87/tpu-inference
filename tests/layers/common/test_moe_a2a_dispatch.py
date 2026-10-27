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
"""Tests for MOE_A2A_DISPATCH: the expert-parallel MoE under attention DP with
the tokens sent to their experts by ragged all-to-all, against the all-gather
dispatch. Needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8
on CPU, where the GMMs run as a pure-JAX stand-in)."""

import math
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)

requires_8_devices = pytest.mark.skipif(len(jax.devices()) < 8,
                                        reason="needs 8 devices")


@pytest.fixture(autouse=True)
def base_sharding_axes():
    with mock.patch.object(ShardingAxisName, "_cls", ShardingAxisNameBase):
        yield


def _cpu_gmm(lhs,
             rhs,
             rhs_scale,
             rhs_bias,
             group_sizes,
             group_offset,
             fuse_act=None,
             preferred_element_type=None):
    """The grouped matmul on rows sorted by group; rows outside the local
    groups give zeros."""
    gid = jnp.searchsorted(jnp.cumsum(group_sizes),
                           jnp.arange(lhs.shape[0]),
                           side="right") - group_offset.reshape(-1)[0]
    ok = (gid >= 0) & (gid < rhs.shape[0])
    out = jnp.einsum(
        "mk,mkn->mn", lhs.astype(jnp.float32),
        rhs[jnp.clip(gid, 0, rhs.shape[0] - 1)].astype(jnp.float32))
    if fuse_act is not None:
        n = out.shape[-1] // 2
        out = jax.nn.silu(out[:, :n]) * out[:, n:]
    return jnp.where(ok[:, None], out, 0).astype(preferred_element_type
                                                 or lhs.dtype)


@pytest.fixture(autouse=True)
def cpu_gmm():
    if jax.default_backend() == "tpu":
        yield
        return
    with mock.patch.object(fused_moe_gmm, "gmm_wrapper", _cpu_gmm):
        yield


def _make_mesh(**sizes) -> Mesh:
    shape = tuple(sizes.get(a, 1) for a in MESH_AXIS_NAMES)
    devices = sorted(jax.devices(), key=lambda d: d.id)[:math.prod(shape)]
    return jax.make_mesh(shape,
                         MESH_AXIS_NAMES, (AxisType.Auto, ) * len(shape),
                         devices=devices)


def _run(mesh,
         a2a,
         valid,
         fp8=False,
         num_tokens=64,
         hidden=256,
         num_experts=16):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(3), 4)
    rows = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    dtype = jnp.bfloat16 if fp8 else jnp.float32
    jax.clear_caches()
    try:
        with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
            out = fused_moe_gmm.fused_moe_func(
                hidden_states=jax.device_put(
                    jax.random.normal(k0, (num_tokens, hidden), dtype), rows),
                w1=jax.random.normal(k1,
                                     (num_experts, hidden, 512), dtype) / 10,
                w2=jax.random.normal(k2,
                                     (num_experts, 256, hidden), dtype) / 10,
                w1_scale=None,
                w2_scale=None,
                w1_bias=None,
                w2_bias=None,
                gating_output=jax.device_put(
                    jax.random.normal(k3, (num_tokens, num_experts), dtype),
                    rows),
                topk=4,
                renormalize=True,
                mesh=mesh,
                use_ep=True,
                activation="silu",
                scoring_fn="softmax",
                all_gather_fp8=fp8,
                onehot_moe_permute_threshold=1 << 20,
                num_valid_tokens=None if valid is None else jax.device_put(
                    valid, NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA))),
                a2a_dispatch=a2a)
            return np.asarray(jax.block_until_ready(out), np.float32)
    finally:
        jax.clear_caches()


def _rank_mask(num_tokens, ranks, real):
    """Each rank's segment starts with real[r] real tokens."""
    seg = num_tokens // ranks
    return jnp.asarray(
        np.concatenate([np.arange(seg) < n for n in real]).astype(bool))


@requires_8_devices
@pytest.mark.parametrize("recv_rows", [None, 16], ids=["one_round", "rounds"])
@pytest.mark.parametrize("sizes, real", [
    (dict(attn_dp=2, model=4), [32, 32]),
    (dict(attn_dp=2, model=4), [32, 0]),
    (dict(attn_dp=4, model=2), [5, 0, 16, 1]),
    (dict(data=2, attn_dp=2, model=2), [16, 3, 0, 9]),
    (dict(attn_dp=8), [8, 0, 0, 1, 0, 0, 0, 2]),
],
                         ids=[
                             "dp2xtp4_full", "dp2xtp4_lone", "dp4xtp2_ragged",
                             "data2xdp2xtp2", "dp8xtp1"
                         ])
def test_a2a_dispatch_matches_all_gather(sizes, real, recv_rows):
    """The real tokens' outputs match the all-gather dispatch's (f32), with
    padding tokens in every rank's segment (a lone prefill: one rank busy) and
    with receive buffers that take several rounds."""
    mesh = _make_mesh(**sizes)
    valid = _rank_mask(64, len(real), real)
    expected = _run(mesh, False, valid)
    patch = (mock.patch.object(fused_moe_gmm, "_a2a_recv_rows",
                               lambda *_: recv_rows)
             if recv_rows else mock.patch.object(fused_moe_gmm, "logger"))
    with patch:
        actual = _run(mesh, True, valid)
    v = np.asarray(valid)
    np.testing.assert_allclose(actual[v], expected[v], rtol=1e-5, atol=1e-5)
    # padding tokens send nothing: their output is 0
    assert not actual[~v].any()


@requires_8_devices
def test_a2a_dispatch_fp8_close_to_all_gather():
    """With the fp8 dispatch both paths quantize the same rows; they differ
    only in the order of the sums (f32 combine at the source here)."""
    mesh = _make_mesh(attn_dp=2, model=4)
    expected = _run(mesh, False, None, fp8=True)
    actual = _run(mesh, True, None, fp8=True)
    err = np.abs(actual - expected).max() / np.abs(expected).max()
    assert err < 2e-2, err


@requires_8_devices
def test_ragged_all_to_all_fallback_semantics():
    """The off-TPU stand-in: device i's chunk for device j lands at
    output_offsets[j] in j's buffer; rows not written keep the buffer's
    values."""
    mesh = _make_mesh(attn_dp=4)
    axes = ("attn_dp", )
    send = np.array([[2, 2, 2, 2], [1, 0, 3, 4], [0, 0, 0, 8], [8, 0, 0, 0]],
                    np.int32)
    in_off = np.cumsum(send, 1) - send
    out_off = np.cumsum(send, 0) - send  # [sender, receiver]

    def f(x, s, i, o):
        me = jax.lax.axis_index(axes)
        all_s = jax.lax.all_gather(s[0], axes)
        out = jnp.full((16, 1), -1.0)
        return fused_moe_gmm._ragged_all_to_all(x[0], out, i[0], s[0], o[0],
                                                all_s[:, me], axes)[None]

    x = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8, 1)
    with jax.set_mesh(mesh):
        got = jax.jit(
            jax.shard_map(f,
                          mesh=mesh,
                          in_specs=(P(axes), ) * 4,
                          out_specs=P(axes),
                          check_vma=False))(x, jnp.asarray(send),
                                            jnp.asarray(in_off),
                                            jnp.asarray(out_off))
    got = np.asarray(got)[:, :, 0]
    for j in range(4):
        want = np.full(16, -1.0)
        for s in range(4):
            n = send[s, j]
            want[out_off[s, j]:out_off[s, j] +
                 n] = (np.asarray(x)[s, in_off[s, j]:in_off[s, j] + n, 0])
        np.testing.assert_array_equal(got[j], want)
