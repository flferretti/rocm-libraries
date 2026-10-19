# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import ctypes
import importlib.util
import os
import struct

import numpy as np
import pytest
from rocke.core.backend import universal_gemm_spec_to_dict
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
)
from rocke.runtime.hip_module import get_device_arch


def _require_gfx942():
    arch = get_device_arch()
    if not arch or arch.split(":")[0] != "gfx942":
        if os.getenv("ROCKE_REQUIRE_GFX942") == "1":
            pytest.fail(f"Required gfx942 device absent: {arch}")
        pytest.skip("requires gfx942")


def _engines():
    # The installed wheel has no pybind extension; demand native parity
    # whenever the binding is supplied.
    if importlib.util.find_spec("rocke_engine") is None:
        return ("python",)
    return ("python", "cpp")


def _spec(tile, pipeline, epilogue, pad=False, warp_tile=(16, 16, 4), acc="fp64"):
    tile_m, tile_n, tile_k, warp_m, warp_n = tile
    return UniversalGemmSpec(
        name="f64_gemm",
        tile=TileSpec(tile_m, tile_n, tile_k, warp_m, warp_n, 1, *warp_tile),
        trait=TraitSpec(
            pipeline=pipeline, epilogue=epilogue, pad_m=pad, pad_n=pad, pad_k=pad
        ),
        data=DataSpec("fp64", "fp64", "fp64", acc),
    )


def _llvm(spec, engine):
    if engine == "python":
        kernel = build_universal_gemm(spec, arch="gfx942")
        return _lower_kernel_to_llvm_python(kernel, arch="gfx942")
    import rocke_engine as native

    return native.gemm_lower_llvm(universal_gemm_spec_to_dict(spec), arch="gfx942")


def _launch(ll, spec, a, bt):
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    m, k = a.shape
    n = bt.shape[0]
    blob, _ = build_hsaco_from_llvm_ir(ll, isa="amdgcn-amd-amdhsa--gfx942")
    runtime = Runtime()
    module = runtime.load_module(blob)
    arrays = [a.copy(), bt.copy(), np.full((m, n), np.nan)]
    ptrs = []
    try:
        for array in arrays:
            ptr = runtime.alloc(array.nbytes)
            ptrs.append(ptr)
            runtime.memcpy_h2d(
                ptr, (ctypes.c_ubyte * array.nbytes).from_buffer(array), array.nbytes
            )
        t = spec.tile
        grid = (-(-n // t.tile_n), -(-m // t.tile_m), 1)
        runtime.launch_blocking(
            module.get_function(spec.kernel_name()),
            grid,
            (spec.block_size, 1, 1),
            struct.pack("3Q3i", *ptrs, m, n, k),
        )
        c = arrays[2]
        runtime.memcpy_d2h(
            (ctypes.c_ubyte * c.nbytes).from_buffer(c), ptrs[2], c.nbytes
        )
        return c
    finally:
        for ptr in ptrs:
            runtime.free(ptr)
        module.unload()


@pytest.mark.parametrize(
    "tile, pipeline, epilogue, pad",
    [
        ((64, 64, 16, 2, 2), "compv4", "default", True),
        ((64, 64, 16, 2, 2), "compv3", "cshuffle", False),
        ((32, 32, 8, 1, 1), "mem", "default", False),
        ((16, 16, 4, 1, 1), "mem", "cshuffle", True),
    ],
)
def test_gfx942_f64_gemm_numeric(tile, pipeline, epilogue, pad):
    _require_gfx942()
    spec = _spec(tile, pipeline, epilogue, pad)
    assert is_valid_spec(spec, arch="gfx942") == (True, "ok")
    tile_m, tile_n, tile_k = tile[:3]
    # Padded configs use extents that are not tile multiples.
    trim = 1 if pad else 0
    m, n, k = 2 * tile_m - trim, 3 * tile_n - trim, 4 * tile_k - trim
    rng = np.random.default_rng(tile_m * 100 + tile_k)
    for engine in _engines():
        ll = _llvm(spec, engine)
        a = rng.integers(-8, 9, (m, k)).astype(np.float64)
        bt = rng.integers(-8, 9, (n, k)).astype(np.float64)
        np.testing.assert_array_equal(_launch(ll, spec, a, bt), a @ bt.T)
        a = rng.standard_normal((m, k))
        bt = rng.standard_normal((n, k))
        c = _launch(ll, spec, a, bt)
        error = np.abs(c - a @ bt.T) / (np.abs(a) @ np.abs(bt).T)
        assert np.max(error) < k * np.finfo(np.float64).eps


@pytest.mark.parametrize(
    "spec, reason",
    [
        (
            _spec((64, 64, 16, 2, 2), "mem", "default", warp_tile=(4, 4, 4)),
            "fp64 GEMM supports only warp_tile (16, 16, 4)",
        ),
        (
            _spec((64, 64, 16, 2, 2), "mem", "default", acc="fp32"),
            "only supports fp64 accumulation for fp64",
        ),
    ],
)
def test_f64_gemm_rejects_unsupported(spec, reason):
    ok, why = is_valid_spec(spec, arch="gfx942")
    assert not ok and reason in why
    if "cpp" in _engines():
        import rocke_engine as native

        sd = universal_gemm_spec_to_dict(spec)
        assert native.gemm_is_valid(sd, arch="gfx942") == (ok, why)
