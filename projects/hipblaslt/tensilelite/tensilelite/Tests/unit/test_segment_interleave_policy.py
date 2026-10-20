# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise paired-placement policy through real gfx1250 Solution derivation."""

import pytest

from Tensile.SolutionStructs.Solution import Solution
from Tensile.Tests.unit.test_halfplr_streamk_rejects import (
    _gp_gfx1250, assembler, gfx1250_iim, _make_params,  # noqa: F401
)

pytestmark = pytest.mark.unit


@pytest.fixture
def derive(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    def run(mode, **overrides):
        params = dict(
            mi=[16, 16, 128, 1, 1, 8, 8, 2, 2],
            StreamK=0, GlobalSplitU=1, SourceSwap=True,
            VectorWidthA=8, VectorWidthB=4, TDMIterateMode=-1,
            LDSSegmentInterleave=mode, LDSSegmentBRotate=0,
        )
        params.update(overrides)
        sol = Solution(_make_params(gfx1250_iim, **params), False, True,
                       False, assembler, gfx1250_iim)
        return sol, capsys.readouterr().out
    return run


def _lds_configuration(sol):
    # Include every LDS-derived field, not just the resolved interleave flag.
    return {key: value for key, value in sol._state.items()
            if key.startswith(("Lds", "lds", "LDSSeg"))
            or key in ("NumLdsBlk", "StoreSwapAddr", "1LDSBuffer",
                       "ExpandPointerSwap", "TDMPlusLdsBuf", "DtlPlusLdsBuf")}


@pytest.mark.parametrize("mode", [-1, 1])
@pytest.mark.parametrize("rotate", [0, 1])
@pytest.mark.parametrize("pgr", [1, 2])
def test_supported_interleave_requires_paired(derive, mode, rotate, pgr):
    sol, out = derive(mode, LDSSegmentBRotate=rotate, PrefetchGlobalRead=pgr)
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert sol["LDSSegInterleaveOffsets"]["pairedBuffers"] is True


def test_unavailable_paired_rejects_forced_but_auto_matches_off(derive):
    overrides = dict(PrefetchGlobalRead=1, HalfPLR=0, ExpandPointerSwap=True)
    off, out = derive(0, **overrides)
    assert off["Valid"], out
    auto, out = derive(-1, **overrides)
    assert auto["Valid"], out
    assert auto["LDSSegmentInterleave"] == 0
    assert auto["LDSSegInterleaveOffsets"] == {}
    assert _lds_configuration(auto) == _lds_configuration(off)
    forced, out = derive(1, **overrides)
    assert not forced["Valid"]
    assert "LDSSegmentInterleave=1 requires paired LDS placement" in out
    assert "ExpandPointerSwap is not supported" in out


@pytest.mark.parametrize("pgr", [1, 2])
@pytest.mark.parametrize("mx_block", [16, 32])
@pytest.mark.parametrize("mode", [-1, 1])
def test_fp4_uses_paired_placement(derive, pgr, mx_block, mode):
    sol, out = derive(mode, PrefetchGlobalRead=pgr, DepthU=512, ProblemType={
        "DataType": "F4", "DestDataType": "s", "MXBlockA": mx_block, "MXBlockB": mx_block,
    })
    assert sol["Valid"], out
    assert sol["LDSSegmentInterleave"] == 1
    assert sol["LDSSegInterleaveOffsets"]["pairedBuffers"] is True


@pytest.mark.parametrize("pgr", [1, 2])
@pytest.mark.parametrize("a,b", [("F4", "F8"), ("F8", "F4")])
def test_mixed_fp4_uses_paired_placement(derive, pgr, a, b):
    sol, out = derive(1, PrefetchGlobalRead=pgr, ProblemType={
        "DataType": "F8", "DestDataType": "s", "MXBlockA": 32, "MXBlockB": 32,
        "DataTypeA": a, "DataTypeB": b, "MacDataTypeA": a, "MacDataTypeB": b,
    })
    assert sol["Valid"], out
    assert sol["LDSSegInterleaveOffsets"]["pairedBuffers"] is True


@pytest.mark.parametrize("pgr", [1, 2])
def test_auto_aligned_layout_matches_off(derive, pgr):
    overrides = dict(mi=[16, 16, 128, 1, 1, 4, 4, 2, 2],
                     VectorWidthA=4, VectorWidthB=1, HalfPLR=0, PrefetchGlobalRead=pgr)
    off, out = derive(0, **overrides)
    assert off["Valid"], out
    auto, out = derive(-1, **overrides)
    assert auto["Valid"], out
    assert _lds_configuration(auto) == _lds_configuration(off)
    forced, out = derive(1, **overrides)
    assert forced["Valid"], out
    assert forced["LDSSegInterleaveOffsets"]["pairedBuffers"] is True


def test_paired_budget_failure_obeys_requested_mode(derive, monkeypatch):
    import importlib
    solution_module = importlib.import_module("Tensile.SolutionStructs.Solution")
    off, out = derive(0)
    assert off["Valid"], out
    # Exercise the caller's failure policy independently of the chosen geometry.
    # The offset generator returns None when its placement exceeds MaxLDS.
    monkeypatch.setattr(solution_module, "segPairedOffsets", lambda state: None)
    auto, out = derive(-1)
    assert auto["Valid"], out
    assert _lds_configuration(auto) == _lds_configuration(off)
    forced, out = derive(1)
    assert not forced["Valid"]
    assert "requires paired LDS placement: layout exceeds MaxLDS" in out
    rotated, out = derive(-1, LDSSegmentBRotate=1)
    assert not rotated["Valid"]
    assert "LDSSegmentBRotate requires paired LDS placement" in out


def test_auto_cannot_commit_a_layout_that_grows_lds(derive, monkeypatch):
    import importlib
    solution_module = importlib.import_module("Tensile.SolutionStructs.Solution")
    generate_offsets = solution_module.segPairedOffsets
    off, out = derive(0)
    assert off["Valid"], out

    def with_extra_reservation(state):
        offsets = generate_offsets(state)
        # Model additional padding that fits MaxLDS but exceeds the ordinary
        # reservation. The automatic policy must discard the entire placement.
        offsets["pairedUsedBytes"] += 16
        return offsets

    monkeypatch.setattr(solution_module, "segPairedOffsets", with_extra_reservation)
    auto, out = derive(-1)
    assert auto["Valid"], out
    assert _lds_configuration(auto) == _lds_configuration(off)
    forced, out = derive(1)
    assert forced["Valid"], out
    assert forced["LDSSegInterleaveOffsets"]["pairedBuffers"] is True
    assert forced["LdsNumBytes"] > off["LdsNumBytes"]


@pytest.mark.parametrize("mode", [-1, 1])
def test_brotate_cannot_fall_back_to_unpaired(derive, mode):
    sol, out = derive(mode, LDSSegmentBRotate=1,
                      PrefetchGlobalRead=1, HalfPLR=0, ExpandPointerSwap=True)
    assert not sol["Valid"]
    assert "LDSSegmentBRotate: needs ExpandPointerSwap=0" in out


def test_tdmsplit_remains_disabled(derive):
    sol, out = derive(1, TDMSplit=True)
    assert not sol["Valid"]
    assert "TDMSplit is currently disabled" in out
