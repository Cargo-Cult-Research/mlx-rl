import pytest

from mlx_rl.memory import MemoryGuardError, assert_fits, estimate_run_gb


def test_fits_passes():
    assert_fits(10.0, available=50.0)


def test_refuses_when_too_big():
    with pytest.raises(MemoryGuardError, match="memory lease"):
        assert_fits(60.0, available=50.0)


def test_safety_margin_applies():
    # 46 GB required vs 50 GB available: over the 0.9 margin -> refuse
    with pytest.raises(MemoryGuardError):
        assert_fits(46.0, available=50.0)


def test_estimate_includes_headroom():
    assert estimate_run_gb(10.0, headroom_gb=4.0) == pytest.approx(33.0)
