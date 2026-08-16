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


def _guard(**kw):
    from mlx_rl.memory import SwapGuard
    g = SwapGuard(**kw)
    g.baseline_gb = 1.0
    return g


def test_rate_detector_fires_only_after_sustained_paging():
    g = _guard(margin_gb=0, rate_mb_s=200, rate_samples=3)
    hot = 2_000_000_000  # 2 GB in 1 s = 2000 MB/s
    assert g.classify(hot, 1.0, 1.0) is None      # 1 sample: not yet
    assert g.classify(hot, 1.0, 1.0) is None      # 2 samples: not yet
    kind, rate = g.classify(hot, 1.0, 1.0)        # 3rd consecutive: abort
    assert kind == "rate" and rate == pytest.approx(2000.0)


def test_rate_detector_resets_on_a_quiet_sample():
    g = _guard(margin_gb=0, rate_mb_s=200, rate_samples=3)
    g.classify(2_000_000_000, 1.0, 1.0)
    g.classify(2_000_000_000, 1.0, 1.0)
    assert g.classify(0, 1.0, 1.0) is None        # quiet -> streak resets
    assert g.classify(2_000_000_000, 1.0, 1.0) is None
    assert g.classify(2_000_000_000, 1.0, 1.0) is None


def test_quiet_volume_growth_does_not_abort():
    """The false positive that killed a healthy run at step 69: swap grew
    while RAM was plentiful and no paging traffic was happening."""
    g = _guard(margin_gb=10, rate_mb_s=200, rate_samples=3)
    for used in (2.0, 4.0, 6.0, 8.0, 10.0):       # +9 GB over baseline
        assert g.classify(0, 3.0, used) is None


def test_level_detector_still_backstops():
    g = _guard(margin_gb=10, rate_mb_s=200, rate_samples=3)
    assert g.classify(0, 3.0, 12.5) == ("level", pytest.approx(11.5))


def test_rate_detector_disabled_by_zero():
    g = _guard(margin_gb=0, rate_mb_s=0, rate_samples=3)
    for _ in range(5):
        assert g.classify(10_000_000_000, 1.0, 1.0) is None
