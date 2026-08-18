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


HOT = 2_000_000_000  # 2 GB moved in 1 s = 2000 MB/s


def _paced(g, step_s=6.0, t0=1000.0):
    """Give the guard a healthy step pace and return the clock after it."""
    for i in range(g.baseline_steps + 1):
        g.note_step(step_s, now=t0 + i * step_s)
    assert g._step_baseline_s == pytest.approx(step_s)
    return t0 + g.baseline_steps * step_s


def test_paging_does_not_abort_a_run_whose_steps_are_healthy():
    """The false positive that killed a correct 200-step run: a nearly-full
    swap file throws ~270 MB/s transients the run never feels."""
    g = _guard(margin_gb=0, rate_mb_s=200, rate_samples=3)
    t = _paced(g)
    for _ in range(10):
        assert g.classify(HOT, 1.0, 1.0, now=t + 1.0) is None


def test_paging_aborts_once_steps_have_actually_slowed():
    g = _guard(margin_gb=0, rate_mb_s=200, rate_samples=3)
    t = _paced(g, step_s=6.0)
    g.note_step(30.0, now=t + 30.0)              # 5x the baseline
    assert g.classify(HOT, 1.0, 1.0, now=t + 31.0) is None
    assert g.classify(HOT, 1.0, 1.0, now=t + 32.0) is None
    kind, rate = g.classify(HOT, 1.0, 1.0, now=t + 33.0)
    assert kind == "rate" and rate == pytest.approx(2000.0)


def test_paging_aborts_when_the_current_step_is_overdue():
    """Thrashing inside a step never reports a duration — the run just stops
    finishing steps. Overdue past slow_factor x baseline counts as hurting."""
    g = _guard(margin_gb=0, rate_mb_s=200, rate_samples=3)
    t = _paced(g, step_s=6.0)
    late = t + 6.0 * 3 + 1                        # past 3x baseline, no note_step
    for _ in range(2):
        assert g.classify(HOT, 1.0, 1.0, now=late) is None
    assert g.classify(HOT, 1.0, 1.0, now=late)[0] == "rate"


def test_paging_before_any_step_still_aborts():
    """Load-time / first-backward thrashing is the rate detector's one true
    positive. With no pace established there is nothing to gate on, so the
    old behaviour must survive."""
    g = _guard(margin_gb=0, rate_mb_s=200, rate_samples=3)
    assert g.classify(HOT, 1.0, 1.0) is None
    assert g.classify(HOT, 1.0, 1.0) is None
    assert g.classify(HOT, 1.0, 1.0)[0] == "rate"


def test_warmup_step_does_not_inflate_the_baseline():
    """Step 0 carries compilation; if it set the pace, a 3x gate would be
    meaningless for the rest of the run."""
    g = _guard(margin_gb=0, rate_mb_s=200, rate_samples=3)
    g.note_step(120.0, now=1000.0)                # compile-heavy first step
    for i in range(1, g.baseline_steps + 1):
        g.note_step(6.0, now=1000.0 + i * 6.0)
    assert g._step_baseline_s == pytest.approx(6.0)


def test_withheld_kill_resets_the_streak_so_it_reports_again():
    g = _guard(margin_gb=0, rate_mb_s=200, rate_samples=3)
    t = _paced(g)
    for _ in range(3):
        g.classify(HOT, 1.0, 1.0, now=t + 1.0)
    assert g._hot == 0


def test_rate_detector_disabled_by_zero():
    g = _guard(margin_gb=0, rate_mb_s=0, rate_samples=3)
    for _ in range(5):
        assert g.classify(10_000_000_000, 1.0, 1.0) is None
