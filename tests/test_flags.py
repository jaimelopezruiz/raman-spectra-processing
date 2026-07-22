"""Unit tests for the fit-quality flag logic in curve_fitting._bound_flags.

Builds parameter blocks exactly as fit_peaks_regionwise does
(lb = [0, cen0 - tol, 1], ub = [2*amp0, cen0 + tol, wmax]) and checks that
each artefact is flagged and a clean peak is not.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curve_fitting import _bound_flags  # noqa: E402

AMP0, CEN0, WID0, TOL, WMAX = 0.5, 500.0, 10.0, 100.0, 100.0
SEED = [AMP0, CEN0, WID0]
LB = [0.0, CEN0 - TOL, 1.0]
UB = [2 * AMP0, CEN0 + TOL, WMAX]


def test_clean_peak_has_no_flags():
    assert _bound_flags("voigt", [0.5, 505.0, 10.0], LB, UB, SEED) == []


def test_amplitude_collapse_flagged():
    flags = _bound_flags("voigt", [1e-9, 500.0, 10.0], LB, UB, SEED)
    assert any("amp~0" in f for f in flags)


def test_width_upper_bound_flagged():
    flags = _bound_flags("voigt", [0.5, 500.0, WMAX], LB, UB, SEED)
    assert any("width at upper bound" in f for f in flags)


def test_centre_at_tolerance_bound_flagged():
    flags = _bound_flags("voigt", [0.5, CEN0 + TOL, 10.0], LB, UB, SEED)
    assert any("centre at +/- tolerance bound" in f for f in flags)


def test_centre_drift_flagged_but_not_at_bound():
    # 60 cm-1 drift: past DRIFT_FRACTION*tol (50) but not at the ±100 bound.
    flags = _bound_flags("voigt", [0.5, 560.0, 10.0], LB, UB, SEED)
    assert any("drifted" in f for f in flags)
    assert not any("tolerance bound" in f for f in flags)


def test_small_drift_not_flagged():
    assert _bound_flags("voigt", [0.5, 540.0, 10.0], LB, UB, SEED) == []
