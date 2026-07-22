"""Regression tests for the line-shape math in models.py.

These lock the FWHM/area conventions and the reconstruction path so that
refactors cannot silently change published numbers (the project's stated
priority). Run with:  pytest  (from the repo root, inside the .venv).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import evaluate, fwhm, area, propagate_error  # noqa: E402


def test_gauss_fwhm_and_area():
    p = [2.0, 500.0, 10.0]  # amp, cen, sigma
    assert abs(fwhm("gauss", p) - 2.3548 * 10.0) < 1e-9
    assert abs(area("gauss", None, p) - 2.0 * 10.0 * np.sqrt(2 * np.pi)) < 1e-9


def test_lorentz_fwhm_and_area():
    p = [2.0, 500.0, 10.0]  # amp, cen, HWHM
    assert abs(fwhm("lorentz", p) - 2 * 10.0) < 1e-9
    assert abs(area("lorentz", None, p) - 2.0 * np.pi * 10.0) < 1e-9


def test_bwf_value_at_centre_is_amp():
    x = np.array([500.0])
    assert abs(evaluate("bwf", x, [3.0, 500.0, 10.0, 2.0])[0] - 3.0) < 1e-12


def test_bwf_fwhm_formula():
    wid, q = 10.0, 2.0
    expected = 2 * abs(wid) * np.sqrt(1 + 1 / q ** 2)
    assert abs(fwhm("bwf", [1.0, 500.0, wid, q]) - expected) < 1e-9


def test_pvoigt_is_evaluated_as_true_voigt():
    # 'pvoigt' must fit and reconstruct identically to 'voigt' (they share
    # true_voigt in evaluate); a divergence would break replot_from_csv.
    x = np.linspace(400, 600, 401)
    p = [1.0, 500.0, 15.0]
    assert np.max(np.abs(evaluate("voigt", x, p) - evaluate("pvoigt", x, p))) == 0.0


def test_propagate_error_nan_on_nonfinite_cov():
    cov = np.array([[np.inf, 0.0], [0.0, 1.0]])
    assert np.isnan(propagate_error(lambda pp: pp[0], [1.0, 2.0], cov))


def test_voigt_area_matches_dense_integral():
    # Analytic Voigt area equals a dense integral over a wide window.
    p = [0.33, 520.0, 17.0]
    xg = np.linspace(520 - 4000, 520 + 4000, 400001)
    num = np.trapezoid(evaluate("voigt", xg, p), xg)
    assert abs(area("voigt", None, p) - num) / num < 2e-3


def test_bwf_area_is_floor_subtracted_and_window_independent():
    # BWF area subtracts the amp/q^2 continuum, so it matches a dense integral
    # of (profile - floor) and does NOT depend on the integration window.
    amp, cen, wid, q = 0.06, 1207.0, 100.0, 3.06
    p = [amp, cen, wid, q]
    xg = np.linspace(cen - 6000, cen + 6000, 600001)
    num = np.trapezoid(evaluate("bwf", xg, p) - amp / q ** 2, xg)
    assert abs(area("bwf", None, p) - num) / abs(num) < 2e-2


def test_bwf_area_negative_for_small_q_antiresonance():
    # |q| < 1 gives a genuine Fano antiresonance -> non-positive area.
    assert area("bwf", None, [0.03, 1082.0, 24.0, 0.8]) < 0
