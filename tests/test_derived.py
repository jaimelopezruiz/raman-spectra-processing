"""Tests for derived_quantities: covariance-aware band-area and χ propagation.

Uses two well-separated Lorentzians with a hand-constructed covariance so the
expected 1σ can be written in closed form (Lorentz area = π·amp·wid is linear
in amp and wid, so the finite-difference Jacobian must match the analytic one).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derived_quantities import (  # noqa: E402
    band_area, chi_ratio, by_center_range, stress_from_shift, chi_integrated,
    line_through_two_points, lo_band_position,
)

PI = np.pi


def _region(pcov):
    # A @ 800 (amp 2, wid 3) = TO/denominator; B @ 1400 (amp 1, wid 5) = D/numerator
    popt = np.array([2.0, 800.0, 3.0, 1.0, 1400.0, 5.0])
    peaks = [
        {"peak": 1, "model": "lorentz", "assignment": "TO", "mu": 800.0, "i0": 0, "n": 3},
        {"peak": 2, "model": "lorentz", "assignment": "D", "mu": 1400.0, "i0": 3, "n": 3},
    ]
    return [{"region": "x", "start": 700, "end": 1500,
             "popt": popt, "pcov": np.asarray(pcov, float), "peaks": peaks}]

DIAG = np.diag([0.1**2, 1.0, 0.2**2, 0.05**2, 1.0, 0.3**2])


def test_band_area_and_1sigma_match_analytic():
    a, u = band_area(_region(DIAG), by_center_range(780, 820))  # peak A
    assert abs(a - PI * 2 * 3) < 1e-9
    grad = np.array([PI * 3, 0.0, PI * 2, 0.0, 0.0, 0.0])       # d(π·amp·wid)
    assert abs(u - np.sqrt(grad @ DIAG @ grad)) < 1e-6


def test_chi_is_quadrature_when_uncorrelated():
    r = chi_ratio(_region(DIAG), by_center_range(1380, 1420), by_center_range(780, 820))
    assert abs(r["chi"] - (PI * 5) / (PI * 6)) < 1e-9
    assert abs(r["corr_num_den"]) < 1e-9
    u_quad = r["chi"] * np.sqrt((r["u_A_num"] / r["A_num"])**2 + (r["u_A_den"] / r["A_den"])**2)
    assert abs(r["u_chi"] - u_quad) < 1e-9


def test_chi_uses_cross_covariance_when_present():
    pcov = DIAG.astype(float).copy()
    pcov[0, 3] = pcov[3, 0] = 0.5 * 0.1 * 0.05            # +0.5 corr between the two amps
    r = chi_ratio(_region(pcov), by_center_range(1380, 1420), by_center_range(780, 820))
    assert r["corr_num_den"] > 0
    u_quad = r["chi"] * np.sqrt((r["u_A_num"] / r["A_num"])**2 + (r["u_A_den"] / r["A_den"])**2)
    assert r["u_chi"] < u_quad          # positive num/den correlation shrinks ratio variance


def test_stress_from_shift_propagation():
    r = stress_from_shift(785.0, 0.3, 788.72, 0.1, k=370.4, strain_coeff=0.0739)
    dw, u_dw = 785.0 - 788.72, np.hypot(0.3, 0.1)
    assert abs(r["delta_omega"] - dw) < 1e-12
    assert abs(r["u_delta_omega"] - u_dw) < 1e-12
    assert abs(r["sigma_MPa"] - (-370.4 * dw)) < 1e-9
    assert abs(r["u_sigma_MPa"] - 370.4 * u_dw) < 1e-9
    assert abs(r["strain_pct"] - 0.0739 * dw) < 1e-12
    assert abs(r["u_strain_pct"] - 0.0739 * u_dw) < 1e-12


def test_chi_integrated_windows():
    x = np.linspace(100, 2000, 19001)
    y = np.ones_like(x)
    r = chi_integrated(x, y, d_window=(1340, 1470), to_window=(700, 850))
    assert abs(r["A_D"] - 130) < 0.5      # ∫1 over a 130 cm⁻¹ window
    assert abs(r["A_TO"] - 150) < 0.5     # ∫1 over a 150 cm⁻¹ window
    assert abs(r["chi"] - 130 / 150) < 1e-3
    # doubling the height in the D window doubles χ
    y2 = y.copy(); y2[(x >= 1340) & (x <= 1470)] = 2.0
    r2 = chi_integrated(x, y2, d_window=(1340, 1470), to_window=(700, 850))
    assert abs(r2["chi"] - 2 * 130 / 150) < 5e-3


def test_line_through_two_points():
    r = line_through_two_points(300, 910.0, 0.5, 750, 900.0, 0.3)
    assert abs(r["slope"] - (-10.0 / 450)) < 1e-12
    assert abs(r["u_slope"] - np.hypot(0.5, 0.3) / 450) < 1e-12
    assert abs(r["intercept"] - (750 * 910 - 300 * 900) / 450) < 1e-9
    assert abs(r["u_intercept"] - np.hypot(750 * 0.5, 300 * 0.3) / 450) < 1e-9
    assert r["dof"] == 0


def test_lo_band_position_subsample():
    # A Gaussian centred midway between grid points; parabolic refinement should
    # recover the true centre to sub-sample precision within the LO window.
    x = np.arange(880, 980, 2.0)
    cen = 925.0                      # exactly between grid points 924 and 926
    y = np.exp(-((x - cen) ** 2) / (2 * 6.0 ** 2))
    assert abs(lo_band_position(x, y, window=(910, 945)) - cen) < 0.3
    # a bare grid argmax is off by ~dx/2 = 1.0 here
    assert abs(x[np.argmax(y)] - cen) >= 0.6
