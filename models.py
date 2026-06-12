"""Shared Raman peak-model definitions.

Single source of truth for the line-shape functions, their derived
quantities (FWHM, integrated area) and first-order uncertainty
propagation. Used by curve_fitting.py, replot_from_csv.py and
multi_spectra_comparison.py.

Width conventions (unchanged from the original pipeline):
  - gauss:   'wid' is the Gaussian sigma          -> FWHM = 2.3548 * wid
  - lorentz: 'wid' is the Lorentzian HWHM         -> FWHM = 2 * wid
  - voigt:   'wid' is a composite width parameter -> FWHM via fwhm_voigt()
  - bwf:     'wid' is a width parameter, 'q' the asymmetry
             -> FWHM = 2 * |wid| * sqrt(1 + 1/q^2)
"""

import numpy as np
from scipy.special import wofz

# np.trapz was renamed to np.trapezoid in numpy 2.0
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# === Peak models ===
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def lorentzian(x, amp, cen, wid):
    return amp * (wid**2 / ((x - cen)**2 + wid**2))

def true_voigt(x, amp, cen, wid):
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    gamma = wid / 2
    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return amp * profile / np.max(profile)

def pseudo_voigt(x, amp, cen, wid, eta=0.5):
    return eta * lorentzian(x, amp, cen, wid) + (1 - eta) * gaussian(x, amp, cen, wid)

def bwf(x, amp, cen, wid, q):
    """Breit-Wigner-Fano line shape (asymmetry parameter q)."""
    s = (x - cen) / wid
    return amp * ((1 + s / q) ** 2) / (1 + s**2)

# Number of free parameters per model (amp, cen, wid[, q])
N_PARAMS = {"gauss": 3, "lorentz": 3, "voigt": 3, "pvoigt": 3, "bwf": 4}


def evaluate(model, x, params):
    """Evaluate a named line shape with its parameter vector (amp, cen, wid[, q])."""
    if model == "gauss":
        return gaussian(x, *params)
    if model == "lorentz":
        return lorentzian(x, *params)
    if model in ("voigt", "pvoigt"):
        return true_voigt(x, *params)
    if model == "bwf":
        return bwf(x, *params)
    raise ValueError(f"Unknown model type: {model}")


# === Derived quantities ===
def fwhm_voigt(wid):    # 'wid' is a Voigt composite width parameter, not equal to FWHM
    gamma = wid / 2
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    fwhm_g = 2.3548 * sigma
    fwhm_l = 2 * gamma
    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)


def fwhm(model, params):
    """FWHM of a named line shape from its parameter vector."""
    if model == "gauss":
        return 2.3548 * np.abs(params[2])
    if model == "lorentz":
        return 2 * params[2]
    if model in ("voigt", "pvoigt"):
        return fwhm_voigt(params[2])
    if model == "bwf":
        wid, q = params[2], params[3]
        return 2 * np.abs(wid) * np.sqrt(1 + 1 / q**2)
    raise ValueError(f"Unknown model type: {model}")


def area(model, x, params):
    """Integrated area of a named line shape (analytic where possible,
    trapezoidal over the supplied x grid otherwise)."""
    if model == "gauss":
        amp, _, wid = params
        return amp * wid * np.sqrt(2 * np.pi)
    if model == "lorentz":
        amp, _, wid = params
        return amp * np.pi * wid
    if model in ("voigt", "pvoigt", "bwf"):
        return _trapezoid(evaluate(model, x, params), x)
    raise ValueError(f"Unknown model type: {model}")


# === Uncertainty propagation ===
def propagate_error(func, params, cov, rel_step=1e-6):
    """1-sigma uncertainty of scalar func(params) by first-order propagation:
    sigma_f = sqrt(J . cov . J^T) with a central-difference Jacobian.

    Returns np.nan if the covariance block is not finite (degenerate fit).
    """
    params = np.asarray(params, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)):
        return np.nan

    jac = np.zeros_like(params)
    for i in range(params.size):
        step = rel_step * max(abs(params[i]), 1.0)
        upper = params.copy(); upper[i] += step
        lower = params.copy(); lower[i] -= step
        jac[i] = (func(upper) - func(lower)) / (2 * step)

    variance = jac @ cov @ jac
    return np.sqrt(variance) if variance >= 0 else np.nan
