"""Derived quantities computed from a fit, with covariance-aware 1σ uncertainty.

These are the values quoted in the report that are *combinations* of fitted
peaks (e.g. the chemical-disorder ratio χ, band-area ratios, peak-shift-derived
stress). Keeping them here — rather than in a spreadsheet — means every number
that feeds the paper is reproducible from the deposited fit.

Uncertainty is propagated from the per-region covariance matrix exposed by
curve_fitting.fit_peaks_regionwise (`region_fits`). Peaks fitted together in one
region may be correlated (overlapping line shapes → non-zero off-diagonal
covariance); peaks in different regions come from independent fits and are
treated as uncorrelated. Validated against a 2×10⁵-sample Monte Carlo
(analytic vs MC u(χ) agree to ~0.1%); see tests/test_derived.py.

Band membership is caller-supplied (a predicate over each peak) because *which*
peaks constitute, say, the "Si-C TO band" is a physics/definition choice that
belongs to the analyst, not a guess in the code. Convenience selectors are
provided below.
"""
import numpy as np

from models import area


# === Peak selectors (build the predicates you pass to chi_ratio/band_area) ===
def by_assignment_contains(*substrings):
    """Select peaks whose assignment contains any of the given substrings
    (case-insensitive)."""
    subs = [s.lower() for s in substrings]
    return lambda pk: any(s in (pk.get("assignment") or "").lower() for s in subs)


def by_center_range(lo, hi):
    """Select peaks whose fitted centre falls in [lo, hi] cm⁻¹."""
    return lambda pk: lo <= pk["mu"] <= hi


# === Internals ===
def _peak_area(pk, theta):
    return area(pk["model"], None, theta[pk["i0"]:pk["i0"] + pk["n"]])

def _group_area(peaks, theta):
    return float(sum(_peak_area(pk, theta) for pk in peaks))

def _group_area_jac(peaks, theta, rel=1e-6):
    """Gradient of the summed area of `peaks` w.r.t. the full region parameter
    vector `theta`, by central differences (area is smooth in the params)."""
    g = np.zeros_like(theta, dtype=float)
    for i in range(theta.size):
        step = rel * max(abs(theta[i]), 1.0)
        up = theta.copy(); up[i] += step
        lo = theta.copy(); lo[i] -= step
        g[i] = (_group_area(peaks, up) - _group_area(peaks, lo)) / (2 * step)
    return g


# === Public API ===
def band_area(region_fits, select):
    """Total area of all peaks matching `select`, summed across regions.

    Returns (area, u_area). Within a region the full covariance is used, so
    overlapping (correlated) sub-peaks are handled correctly; different regions
    are independent and their variances add.
    """
    total, var = 0.0, 0.0
    for rf in region_fits:
        sel = [pk for pk in rf["peaks"] if select(pk)]
        if not sel:
            continue
        theta, pcov = rf["popt"], rf["pcov"]
        total += _group_area(sel, theta)
        if np.all(np.isfinite(pcov)):
            j = _group_area_jac(sel, theta)
            var += float(j @ pcov @ j)
        else:
            var = np.nan
    return total, np.sqrt(var)


def chi_ratio(region_fits, numerator_select, denominator_select):
    """Area ratio χ = A_numerator / A_denominator with covariance-aware 1σ.

    If numerator and denominator peaks share a region, their (usually small)
    covariance is included; otherwise they are independent. Returns a dict with
    χ, u(χ), the two band areas and their 1σ, and the numerator/denominator
    area correlation actually used (so the independence assumption is auditable
    rather than assumed).
    """
    A_num = A_den = 0.0
    var_num = var_den = cov = 0.0
    for rf in region_fits:
        theta, pcov = rf["popt"], rf["pcov"]
        num = [pk for pk in rf["peaks"] if numerator_select(pk)]
        den = [pk for pk in rf["peaks"] if denominator_select(pk)]
        if not num and not den:
            continue
        A_num += _group_area(num, theta) if num else 0.0
        A_den += _group_area(den, theta) if den else 0.0
        if not np.all(np.isfinite(pcov)):
            var_num = var_den = cov = np.nan
            continue
        jn = _group_area_jac(num, theta) if num else np.zeros_like(theta, dtype=float)
        jd = _group_area_jac(den, theta) if den else np.zeros_like(theta, dtype=float)
        var_num += float(jn @ pcov @ jn)
        var_den += float(jd @ pcov @ jd)
        cov += float(jn @ pcov @ jd)

    result = {
        "chi": np.nan, "u_chi": np.nan,
        "A_num": A_num, "u_A_num": np.sqrt(var_num),
        "A_den": A_den, "u_A_den": np.sqrt(var_den),
        "corr_num_den": (cov / np.sqrt(var_num * var_den)
                         if var_num > 0 and var_den > 0 else np.nan),
    }
    if A_den != 0:
        chi = A_num / A_den
        d_num, d_den = 1.0 / A_den, -A_num / A_den ** 2
        result["chi"] = chi
        result["u_chi"] = np.sqrt(
            d_num ** 2 * var_num + d_den ** 2 * var_den + 2 * d_num * d_den * cov
        )
    return result


# === Report's confirmed χ definition (2026-07-21) ===
# Numerator: the C-C D band (single ~1400 cm⁻¹ peak).
# Denominator: the WHOLE Si-C TO band 767–797 cm⁻¹, *including* the broad
# disordered Si-C band overlapping it — so the denominator degrades gracefully
# rather than collapsing to ~0 once the sharp crystalline TO peaks vanish at
# high damage (see report §4.3; rationale to be confirmed by JLR). Membership
# is by fitted centre, matching the paper's wavenumber-window definition; the
# ranges are module constants so they are auditable and easy to tune.
CHI_D_BAND = (1350.0, 1450.0)
CHI_TO_BAND = (755.0, 805.0)


def _selected(region_fits, select):
    return [
        {"peak": pk["peak"], "mu": round(float(pk["mu"]), 1), "model": pk["model"],
         "assignment": pk.get("assignment", ""), "flags": pk.get("flags", "")}
        for rf in region_fits for pk in rf["peaks"] if select(pk)
    ]


def chi_default(region_fits, d_band=CHI_D_BAND, to_band=CHI_TO_BAND):
    """χ using the report's confirmed band definition (centre-range membership).

    Returns the chi_ratio dict plus, for auditability, the band windows used and
    the list of peaks each band actually included (with any fit-quality flags).
    """
    num = by_center_range(*d_band)
    den = by_center_range(*to_band)
    res = chi_ratio(region_fits, num, den)
    res["numerator_band"] = d_band
    res["denominator_band"] = to_band
    res["numerator_peaks"] = _selected(region_fits, num)
    res["denominator_peaks"] = _selected(region_fits, den)
    return res


# === Integrated-band χ (the report's annealing χ; §2.3.3) ===
# χ = ∫(D window) / ∫(TO window) of the baseline-corrected, normalised spectrum.
# Chosen 2026-07-21 over summed fitted-peak areas because a fixed peak set cannot
# track the annealing series (amorphous→recovered): a mis-decomposition collapses
# the TO area and χ blows up. Direct integration is decomposition-free, robust,
# and reproduces the manual-Excel χ-vs-T trend. Windows: the Si-C TO band
# (including the broad disordered SiC that overlaps it) and the C-C D band —
# matching how the bands were summed in the Excel workbook.
CHI_D_WINDOW = (1340.0, 1470.0)
CHI_TO_WINDOW = (700.0, 850.0)


def _window_integral(x, y, w):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (x >= w[0]) & (x <= w[1])
    if m.sum() < 2:
        return float("nan")
    return float(np.trapezoid(y[m], x[m]))


def chi_integrated(x, y, d_window=CHI_D_WINDOW, to_window=CHI_TO_WINDOW):
    """χ from direct window integration of a baseline-corrected spectrum.

    Decomposition-free and robust across the annealing series. Returns χ, the
    two band integrals and the windows used. This is a point value; the dominant
    uncertainty is the I-ModPoly baseline-order systematic, characterised by
    re-integrating at several orders (see annealing_chi.py) rather than from a
    fit covariance (there is no fit here).
    """
    a_d = _window_integral(x, y, d_window)
    a_to = _window_integral(x, y, to_window)
    return {
        "chi": a_d / a_to if a_to else float("nan"),
        "A_D": a_d, "A_TO": a_to,
        "d_window": tuple(d_window), "to_window": tuple(to_window),
    }


def line_through_two_points(x1, y1, uy1, x2, y2, uy2):
    """Slope/intercept of the line through two (x, y±u_y) points (x exact).

    Report Fig 9b/18 fits the LO position at only two irradiation temperatures
    (300 & 750 °C) per dose, so the "linear fit" is exact: DoF = 0, there is no
    residual scatter, and the slope/intercept uncertainty comes *entirely* from
    the two peak-position errors — NOT from a regression. Quote it that way.
    """
    dx = float(x2 - x1)
    return {
        "slope": (y2 - y1) / dx,
        "u_slope": float(np.hypot(uy1, uy2) / abs(dx)),
        "intercept": (x2 * y1 - x1 * y2) / dx,
        "u_intercept": float(np.hypot(x2 * uy1, x1 * uy2) / abs(dx)),
        "dof": 0,
    }


# === Folded-LO band position (report Fig 9b/18) ===
# The graph tracks the LO *band* envelope position, not a single fitted
# sub-peak. A parabolic-refined argmax over [910, 945] reproduces the published
# LO positions to ~2 cm-1 RMS (tuned 2026-07-21 against the graph values
# 917/935/929/939), and gives the 2.5 dpa line an intercept ~902 cm-1, matching
# the report's ~903.3.
LO_WINDOW = (910.0, 945.0)


def lo_band_position(x, y, window=LO_WINDOW):
    """Folded-LO band position = the envelope maximum (parabolic-refined argmax)
    over `window`. Decomposition-free and normalisation-invariant, so it is
    robust where a multi-peak LO fit is not (the fit splits the band into
    disordered/crystalline components whose identities switch across samples).
    Pair with line_through_two_points for the LO-vs-irradiation-T slope.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (x >= window[0]) & (x <= window[1])
    xi, yi = x[m], y[m]
    if xi.size == 0:
        return float("nan")
    i = int(np.argmax(yi))
    if 0 < i < len(yi) - 1:  # parabolic (sub-sample) refinement of the peak
        y0, y1, y2 = yi[i - 1], yi[i], yi[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            return float(xi[i] + 0.5 * (y0 - y2) / denom * (xi[i + 1] - xi[i]))
    return float(xi[i])


def peak_in_window(region_fits, window):
    """(centre, u_centre, area) of the largest-area fitted peak whose centre is
    in `window`, across all regions. u_centre is the fit 1σ on the centre
    (√ of the covariance diagonal). Returns None if no peak lies in the window.
    Peak positions are independent of intensity normalisation, so this is a
    normalisation-robust way to track a band's location (e.g. the LO mode)."""
    best = None
    for rf in region_fits:
        popt, pcov = rf["popt"], rf["pcov"]
        for pk in rf["peaks"]:
            if window[0] <= pk["mu"] <= window[1]:
                a = area(pk["model"], None, popt[pk["i0"]:pk["i0"] + pk["n"]])
                if best is None or a > best[2]:
                    ci = pk["i0"] + 1  # centre is the 2nd parameter
                    u_c = float(np.sqrt(pcov[ci, ci])) if np.isfinite(pcov[ci, ci]) else float("nan")
                    best = (float(pk["mu"]), u_c, float(a))
    return best


# === Stress / strain from a folded-TO peak shift (report §2.3.4, Eqs 2 & 4) ===
# 6H-SiC folded-TO (E2) stress sensitivity and biaxial strain coefficient.
STRESS_COEFF_6H = 370.4      # MPa per cm⁻¹ ; σ_xx = −k·Δω
STRAIN_COEFF_6H = 0.0739     # % strain per cm⁻¹ ; ε_xx% = c·Δω


def stress_from_shift(omega, u_omega, omega0, u_omega0,
                      k=STRESS_COEFF_6H, strain_coeff=STRAIN_COEFF_6H):
    """In-plane biaxial stress and strain from a folded-TO peak shift.

    Δω = omega − omega0 (cm⁻¹); σ_xx = −k·Δω (MPa); ε_xx% = strain_coeff·Δω.
    The statistical 1σ is propagated from the fitted peak position and the
    reference position: u(Δω)=√(u_omega²+u_omega0²), u(σ)=k·u(Δω),
    u(ε)=strain_coeff·u(Δω).

    NOTE: this is the *statistical* part only. The literature spread in k and
    the uncertainty in the stress-free reference ω0 are additional systematics
    (report D1/D2) that must be stated separately wherever an absolute stress is
    quoted; they are not included here.
    """
    dw = float(omega - omega0)
    u_dw = float(np.hypot(u_omega, u_omega0))
    return {
        "delta_omega": dw, "u_delta_omega": u_dw,
        "sigma_MPa": -k * dw, "u_sigma_MPa": k * u_dw,
        "strain_pct": strain_coeff * dw, "u_strain_pct": strain_coeff * u_dw,
        "k": k, "strain_coeff": strain_coeff,
    }
