"""Refit Monte-Carlo test of the fit uncertainties (ticket C31).

What this validates: on a SYNTHETIC spectrum with KNOWN parameters and WHITE
Gaussian noise, the 1-sigma that the pipeline reports (curve_fit covariance,
scaled by the residual variance, propagated to FWHM by models.propagate_error)
matches the scatter of the same quantity over N independent noise realisations
that are each refitted from the same start, is unbiased, and covers the truth
at the nominal rate. It tests the code path only. It does NOT test real-data
accuracy: real residuals are not white (lag-1 autocorrelation ~0.5 on the Si 0.25 dpa/750 C residuals), and real fits carry
bound-pinned components whose covariance ignores the bound.

Result (N = 2000, seed 20261031): MC std / covariance sigma = 1.003 (centre),
0.993 (FWHM), 1.013 (amp), 0.979 (area ratio); all within the tolerances below.

Synthetic spectrum (FTO region): a sharp Lorentzian (788 cm-1, FWHM 6) on a
broad Gaussian (769 cm-1, FWHM 85), the shapes/widths of the Si-C TO fits in
params/configs/si_0.25dpa_750C.yaml. Baseline: the raw counts carry an order-5
polynomial baseline and are put through preprocessing's own IModPoly step
(order 5, tol 1e-3, max_iter 100, as in main.py) once, noise-free; it recovers
the peaks to RMS ~8e-5, ~0.6 % of the noise, so the processed curve equals the
truth to well below the MC resolution. White noise is then added to the
PROCESSED spectrum, which is the data the fitter sees (the baseline estimator's
own variance is outside any fit covariance, by design).

Noise level: sigma = 0.0118, the region RMSE of the real Si 0.25 dpa/750 C fit
(output/3 Si 750 0.25_fit_statistics.csv, region 170-1450, RMSE 0.01185; units
of vector-0to1-normalised intensity). Grid spacing 1.78 cm-1 as in that data.

Tolerances (N = 2000 noise realisations, fixed seed):
  * ratio |MC std / cov sigma - 1| < 3 / sqrt(2 (N-1)) = 0.0474 (3x the
    sampling error of a standard deviation);
  * bias |mean - truth| < 3 x (MC std / sqrt(N));
  * 1-sigma coverage of the truth within 3 binomial sigma of 68.27 % (+-3.1 %),
    using each draw's own reported sigma.
Runtime ~15 s, so no slow marker is needed.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curve_fitting import fit_peaks_regionwise  # noqa: E402
from derived_quantities import chi_ratio, by_center_range  # noqa: E402
from models import area, evaluate, propagate_error  # noqa: E402
from preprocessing import preprocess  # noqa: E402

N = 2000
SEED = 20261031
SIGMA_NOISE = 0.0118
RATIO_TOL = 3.0 / np.sqrt(2.0 * (N - 1))
NORMAL_1SIGMA = 0.682689492
COVER_TOL = 3.0 * np.sqrt(NORMAL_1SIGMA * (1 - NORMAL_1SIGMA) / N)

TRUE_L = (0.5, 788.0, 3.0)                 # lorentz: amp, centre, HWHM  -> FWHM 6
TRUE_G = (0.6, 769.0, 85.0 / 2.3548)       # gauss:   amp, centre, sigma -> FWHM 85
TRUE_THETA = np.r_[TRUE_L, TRUE_G]
FIT_REGION = (600.0, 1000.0)
# Start values: deliberately not the truth; the same for every realisation.
REGIONS = [(FIT_REGION[0], FIT_REGION[1], [
    {"model": "lorentz", "amp": 0.4, "center": 786.0, "width": 2.5, "assignment": "FTO"},
    {"model": "gauss", "amp": 0.5, "center": 770.0, "width": 30.0, "assignment": "broad"},
])]


def _band_area_ratio(theta):
    return area("lorentz", None, theta[:3]) / area("gauss", None, theta[3:6])


TRUTH = {
    "centre": TRUE_L[1],
    "FWHM": 2.0 * TRUE_L[2],
    "amp": TRUE_L[0],
    "area ratio": _band_area_ratio(TRUE_THETA),
}


@pytest.fixture(scope="module", autouse=True)
def _pin_blas_threads():
    """Pin BLAS to one thread for reproducibility (threadpoolctl is in
    requirements.txt; if it is missing, fall back to the seeded RNG and the
    statistical tolerances, which do not depend on thread count)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def synthetic():
    """x grid and the noise-free baseline-corrected spectrum."""
    x = np.arange(170.0, 2000.0, 1.78)
    u = (x - x[0]) / (x[-1] - x[0])
    baseline = 0.3 + 0.5 * u - 0.9 * u**2 + 1.2 * u**3 - 0.8 * u**4 + 0.3 * u**5
    signal = evaluate("lorentz", x, TRUE_L) + evaluate("gauss", x, TRUE_G)
    counts = 1000.0 * (signal + baseline)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "synthetic.csv")
        np.savetxt(path, np.c_[x, counts], delimiter=",", header="x,y", comments="")
        xp, yp = preprocess(path, crop_min=170, crop_max=2000, imodpoly_order=5,
                            imodpoly_tol=1e-3, imodpoly_max_iter=100,
                            normalisation="none", plot=False, save_path=None,
                            convert_wavelength_to_shift=False)
    yp = yp / 1000.0
    # The baseline step must not have distorted the truth beyond the noise.
    assert np.sqrt(np.mean((yp - signal) ** 2)) < 0.02 * SIGMA_NOISE
    return xp, yp


@pytest.fixture(scope="module")
def mc(synthetic):
    x, y0 = synthetic
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(N):
        y = y0 + rng.normal(0.0, SIGMA_NOISE, y0.size)
        _, _, peaks, _, region_fits = fit_peaks_regionwise(x, y, REGIONS, center_tolerance=100)
        sharp = peaks[0]
        theta, pcov = region_fits[0]["popt"], region_fits[0]["pcov"]
        rows.append((
            sharp["mu"], sharp["mu_err"],
            sharp["FWHM"], sharp["FWHM_err"],
            sharp["amp"], sharp["amp_err"],
            _band_area_ratio(theta), propagate_error(_band_area_ratio, theta, pcov),
            bool(sharp["flags"]) or bool(peaks[1]["flags"]),
        ))
    a = np.array(rows, dtype=float)
    return {
        "centre": (a[:, 0], a[:, 1]),
        "FWHM": (a[:, 2], a[:, 3]),
        "amp": (a[:, 4], a[:, 5]),
        "area ratio": (a[:, 6], a[:, 7]),
        "flagged": a[:, 8],
    }


QUANTITIES = ["centre", "FWHM", "amp", "area ratio"]


def test_synthetic_fits_are_clean(mc):
    assert mc["flagged"].sum() == 0          # nothing pinned/collapsed/drifted
    assert all(np.all(np.isfinite(mc[q][1])) for q in QUANTITIES)


@pytest.mark.parametrize("q", QUANTITIES)
def test_mc_std_matches_covariance_sigma(mc, q):
    vals, sig = mc[q]
    ratio = vals.std(ddof=1) / sig.mean()
    assert abs(ratio - 1.0) < RATIO_TOL, f"{q}: MC std / cov sigma = {ratio:.4f}, tol {RATIO_TOL:.4f}"


@pytest.mark.parametrize("q", QUANTITIES)
def test_no_bias(mc, q):
    vals, _ = mc[q]
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    assert abs(vals.mean() - TRUTH[q]) < 3.0 * se, f"{q}: bias {(vals.mean() - TRUTH[q]) / se:+.2f} SE"


@pytest.mark.parametrize("q", QUANTITIES)
def test_one_sigma_coverage(mc, q):
    vals, sig = mc[q]
    cover = np.mean(np.abs(vals - TRUTH[q]) <= sig)
    assert abs(cover - NORMAL_1SIGMA) < COVER_TOL, f"{q}: coverage {cover:.3f}, tol +-{COVER_TOL:.3f}"


def test_chi_ratio_agrees_with_propagate_error(synthetic):
    """derived_quantities.chi_ratio (its own Jacobian) and models.propagate_error
    on the same area ratio must give the same sigma for a single fit."""
    x, y0 = synthetic
    rng = np.random.default_rng(SEED)
    y = y0 + rng.normal(0.0, SIGMA_NOISE, y0.size)
    _, _, _, _, rf = fit_peaks_regionwise(x, y, REGIONS, center_tolerance=100)
    r = chi_ratio(rf, by_center_range(780, 800), by_center_range(750, 780))
    u_pe = propagate_error(_band_area_ratio, rf[0]["popt"], rf[0]["pcov"])
    # numerator selects the Lorentzian (788), denominator the Gaussian (769)
    assert abs(r["chi"] - _band_area_ratio(rf[0]["popt"])) < 1e-9
    assert abs(r["u_chi"] / u_pe - 1.0) < 1e-4
