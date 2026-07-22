import numpy as np
from scipy.optimize import curve_fit

from models import N_PARAMS, evaluate, fwhm, area, propagate_error

# A fitted centre that used up more than this fraction of its allowed ± window
# (center_tolerance) is flagged: the wide window is kept for robustness, but a
# large drift from the literature seed is surfaced so its assignment label is
# never taken at face value.
DRIFT_FRACTION = 0.5


def _normalise_peak_def(peak_def):
    """Accept a peak definition as either a legacy tuple
    (model, amp, center, width[, q]) or a config dict
    {model, amp, center, width[, q][, assignment]}.
    Returns (model, init_params, assignment)."""
    if isinstance(peak_def, dict):
        model = peak_def["model"]
        params = [peak_def["amp"], peak_def["center"], peak_def["width"]]
        if model == "bwf":
            params.append(peak_def["q"])
        return model, params, peak_def.get("assignment", "")
    model = peak_def[0]
    return model, list(peak_def[1:]), ""


def _bound_flags(model_type, p, lb, ub, seed, rtol=1e-3):
    """Flag a fitted peak whose parameters landed at a fit bound or collapsed.

    A peak pinned at a bound, whose amplitude has gone to ~0, or whose centre
    has drifted far from its seed is a fit artefact (or an assignment that no
    longer matches the fitted position) rather than a physically refined band:
    the reported centre/FWHM/area should not be read at face value.
    Returns a list of short human-readable flag strings (empty = clean).
    """
    flags = []
    amp, cen, wid = p[0], p[1], p[2]
    seed_amp = seed[0]
    if seed_amp > 0 and amp < 1e-3 * seed_amp:
        flags.append("amp~0 (peak effectively absent)")
    elif amp >= ub[0] * (1 - rtol):
        flags.append("amp at upper bound (2x seed)")
    if wid <= lb[2] * (1 + rtol):
        flags.append("width at lower bound")
    elif wid >= ub[2] * (1 - rtol):
        flags.append("width at upper bound")
    span = ub[1] - lb[1]
    if span > 0:
        if cen <= lb[1] + rtol * span or cen >= ub[1] - rtol * span:
            flags.append("centre at +/- tolerance bound")
        elif abs(cen - seed[1]) > DRIFT_FRACTION * (span / 2.0):
            flags.append(f"centre drifted {abs(cen - seed[1]):.0f} cm-1 from seed")
    if model_type == "bwf":
        q, qspan = p[3], (ub[3] - lb[3])
        if qspan > 0 and (q <= lb[3] + rtol * qspan or q >= ub[3] - rtol * qspan):
            flags.append("q at bound")
    return flags


# === Regional Fitting Function ===
def fit_peaks_regionwise(x_full, y_full, regions, center_tolerance=30):
    """Fit each spectral region independently with its user-assigned peaks.

    Regions: [(start, end, [peak_def, ...]), ...] where each peak_def is a
    tuple (model, amp, center, width[, q]) or a dict from a sample config.
    A dict peak may set an optional `width_max` to lower that peak's width
    upper bound (default 100), e.g. to keep a BWF from broadening to the bound.

    Returns (y_fit_total, fitted_peaks, peak_params, fit_stats, region_fits).
    Parameter uncertainties are 1-sigma values from the curve_fit covariance
    matrix (scaled by the residual variance, since no per-point measurement
    errors are supplied) propagated to FWHM and area to first order.

    Each peak_params entry also carries a `flags` string: non-empty when the
    peak landed at a fit bound or collapsed to ~zero amplitude, i.e. when its
    reported parameters are a fit artefact rather than a refined physical band.

    region_fits is a list (one dict per region) exposing the raw fit for
    downstream derived-quantity propagation (e.g. the chemical-disorder ratio
    chi): {region, start, end, popt, pcov, peaks:[{peak, model, assignment,
    mu, i0, n}]}, where i0/n index that peak's parameters within popt/pcov.
    This is what makes a *correct* (covariance-aware) uncertainty possible for
    quantities that combine several peaks fitted together in one region.
    """
    y_fit_total = np.zeros_like(x_full)
    fitted_peaks = []
    peak_params = []
    fit_stats = []
    region_fits = []
    peak_counter = 1

    for region_index, (start, end, peak_defs) in enumerate(regions):
        mask = (x_full >= start) & (x_full <= end)
        x_crop = x_full[mask]
        y_crop = y_full[mask]

        peaks = [_normalise_peak_def(pd) for pd in peak_defs]

        init, lb, ub = [], [], []
        for (model_type, p0, _), pd in zip(peaks, peak_defs):
            # Optional per-peak width ceiling (defaults to 100 so configs that
            # don't set it fit identically); lets a peak be held narrow, e.g. a
            # modest BWF that would otherwise balloon to the bound to absorb a
            # broad neighbouring hump.
            wmax = pd.get("width_max", 100) if isinstance(pd, dict) else 100
            if model_type == "bwf":
                amp, cen, wid, q = p0
                # Optional per-peak bounds on the BWF asymmetry q (default the
                # full [-100, 100]). The line shape is singular at q = 0, where
                # it degenerates into a flat offset; bounding q to one sign and
                # away from zero (e.g. q_min: 0.5) keeps the peak a real peak.
                qmin = pd.get("q_min", -100) if isinstance(pd, dict) else -100
                qmax = pd.get("q_max", 100) if isinstance(pd, dict) else 100
                init += [amp, cen, min(wid, wmax), min(max(q, qmin), qmax)]
                lb += [0, cen - center_tolerance, 1, qmin]
                ub += [2 * amp, cen + center_tolerance, wmax, qmax]
            else:
                amp, cen, wid = p0
                init += [amp, cen, min(wid, wmax)]
                lb += [0, cen - center_tolerance, 1]
                ub += [2 * amp, cen + center_tolerance, wmax]

        def model(x, *params):
            y = np.zeros_like(x)
            offset = 0
            for model_type, p0, _ in peaks:
                count = N_PARAMS[model_type]
                y += evaluate(model_type, x, params[offset:offset + count])
                offset += count
            return y

        try:
            popt, pcov = curve_fit(model, x_crop, y_crop, p0=init, bounds=(lb, ub), maxfev=100000)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"[!] Curve fit failed for region {start}-{end} cm^-1 "
                f"({len(x_crop)} points, {len(init)} parameters): {exc}. "
                "Check the region's seed parameters/bounds in the config."
            ) from exc
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full_like(popt, np.nan)
        y_fit_total += model(x_full, *popt)

        region_fit = {
            "region": f"{start}-{end}", "start": start, "end": end,
            "popt": np.asarray(popt, dtype=float),
            "pcov": np.asarray(pcov, dtype=float),
            "peaks": [],
        }
        region_fits.append(region_fit)

        # --- Goodness of fit for this region ---
        residuals = y_crop - model(x_crop, *popt)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_crop - np.mean(y_crop))**2))
        dof = max(len(x_crop) - len(popt), 1)
        fit_stats.append({
            "region": f"{start}-{end}",
            "n_points": len(x_crop),
            "n_params": len(popt),
            "dof": dof,
            "R2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "RMSE": np.sqrt(ss_res / len(x_crop)),
            "residual_variance": ss_res / dof,
        })

        offset = 0
        for model_type, seed, assignment in peaks:
            count = N_PARAMS[model_type]
            sl = slice(offset, offset + count)
            p = popt[sl]
            p_err = perr[sl]
            cov_block = pcov[sl, sl]
            flags = _bound_flags(model_type, p, lb[sl], ub[sl], seed)
            region_fit["peaks"].append({
                "peak": peak_counter, "model": model_type,
                "assignment": assignment, "mu": p[1], "i0": offset, "n": count,
                "flags": "; ".join(flags),
            })
            offset += count

            y_peak = evaluate(model_type, x_full, p)
            peak_fwhm = fwhm(model_type, p)
            peak_area = area(model_type, x_full, p)
            fwhm_err = propagate_error(lambda pp: fwhm(model_type, pp), p, cov_block)
            area_err = propagate_error(lambda pp: area(model_type, x_full, pp), p, cov_block)

            fitted_peaks.append((x_full, y_peak))
            peak_params.append({
                "peak": peak_counter,
                "model": model_type,
                "assignment": assignment,
                "region": f"{start}-{end}",
                "mu": p[1],
                "mu_err": p_err[1],
                "FWHM": peak_fwhm,
                "FWHM_err": fwhm_err,
                "Area": peak_area,
                "Area_err": area_err,
                "Relative_Intensity": np.max(y_peak),
                "amp": p[0],
                "amp_err": p_err[0],
                "wid": p[2],
                "wid_err": p_err[2],
                "q": p[3] if model_type == "bwf" else np.nan,
                "q_err": p_err[3] if model_type == "bwf" else np.nan,
                "flags": "; ".join(flags),
            })
            peak_counter += 1

    return y_fit_total, fitted_peaks, peak_params, fit_stats, region_fits
