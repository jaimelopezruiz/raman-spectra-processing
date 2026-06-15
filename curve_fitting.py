import numpy as np
from scipy.optimize import curve_fit

from models import N_PARAMS, evaluate, fwhm, area, propagate_error


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


# === Regional Fitting Function ===
def fit_peaks_regionwise(x_full, y_full, regions, center_tolerance=30):
    """Fit each spectral region independently with its user-assigned peaks.

    Regions: [(start, end, [peak_def, ...]), ...] where each peak_def is a
    tuple (model, amp, center, width[, q]) or a dict from a sample config.
    A dict peak may set an optional `width_max` to lower that peak's width
    upper bound (default 100), e.g. to keep a BWF from broadening to the bound.

    Returns (y_fit_total, fitted_peaks, peak_params, fit_stats).
    Parameter uncertainties are 1-sigma values from the curve_fit covariance
    matrix (scaled by the residual variance, since no per-point measurement
    errors are supplied) propagated to FWHM and area to first order.
    """
    y_fit_total = np.zeros_like(x_full)
    fitted_peaks = []
    peak_params = []
    fit_stats = []
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
                init += [amp, cen, min(wid, wmax), q]
                lb += [0, cen - center_tolerance, 1, -100]
                ub += [2 * amp, cen + center_tolerance, wmax, 100]
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

        popt, pcov = curve_fit(model, x_crop, y_crop, p0=init, bounds=(lb, ub), maxfev=100000)
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full_like(popt, np.nan)
        y_fit_total += model(x_full, *popt)

        # --- Goodness of fit for this region ---
        residuals = y_crop - model(x_crop, *popt)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_crop - np.mean(y_crop))**2))
        dof = max(len(x_crop) - len(popt), 1)
        fit_stats.append({
            "region": f"{start}-{end}",
            "n_points": len(x_crop),
            "n_params": len(popt),
            "R2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "RMSE": np.sqrt(ss_res / len(x_crop)),
            "residual_variance": ss_res / dof,
        })

        offset = 0
        for model_type, _, assignment in peaks:
            count = N_PARAMS[model_type]
            p = popt[offset:offset + count]
            p_err = perr[offset:offset + count]
            cov_block = pcov[offset:offset + count, offset:offset + count]
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
            })
            peak_counter += 1

    return y_fit_total, fitted_peaks, peak_params, fit_stats
