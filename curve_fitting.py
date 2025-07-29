import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz

# === Peak Models ===
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

def bwf(x, amp, cen, wid, q):
    s = (x - cen) / wid
    return amp * ((1 + s / q) ** 2) / (1 + s**2)



# === Regional Fitting Function ===
def fit_peaks_regionwise(x_full, y_full, regions, center_tolerance=30):
    y_fit_total = np.zeros_like(x_full)
    fitted_peaks = []
    peak_params = []
    peak_counter = 1

    for region_index, (start, end, peak_defs) in enumerate(regions):
        mask = (x_full >= start) & (x_full <= end)
        x_crop = x_full[mask]
        y_crop = y_full[mask]

        init, lb, ub = [], [], []
        param_counts = []

        for peak_def in peak_defs:
            model_type = peak_def[0]

            if model_type == "bwf":
                amp, cen, wid, q = peak_def[1:]
                init += [amp, cen, wid, q]
                lb += [0, cen - center_tolerance, 1, -100]
                ub += [2 * amp, cen + center_tolerance, 100, 100]
                param_counts.append(4)
            else:
                amp, cen, wid = peak_def[1:]
                init += [amp, cen, wid]
                lb += [0, cen - center_tolerance, 1]
                ub += [2 * amp, cen + center_tolerance, 100]
                param_counts.append(3)

        # Add baseline param
        init += [0.0]
        lb += [-1e-6]
        ub += [1e-6]

        def model(x, *params):
            y = np.zeros_like(x)
            offset = 0
            for peak_def, count in zip(peak_defs, param_counts):
                shape = peak_def[0]
                if shape == "bwf":
                    amp, cen, wid, q = params[offset:offset+4]
                    y += bwf(x, amp, cen, wid, q)
                else:
                    amp, cen, wid = params[offset:offset+3]
                    if shape == "gauss":
                        y += gaussian(x, amp, cen, wid)
                    elif shape == "lorentz":
                        y += lorentzian(x, amp, cen, wid)
                    elif shape == "pvoigt" or shape == "voigt":
                        y += true_voigt(x, amp, cen, wid)
                offset += count
            return y + params[-1]

        popt, _ = curve_fit(model, x_crop, y_crop, p0=init, bounds=(lb, ub), maxfev=100000)
        y_fit_total += model(x_full, *popt)

        offset = 0
        for peak_def, count in zip(peak_defs, param_counts):
            shape = peak_def[0]

            if shape == "bwf":
                amp, cen, wid, q = popt[offset:offset+4]
                y_peak = bwf(x_full, amp, cen, wid, q)
                fwhm = 2 * abs(wid)  # crude estimate
                area = np.trapz(y_peak, x_full)
            else:
                amp, cen, wid = popt[offset:offset+3]
                if shape == "gauss":
                    y_peak = gaussian(x_full, amp, cen, wid)
                    fwhm = 2.3548 * abs(wid)
                    area = amp * wid * np.sqrt(2 * np.pi)
                elif shape == "lorentz":
                    y_peak = lorentzian(x_full, amp, cen, wid)
                    fwhm = 2 * wid
                    area = amp * np.pi * wid
                elif shape == "pvoigt" or shape == "voigt":
                    y_peak = true_voigt(x_full, amp, cen, wid)
                    fwhm = wid
                    area = amp
            offset += count

            fitted_peaks.append((x_full, y_peak))
            peak_params.append({
                "peak": peak_counter,
                "model": shape,
                "mu": cen,
                "FWHM": fwhm,
                "Area": area,
                "Relative_Intensity": np.max(y_peak)
            })
            peak_counter += 1

    return y_fit_total, fitted_peaks, peak_params



