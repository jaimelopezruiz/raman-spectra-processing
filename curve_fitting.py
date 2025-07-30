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

def double_voigt(x, amp1, amp2, cen, wid1, wid2):
    """
    A sum of two Voigt profiles sharing the same center.
    amp1, wid1: amplitude and width of the narrower Voigt
    amp2, wid2: amplitude and width of the broader Voigt
    cen: shared center position (cm⁻¹)
    """
    profile1 = true_voigt(x, amp1, cen, wid1)
    profile2 = true_voigt(x, amp2, cen, wid2)
    return profile1 + profile2


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

            elif model_type == "double_voigt":
                amp1, amp2, cen, wid1, wid2 = peak_def[1:]
                # Shared tolerance on amplitude difference
                init += [amp1, amp2, cen, wid1, wid2]
                lb += [0, 0, cen - center_tolerance, 1, 1]
                ub += [2 * amp1, 2 * amp2, cen + center_tolerance, 100, 100]
                param_counts.append(5)  


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

                elif shape == "double_voigt":
                    amp1, amp2, cen, wid1, wid2 = params[offset:offset+5]
                    y += true_voigt(x, amp1, cen, wid1) + true_voigt(x, amp2, cen, wid2)

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
                fwhm = 2 * abs(wid)
                area = np.trapz(y_peak, x_full)

            elif shape == "double_voigt":
                amp1, amp2, cen, wid1, wid2 = popt[offset:offset+5]
                y_voigt1 = true_voigt(x_full, amp1, cen, wid1)
                y_voigt2 = true_voigt(x_full, amp2, cen, wid2)

                fitted_peaks.append((x_full, y_voigt1))
                peak_params.append({
                    "peak": peak_counter,
                    "model": "double_voigt_narrow",
                    "mu": cen,
                    "FWHM": 2.3548 * abs(wid1),
                    "Area": amp1 * wid1 * np.sqrt(2 * np.pi),
                    "Relative_Intensity": np.max(y_voigt1)
                })
                peak_counter += 1

                fitted_peaks.append((x_full, y_voigt2))
                peak_params.append({
                    "peak": peak_counter,
                    "model": "double_voigt_broad",
                    "mu": cen,
                    "FWHM": 2.3548 * abs(wid2),
                    "Area": amp2 * wid2 * np.sqrt(2 * np.pi),
                    "Relative_Intensity": np.max(y_voigt2)
                })
                peak_counter += 1

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

            # ONLY append here for non-double_voigt cases
            if shape != "double_voigt":
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



