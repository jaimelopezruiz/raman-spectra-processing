import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.integrate import simps

# === Peak Models ===
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def lorentzian(x, A, mu, gamma):
    return A * gamma**2 / ((x - mu)**2 + gamma**2)

def voigt(x, A, mu, sigma, gamma):
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    return A * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

model_funcs = {
    "gaussian": gaussian,
    "lorentzian": lorentzian,
    "voigt": voigt
}

# === Fit Function ===
def fit_peaks(x_full, y_full, peak_groups, bounds=None, maxfev=20000, auto_bounds=True):
    """
    Fits one or more peak groups to the provided spectrum.

    Parameters:
        x_full (array): Full Raman shift values.
        y_full (array): Full intensity values.
        peak_groups (list): List of dicts. Each defines a group:
            {
              "model": ["gaussian", "voigt"],
              "center": [770, 900],
              "window": 80
            }
        bounds (tuple or None): (lower, upper) bounds for parameters.
        maxfev (int): Maximum number of function evaluations.

    Returns:
        tuple: (y_fit_total, fitted_peaks, peak_params)
    """

    y_fit_total = np.zeros_like(x_full)
    fitted_peaks = []
    peak_params = []

    for i, group in enumerate(peak_groups):
        is_group = isinstance(group["center"], (list, tuple))
        centers = group["center"] if is_group else [group["center"]]
        models = group["model"] if is_group else [group["model"]]
        window = group["window"]

        x_min = min(centers) - window
        x_max = max(centers) + window
        mask = (x_full >= x_min) & (x_full <= x_max)
        x = x_full[mask]
        y = y_full[mask]

        # Composite function
        def composite(x, *params):
            y_sum = np.zeros_like(x)
            idx = 0
            for m in models:
                if m == "gaussian":
                    A, mu, sigma = params[idx:idx+3]
                    y_sum += gaussian(x, A, mu, sigma)
                    idx += 3
                elif m == "lorentzian":
                    A, mu, gamma = params[idx:idx+3]
                    y_sum += lorentzian(x, A, mu, gamma)
                    idx += 3
                elif m == "voigt":
                    A, mu, sigma, gamma = params[idx:idx+4]
                    y_sum += voigt(x, A, mu, sigma, gamma)
                    idx += 4
            return y_sum

        # Initial guess
        p0 = []
        for center, m in zip(centers, models):
            A0 = max(y)
            if m == "gaussian":
                p0 += [A0, center, 10]
            elif m == "lorentzian":
                p0 += [A0, center, 10]
            elif m == "voigt":
                p0 += [A0, center, 10, 10]

        # Fit
        try:
            if auto_bounds:
                lower, upper = [], []
                for center, m in zip(centers, models):
                    if m == "gaussian" or m == "lorentzian":
                        lower += [0, center - 100, 1]
                        upper += [np.max(y), center + 100, 100]
                    elif m == "voigt":
                        lower += [0, center - 100, 1, 1]
                        upper += [np.max(y), center + 100, 100, 100]
                bounds = (lower, upper)
            popt, _ = curve_fit(composite, x, y, p0=p0, maxfev=maxfev, bounds=bounds or (-np.inf, np.inf))
            y_fit = composite(x_full, *popt)
            y_fit_total += y_fit
            fitted_peaks.append((x_full, y_fit))

            idx = 0
            for j, m in enumerate(models):
                row = {
                    "peak": f"{i+1}.{j+1}" if is_group else f"{i+1}",
                    "model": m
                }

                A = popt[idx]
                mu = popt[idx + 1]
                row["A"] = A
                row["mu"] = mu

                if m == "gaussian":
                    sigma = popt[idx + 2]
                    fwhm = 2.3548 * sigma
                    area = A * sigma * np.sqrt(2 * np.pi)
                    idx += 3

                elif m == "lorentzian":
                    gamma = popt[idx + 2]
                    fwhm = 2 * gamma
                    area = A * np.pi * gamma
                    idx += 3

                elif m == "voigt":
                    sigma = popt[idx + 2]
                    gamma = popt[idx + 3]
                    fwhm = 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma)**2 + (2.3548 * sigma)**2)
                    peak_y = voigt(x_full, A, mu, sigma, gamma)
                    area = simps(peak_y, x_full)
                    idx += 4

                row["FWHM"] = fwhm
                row["Area"] = area
                row["Relative_Intensity"] = A / np.max(y_full)

                peak_params.append(row)

        except Exception as e:
            print(f"[!] Fit failed for group {i+1}: {e}")

    return y_fit_total, fitted_peaks, peak_params
