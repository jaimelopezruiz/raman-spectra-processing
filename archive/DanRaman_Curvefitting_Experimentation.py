#Raman Code Testing for better fitting by zooming in
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ramanspy as rp
from ramanspy import preprocessing

#Read Data
data = pd.read_csv("c:/Users/danie/OneDrive/Oxford 2025/03 Data/02 irradiated/Refel_Ne_300_2.5/Dark Grey Focused.csv", encoding="latin1", sep=",")
data_sorted = data.sort_values(by="#Wave")
x = data_sorted["#Wave"].values
y = data_sorted["#Intensity"].values

# Wrap & preprocess
spectrum = rp.Spectrum(y, x)

pipeline = preprocessing.Pipeline([
    preprocessing.denoise.SavGol(window_length=11, polyorder=3),
    preprocessing.baseline.IModPoly(poly_order=5),
    preprocessing.normalise.Vector()
])
processed = pipeline.apply(spectrum)

# === Store full spectrum just in case ===
x_proc_full = processed.spectral_axis
y_proc_full = processed.spectral_data


from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz

# === Peak functions ===
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def lorentzian(x, amp, cen, wid):
    return amp * (wid**2 / ((x - cen)**2 + wid**2))

def pseudo_voigt(x, amp, cen, wid, eta=0.5):
    g = gaussian(x, amp, cen, wid)
    l = lorentzian(x, amp, cen, wid)
    return eta * l + (1 - eta) * g

# === Define regions and peaks for fitting ===
regions = [
    (150, 400, [("gauss", 0.06, 200, 5), ("gauss", 0.06, 280, 5)]),
    (500, 600, [("pvoigt", 0.06, 525, 5), ("pvoigt", 0.06, 560, 5)]),
    (730, 945, [("gauss", 0.06, 875, 5), ("pvoigt", 0.06, 910, 5),("pvoigt", 0.06,775,5)]),
    (1300, 1650, [("gauss", 0.06, 1380, 10), ("gauss", 0.06, 1580, 10)])
]

x_grid = x_proc_full
y_fit_total = np.zeros_like(x_grid)
all_peak_centers = []

for start, end, peak_defs in regions:
    mask = (x_proc_full >= start) & (x_proc_full <= end)
    x_crop = x_proc_full[mask]
    y_crop = y_proc_full[mask]

    init, lb, ub = [], [], []
    for shape, amp, cen, wid in peak_defs:
        init += [amp, cen, wid]
        lb += [0, cen - 30, 1]
        ub += [2*amp, cen + 30, 100]
    init += [0.0]
    lb += [-1e-6]
    ub += [1e-6]


    def model(x, *params):
        y = np.zeros_like(x)
        for i, (shape, _, _, _) in enumerate(peak_defs):
            amp, cen, wid = params[3*i:3*i+3]
            if shape == "gauss":
                y += gaussian(x, amp, cen, wid)
            elif shape == "lorentz":
                y += lorentzian(x, amp, cen, wid)
            elif shape == "pvoigt":
                y += pseudo_voigt(x, amp, cen, wid)
        return y + params[-1]

    popt, _ = curve_fit(model, x_crop, y_crop, p0=init, bounds=(lb, ub), maxfev=100000)
    y_fit_total += model(x_grid, *popt)

    for i in range(len(peak_defs)):
        _, cen, _ = popt[3*i:3*i+3]
        all_peak_centers.append(cen)

# === Plot: full spectrum + combined fit and all individual peaks ===
plt.figure(figsize=(12, 6))

# Plot original processed spectrum
plt.plot(x_proc_full, y_proc_full, 'k-', label="Processed Spectrum")

# Plot total fit
plt.plot(x_proc_full, y_fit_total, 'r--', label="Summed Regional Fit")

# === Plot individual component peaks again, globally ===
# Redo loop to overlay components from each region
global_peak_counter = 1


for start, end, peak_defs in regions:
    mask = (x_proc_full >= start) & (x_proc_full <= end)
    x_crop = x_proc_full[mask]
    y_crop = y_proc_full[mask]

    init, lb, ub = [], [], []
    for shape, amp, cen, wid in peak_defs:
        init += [amp, cen, wid]
        lb += [0, cen - 30, 1]
        ub += [2*amp, cen + 30, 100]
    init += [0.0]
    lb += [-1e-6]
    ub += [1e-6]

    def model(x, *params):
        y = np.zeros_like(x)
        for i, (shape, _, _, _) in enumerate(peak_defs):
            amp, cen, wid = params[3*i:3*i+3]
            if shape == "gauss":
                y += gaussian(x, amp, cen, wid)
            elif shape == "lorentz":
                y += lorentzian(x, amp, cen, wid)
            elif shape == "pvoigt":
                y += pseudo_voigt(x, amp, cen, wid)
        return y + params[-1]

    popt, _ = curve_fit(model, x_crop, y_crop, p0=init, bounds=(lb, ub), maxfev=100000)

    # Plot individual peaks
    for i, (shape, _, _, _) in enumerate(peak_defs):
        amp, cen, wid = popt[3*i:3*i+3]
        if shape == "gauss":
            y_peak = gaussian(x_proc_full, amp, cen, wid)
        elif shape == "lorentz":
            y_peak = lorentzian(x_proc_full, amp, cen, wid)
        elif shape == "pvoigt":
            y_peak = pseudo_voigt(x_proc_full, amp, cen, wid)

        plt.plot(x_proc_full, y_peak, linestyle=':', linewidth=1,
                 label=f'Peak {global_peak_counter} ({shape}, {cen:.1f})')
        global_peak_counter += 1

# Final plot styling
plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Intensity (a.u.)")
plt.title("Combined Fit with All Component Peaks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Final labeled plot with staggered peak labels ===
plt.figure(figsize=(12, 6))
plt.plot(x_proc_full, y_proc_full, color='red', label='Processed Data')
#plt.plot(x_proc_full, y_fit_total, 'r--', label="Summed Regional Fit")

label_count = 0  # to stagger labels

# Loop over regions again to re-fit and plot vertical lines + text
for start, end, peak_defs in regions:
    mask = (x_proc_full >= start) & (x_proc_full <= end)
    x_crop = x_proc_full[mask]
    y_crop = y_proc_full[mask]

    init, lb, ub = [], [], []
    for shape, amp, cen, wid in peak_defs:
        init += [amp, cen, wid]
        lb += [0, cen - 30, 1]
        ub += [2*amp, cen + 30, 100]
    init += [0.0]
    lb += [-1e-6]
    ub += [1e-6]

    def model(x, *params):
        y = np.zeros_like(x)
        for i, (shape, _, _, _) in enumerate(peak_defs):
            amp, cen, wid = params[3*i:3*i+3]
            if shape == "gauss":
                y += gaussian(x, amp, cen, wid)
            elif shape == "lorentz":
                y += lorentzian(x, amp, cen, wid)
            elif shape == "pvoigt":
                y += pseudo_voigt(x, amp, cen, wid)
        return y + params[-1]

    popt, _ = curve_fit(model, x_crop, y_crop, p0=init, bounds=(lb, ub), maxfev=100000)

    for i in range(len(peak_defs)):
        _, cen, _ = popt[3*i:3*i+3]
        y_offset = max(y_proc_full) * (0.05 if label_count % 2 == 0 else 0.1)
        plt.axvline(x=cen, color='gray', linestyle='--', linewidth=1)
        plt.text(cen, y_offset, f"{cen:.1f}",
                 rotation=0, ha='center', va='bottom',
                 fontsize=9, color='black', fontweight='bold')
        label_count += 1

plt.xlabel("Raman Shift")
plt.ylabel("Intensity")
plt.title("Fitted Peak Centers (Wavenumbers)")
plt.grid(True)
plt.tight_layout()
plt.show()

def fwhm(w): return 2.3548 * abs(w)  # for Gaussian estimate

print("\nFitted Peaks (by Region):\n")

for region_idx, (start, end, peak_defs) in enumerate(regions):
    print(f"--- Region {region_idx + 1}: {start}–{end} cm⁻¹ ---")

    # Mask and prepare
    mask = (x_proc_full >= start) & (x_proc_full <= end)
    x_crop = x_proc_full[mask]
    y_crop = y_proc_full[mask]

    # Init + bounds
    init, lb, ub = [], [], []
    for shape, amp, cen, wid in peak_defs:
        init += [amp, cen, wid]
        lb += [0, cen - 30, 1]
        ub += [2*amp, cen + 30, 100]
    init += [0.0]
    lb += [-1e-6]
    ub += [1e-6]

    # Model
    def model(x, *params):
        y = np.zeros_like(x)
        for i, (shape, _, _, _) in enumerate(peak_defs):
            amp, cen, wid = params[3*i:3*i+3]
            if shape == "gauss":
                y += gaussian(x, amp, cen, wid)
            elif shape == "lorentz":
                y += lorentzian(x, amp, cen, wid)
            elif shape == "pvoigt":
                y += pseudo_voigt(x, amp, cen, wid)
        return y + params[-1]

    popt, _ = curve_fit(model, x_crop, y_crop, p0=init, bounds=(lb, ub), maxfev=100000)

    # Print each peak
    for i, (shape, _, _, _) in enumerate(peak_defs):
        amp, cen, wid = popt[3*i:3*i+3]
        if shape == "gauss":
            fwhm_val = 2.3548 * abs(wid)
            area = amp * wid * np.sqrt(2 * np.pi)
        elif shape == "lorentz":
            fwhm_val = 2 * wid
            area = amp * np.pi * wid
        elif shape == "pvoigt":
            fwhm_val = 0.5346 * 2 * wid + np.sqrt(0.2166 * (2 * wid)**2 + (2.3548 * wid)**2)
            area = amp * wid * np.sqrt(2 * np.pi)  # rough approx

        print(f"Peak {i+1} ({shape}):")
        print(f"  Center = {cen:.2f} cm⁻¹")
        print(f"  FWHM   = {fwhm_val:.2f} cm⁻¹")
        print(f"  Height = {amp:.3f}")
        print(f"  Area   = {area:.3f}\n")


# for cen in all_peak_centers:
#     plt.axvline(cen, color="gray", linestyle="--", linewidth=1)
#     plt.text(cen, max(y_proc_full)*0.05, f"{cen:.1f}", ha='center', va='bottom',
#              fontsize=9, color='black', fontweight='bold')




# # === Print Origin-style peak table ===
# def fwhm(w): return 2.3548 * abs(w)  # for Gaussian estimate

# print("Fitted Peaks:\n")
# for i, (shape, _, _, _) in enumerate(peak_definitions):
#     amp, cen, wid = popt[3*i:3*i+3]
#     area = amp * wid * np.sqrt(2 * np.pi)  # rough estimate
#     print(f"Peak {i+1} ({shape}):")
#     print(f"  Center = {cen:.2f} cm⁻¹")
#     print(f"  FWHM   = {fwhm(wid):.2f} cm⁻¹")
#     print(f"  Height = {amp:.3f}")
#     print(f"  Area   = {area:.3f}\n")


# # === Final labeled plot with bold, staggered wavenumber labels ===
# plt.figure(figsize=(12, 6))
# plt.plot(x_proc, y_proc, color='red', label='Processed Data')

# # Stagger heights: alternate label height to reduce overlap
# for i in range(len(peak_definitions)):
#     _, cen, _ = popt[3*i:3*i+3]
#     y_offset = max(y_proc) * (0.05 if i % 2 == 0 else 0.1)  # staggered
#     plt.axvline(x=cen, color='gray', linestyle='--', linewidth=1)
#     plt.text(cen, y_offset, f"{cen:.1f}",
#              rotation=0, ha='center', va='bottom',
#              fontsize=9, color='black', fontweight='bold')

# plt.xlabel("Raman Shift")
# plt.ylabel("Intensity")
# plt.title("Fitted Peak Centers (Wavenumbers)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#import matplotlib.pyplot as plt
#import textwrap

# === Collect peak info into a string ===
# def fwhm(w): return 2.3548 * abs(w)

# peak_text = "Fitted Peaks:\n\n"
# for i, (shape, _, _, _) in enumerate(peak_definitions):
#     amp, cen, wid = popt[3*i:3*i+3]
#     area = amp * wid * np.sqrt(2 * np.pi)
#     peak_text += (
#         f"Peak {i+1} ({shape}):\n"
#         f"  Center = {cen:.2f} cm⁻¹\n"
#         f"  FWHM   = {fwhm(wid):.2f} cm⁻¹\n"
#         f"  Height = {amp:.3f}\n"
#         f"  Area   = {area:.3f}\n\n"
#     )

# # === Display it in a figure as text ===
# plt.figure(figsize=(8, 6))
# plt.axis("off")
# wrapped_text = textwrap.fill(peak_text, width=80)
# plt.text(0.01, 0.99, peak_text, fontsize=10, va='top', ha='left', family='monospace')
# plt.title("Peak Fit Summary", fontsize=12, fontweight='bold')
# plt.tight_layout()
# plt.show()

