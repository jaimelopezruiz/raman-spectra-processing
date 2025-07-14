#Raman Code Testing for better fitting by zooming in
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ramanspy as rp
from ramanspy import preprocessing

#Read Data
data = pd.read_csv("c:/Users/danie/OneDrive/Oxford 2025/03 Data/02 irradiated/Refel_Ne_300_2.5/Dark Grey.csv", encoding="latin1", sep=",")
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

# === Zoom in to region of interest (e.g. 700–1000 cm⁻¹) ===
mask = (x_proc_full >= 700) & (x_proc_full <= 1000)
x_proc = x_proc_full[mask]
y_proc = y_proc_full[mask]


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



# === Define peaks: ("shape", amplitude, center, width) ===
peak_definitions = [
    ("pvoigt", 0.1, 790, 5),
    ("gauss",  0.1, 880, 10),
    ("pvoigt", 0.2, 910, 10),
]


baseline_offset = 0.0

# === Build initial guess from peak_definitions ===
initial_guesses = []
for shape, amp, cen, wid in peak_definitions:
    initial_guesses.extend([amp, cen, wid])
initial_guesses.append(baseline_offset)

# === Mixed model using all shapes ===
def mixed_model(x, *params):
    y = np.zeros_like(x)
    for i, (shape, _, _, _) in enumerate(peak_definitions):
        amp, cen, wid = params[3*i:3*i+3]
        if shape == "gauss":
            y += gaussian(x, amp, cen, wid)
        elif shape == "lorentz":
            y += lorentzian(x, amp, cen, wid)
        elif shape == "pvoigt":
            y += pseudo_voigt(x, amp, cen, wid)

        else:
            raise ValueError(f"Unknown peak shape: {shape}")
    return y + params[-1]

#set boundaries
# === Create parameter bounds to constrain the fit ===
lower_bounds = []
upper_bounds = []

for shape, amp, cen, wid in peak_definitions:
    lower_bounds += [0, cen - 100, 1]          # amplitude > 0, width > 1
    upper_bounds += [2*amp, cen + 100, 100]    # reasonable amplitude/width limits

lower_bounds.append(-0.2)  # offset
upper_bounds.append(0.2)



# === Fit the model to your preprocessed spectrum ===
popt, pcov = curve_fit(mixed_model, x_proc, y_proc, p0=initial_guesses, bounds = (lower_bounds, upper_bounds), maxfev = 200000)

# === Plot: Raw data, total fit, and each peak ===
plt.figure(figsize=(12, 6))
plt.plot(x_proc, y_proc, 'k-', label='Processed Data (Full)')
plt.plot(x_proc, mixed_model(x_proc, *popt), 'r--', label='Total Fit')

n_peaks = len(peak_definitions)
for i, (shape, _, _, _) in enumerate(peak_definitions):
    amp, cen, wid = popt[3*i:3*i+3]
    if shape == "gauss":
        peak = gaussian(x_proc, amp, cen, wid)
    elif shape == "lorentz":
        peak = lorentzian(x_proc, amp, cen, wid)
    elif shape == "pvoigt":
        peak = pseudo_voigt(x_proc, amp, cen, wid)

    plt.plot(x_proc, peak, linestyle=':', label=f'Peak {i+1} ({shape}, {cen:.1f} cm⁻¹)')

plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Intensity (a.u.)")
plt.title("Mixed-Peak Fit (Zoomed 700–1000 cm⁻¹)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print Origin-style peak table ===
def fwhm(w): return 2.3548 * abs(w)  # for Gaussian estimate

print("Fitted Peaks:\n")
for i, (shape, _, _, _) in enumerate(peak_definitions):
    amp, cen, wid = popt[3*i:3*i+3]
    area = amp * wid * np.sqrt(2 * np.pi)  # rough estimate
    print(f"Peak {i+1} ({shape}):")
    print(f"  Center = {cen:.2f} cm⁻¹")
    print(f"  FWHM   = {fwhm(wid):.2f} cm⁻¹")
    print(f"  Height = {amp:.3f}")
    print(f"  Area   = {area:.3f}\n")


# === Final labeled plot with bold, staggered wavenumber labels ===
plt.figure(figsize=(12, 6))
plt.plot(x_proc, y_proc, color='red', label='Processed Data')

# Stagger heights: alternate label height to reduce overlap
for i in range(len(peak_definitions)):
    _, cen, _ = popt[3*i:3*i+3]
    y_offset = max(y_proc) * (0.05 if i % 2 == 0 else 0.1)  # staggered
    plt.axvline(x=cen, color='gray', linestyle='--', linewidth=1)
    plt.text(cen, y_offset, f"{cen:.1f}",
             rotation=0, ha='center', va='bottom',
             fontsize=9, color='black', fontweight='bold')

plt.xlabel("Raman Shift")
plt.ylabel("Intensity")
plt.title("Fitted Peak Centers (Wavenumbers)")
plt.grid(True)
plt.tight_layout()
plt.show()

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

