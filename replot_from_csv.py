"""Rebuild the fitted-spectrum figure from the CSVs saved by main.py.

Peak-parameter CSVs written by the current pipeline include the raw model
parameters (amp, wid, q), so every peak — including BWF — is reconstructed
exactly. Older CSVs without those columns fall back to inverting the FWHM,
which is approximate for voigt and unavailable for bwf.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from models import evaluate, gaussian, lorentzian, true_voigt, pseudo_voigt

# === GUI Folder Selection ===
def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Folder with Raman Output CSVs")
    return folder_selected

# === Load CSV Files ===
def load_csv_files(folder):
    files = os.listdir(folder)
    processed_file = [f for f in files if f.endswith("_processed.csv")]
    fitted_file = [f for f in files if f.endswith("_fitted_curve.csv")]
    params_file = [f for f in files if f.endswith("_peak_parameters.csv")]

    if not (processed_file and fitted_file and params_file):
        raise FileNotFoundError("One or more required CSV files not found in selected folder.")

    df_processed = pd.read_csv(os.path.join(folder, processed_file[0]))
    df_fitted = pd.read_csv(os.path.join(folder, fitted_file[0]))
    df_params = pd.read_csv(os.path.join(folder, params_file[0]))

    df_params = df_params.rename(columns={
        "Model": "model",
        "Assignment": "assignment",
        "Center (cm⁻¹)": "mu",
        "FWHM (cm⁻¹)": "FWHM",
        "Relative Intensity": "Relative_Intensity",
    })

    return df_processed, df_fitted, df_params

# === Reconstruct Peaks ===
def reconstruct_peaks(x, df_params, eta=0.5):
    has_raw_params = {"amp", "wid"}.issubset(df_params.columns)

    peak_curves = []
    for _, row in df_params.iterrows():
        model = row['model']

        if has_raw_params and np.isfinite(row['amp']) and np.isfinite(row['wid']):
            # Exact reconstruction from the stored raw model parameters
            params = [row['amp'], row['mu'], row['wid']]
            if model == 'bwf':
                params.append(row['q'])
            y = evaluate(model, x, params)
            peak_curves.append((x, y))
            continue

        # --- Legacy fallback: invert the FWHM (approximate for voigt) ---
        amp = row['Relative_Intensity']
        mu = row['mu']
        if model == 'gauss':
            wid = row['FWHM'] / 2.3548
            y = gaussian(x, amp, mu, wid)
        elif model == 'lorentz':
            wid = row['FWHM'] / 2
            y = lorentzian(x, amp, mu, wid)
        elif model == 'voigt':
            wid = row['FWHM'] / (0.5346 * 2 + np.sqrt(0.2166 * (2) ** 2 + 2.3548 ** 2))
            y = true_voigt(x, amp, mu, wid)
        elif model == 'pvoigt':
            wid = row['FWHM'] / (0.5346 * 2 + np.sqrt(0.2166 * (2) ** 2 + 2.3548 ** 2))
            y = pseudo_voigt(x, amp, mu, wid, eta)
        elif model == 'bwf':
            raise ValueError(
                "Cannot reconstruct 'bwf' peaks from a legacy peak-parameter CSV: the "
                "asymmetry parameter q is not stored. Re-run the fit with the current "
                "pipeline, which saves the raw parameters."
            )
        else:
            raise ValueError(f"Unknown model type: {model}")

        peak_curves.append((x, y))
    return peak_curves

# === Plotting Function ===
def plot_raman_spectrum(x, y, y_fit, peaks, df_params):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Processed Spectrum', color='black')
    plt.plot(x, y_fit, 'r--', label='Fitted Total Curve')

    for i, (x_peak, y_peak) in enumerate(peaks):
        model = df_params.iloc[i]["model"]
        mu = df_params.iloc[i]["mu"]
        plt.plot(x_peak, y_peak, linestyle=':', label=f'Peak {i+1} ({model}, {mu:.1f})')

    for i, row in df_params.iterrows():
        mu = row["mu"]
        y_offset = max(y) * (0.05 if i % 2 == 0 else 0.1)
        plt.axvline(x=mu, color='gray', linestyle='--', linewidth=1)
        plt.text(mu, y_offset, f"{mu:.1f}",
                 rotation=0, ha='center', va='bottom',
                 fontsize=9, color='black', fontweight='bold')

    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.title("Reconstructed Fitted Raman Spectrum")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Main Execution ===
if __name__ == '__main__':
    folder = select_folder()
    df_processed, df_fitted, df_params = load_csv_files(folder)
    x = df_processed.iloc[:, 0].values
    y = df_processed.iloc[:, 1].values
    y_fit = df_fitted.iloc[:, 1].values
    peaks = reconstruct_peaks(x, df_params)
    plot_raman_spectrum(x, y, y_fit, peaks, df_params)
