import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap

# === File Picker ===
def choose_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with Raman CSV Outputs")
    return folder_path

# === Load Data ===
def load_csv_files(folder):
    processed_file = next(f for f in os.listdir(folder) if f.endswith("_processed.csv"))
    fitted_file = next(f for f in os.listdir(folder) if f.endswith("_fitted_curve.csv"))
    params_file = next(f for f in os.listdir(folder) if f.endswith("_peak_parameters.csv"))

    df_processed = pd.read_csv(os.path.join(folder, processed_file))
    df_fitted = pd.read_csv(os.path.join(folder, fitted_file))
    df_params = pd.read_csv(os.path.join(folder, params_file))

    return df_processed, df_fitted, df_params

# === Main Plot ===
def plot_fit(df_proc, df_fit, df_params, show_components=True):
    x = df_proc["Raman Shift (cm-1)"]
    y = df_proc["Processed Intensity"]
    y_fit = df_fit["Fitted Intensity"]

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'k-', label="Processed Data")
    plt.plot(x, y_fit, 'r--', label="Total Fit")

    if show_components:
        for i, row in df_params.iterrows():
            model = row["Model"]
            mu = row["Center (cm⁻¹)"]
            fwhm = row["FWHM (cm⁻¹)"]
            height = row["Relative Intensity"]
            if model == "gauss":
                sigma = fwhm / 2.3548
                y_peak = height * np.exp(-(x - mu)**2 / (2 * sigma**2))
            elif model == "lorentz":
                gamma = fwhm / 2
                y_peak = height * gamma**2 / ((x - mu)**2 + gamma**2)
            elif model == "pvoigt":
                sigma = fwhm / 2.3548  # approximate
                gamma = fwhm / 2       # approximate
                g = height * np.exp(-(x - mu)**2 / (2 * sigma**2))
                l = height * gamma**2 / ((x - mu)**2 + gamma**2)
                y_peak = 0.5 * l + 0.5 * g
            else:
                continue

            plt.plot(x, y_peak, linestyle=':', linewidth=1,
                     label=f'Peak {int(row["Peak"])} ({model}, {mu:.1f})')

    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.title("Fitted Raman Spectrum with Peak Centers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Peak Center Plot ===
def plot_peak_centers(df_proc, df_params):
    x = df_proc["Raman Shift (cm-1)"]
    y = df_proc["Processed Intensity"]

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, color='red', label='Processed Data')

    for i, row in df_params.iterrows():
        mu = row["Center (cm⁻¹)"]
        offset = max(y) * (0.05 if i % 2 == 0 else 0.1)
        plt.axvline(mu, color='gray', linestyle='--', linewidth=1)
        plt.text(mu, offset, f"{mu:.1f}", ha='center', va='bottom',
                 fontsize=9, color='black', fontweight='bold')

    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.title("Fitted Peak Centers (Wavenumbers)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Console Summary ===
def print_peak_table(df_params):
    print("\n--- Fitted Peaks ---\n")
    for _, row in df_params.iterrows():
        print(f"Peak {int(row['Peak'])} ({row['Model']}):")
        print(f"  Center = {row['Center (cm⁻¹)']:.2f} cm⁻¹")
        print(f"  FWHM   = {row['FWHM (cm⁻¹)']:.2f} cm⁻¹")
        print(f"  Height = {row['Relative Intensity']:.3f}")
        print(f"  Area   = {row['Area']:.3f}")
        print("-" * 35)

# === Optional Text-Mode Summary ===
def plot_text_summary(df_params):
    text = "Fitted Peaks:\n\n"
    for _, row in df_params.iterrows():
        text += (
            f"Peak {int(row['Peak'])} ({row['Model']}):\n"
            f"  Center = {row['Center (cm⁻¹)']:.2f} cm⁻¹\n"
            f"  FWHM   = {row['FWHM (cm⁻¹)']:.2f} cm⁻¹\n"
            f"  Height = {row['Relative Intensity']:.3f}\n"
            f"  Area   = {row['Area']:.3f}\n\n"
        )
    plt.figure(figsize=(8, 6))
    plt.axis("off")
    wrapped = textwrap.fill(text, width=80)
    plt.text(0.01, 0.99, wrapped, fontsize=10, va='top', ha='left', family='monospace')
    plt.title("Peak Fit Summary", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# === Main Runner ===
def main():
    folder = choose_folder()
    if not folder:
        print("[!] No folder selected.")
        return

    df_proc, df_fit, df_params = load_csv_files(folder)

    print_peak_table(df_params)

    # Interactive toggles
    show_components = input("Show individual component peaks? [y/N] ").strip().lower() == 'y'
    show_text_plot = input("Show monospace summary plot? [y/N] ").strip().lower() == 'y'

    plot_fit(df_proc, df_fit, df_params, show_components=show_components)
    plot_peak_centers(df_proc, df_params)

    if show_text_plot:
        plot_text_summary(df_params)


if __name__ == "__main__":
    main()
