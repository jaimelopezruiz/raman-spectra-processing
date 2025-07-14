# === Raman Spectrum Analysis Pipeline ===
import os
import argparse
import tkinter as tk
from tkinter import filedialog

from preprocessing import preprocess
from curve_fitting import fit_peaks
from analysis_plotting import plot_and_report

# === File Input Handling ===
def choose_file_dialog():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Raman CSV File",
        filetypes=[("CSV files", "*.csv")]
    )

def get_input_file():
    parser = argparse.ArgumentParser(description="Raman Curve Fitting Pipeline")
    parser.add_argument("--input", help="Path to input CSV file")
    args = parser.parse_args()
    return args.input if args.input else choose_file_dialog()

# === Main Execution ===
def main():
    input_file = get_input_file()
    if not input_file or not os.path.isfile(input_file):
        print("[!] Invalid file selected.")
        return

    filename = os.path.splitext(os.path.basename(input_file))[0]
    print(f"[✓] Selected file: {filename}.csv")

    # Output paths
    os.makedirs("output", exist_ok=True)
    save_processed = f"output/{filename}_processed.csv"
    save_fitted_curve = f"output/{filename}_fitted_curve.csv"
    save_fitted_params = f"output/{filename}_fitted_params.xlsx"  # now xlsx

    # === Preprocessing Settings ===
    crop_min = 150
    crop_max = 2000
    sg_window = 11
    sg_polyorder = 3
    imodpoly_order = 8
    imodpoly_tol = 1e-3
    imodpoly_max_iter = 100
    normalisation = "vector-0to1"  # Options: vector, max, min-max, vector-0to1

    # === Peak Definitions ===
    peak_groups = [
        {"model": "gaussian", "center": 520, "window": 60},
        {"model": ["gaussian", "gaussian", "gaussian"], "center": [770, 870, 900], "window": 150},
        {"model": "gaussian", "center": 1400, "window": 60},
        {"model": "gaussian", "center": 1750, "window": 60},
    ]

    # === Run Pipeline ===
    print("[1] Preprocessing...")
    x_proc, y_proc = preprocess(
        input_path=input_file,
        crop_min=crop_min,
        crop_max=crop_max,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
        imodpoly_order=imodpoly_order,
        imodpoly_tol=imodpoly_tol,
        imodpoly_max_iter=imodpoly_max_iter,
        normalisation=normalisation,
        plot=True,
        save_path=save_processed
    )

    print("[2] Curve fitting...")
    y_fit_total, fitted_peaks, peak_params = fit_peaks(
        x_full=x_proc,
        y_full=y_proc,
        peak_groups=peak_groups,
        bounds=None,         # will be auto-generated
        auto_bounds=True     # enforce ±100 cm⁻¹ bounds
    )

    print("[3] Plotting & analysis...")
    plot_and_report(
        x=x_proc,
        y=y_proc,
        y_fit_total=y_fit_total,
        fitted_peaks=fitted_peaks,
        peak_params=peak_params,
        annotate=True,
        stagger_labels=True,
        font_size=9,
        label_offset=0.05,
        show_components=True,
        save_curve_path=save_fitted_curve,
        save_params_path=save_fitted_params,
        show=True
    )

    print(f"\n[✓] Analysis complete for: {filename}.csv")

if __name__ == "__main__":
    main()
