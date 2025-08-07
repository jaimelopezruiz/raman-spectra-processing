# === Raman Spectrum Analysis Pipeline (Region-Based Fitting) ===
import os
import argparse
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import preprocess
from curve_fitting import fit_peaks_regionwise
from analysis_plotting import plot_and_report

# === Region & Cropping Definitions ===
cmin = 170
cmax = 1800

# Format: (start, end, [ (model, amp, center, width), ... ])
REGIONS = [(1000, 2100, [("gauss", 0.3, 1080,5), ("lorentz", 2, 1415, 50)])]

# === File Input Handling ===
def choose_file_dialog():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilenames(  # ← enables multi-select
        title="Select Raman CSV File(s)",
        filetypes=[("CSV files", "*.csv")]
    )

def get_input_file():
    parser = argparse.ArgumentParser(description="Raman Curve Fitting Pipeline (Region-Based)")
    parser.add_argument("--input", help="Path to input CSV file")
    args = parser.parse_args()
    return args.input if args.input else choose_file_dialog()

# === Overlaying Multiple Spectra ===
def overlay_multiple_spectra(file_paths, crop_min = cmin, crop_max = cmax, scale_unirradiated = True):

    plt.figure(figsize=(12, 6))
    offset_step = 1.2

    for i, file in enumerate(file_paths):
        folder = os.path.basename(os.path.dirname(file))
        name = os.path.splitext(os.path.basename(file))[0]
        label = f"{folder} {name}"

        # Use your existing preprocessing pipeline
        x, y = preprocess(
            input_path=file,
            crop_min=crop_min,
            crop_max=crop_max,
            sg_window=11,
            sg_polyorder=10,
            imodpoly_order=2,
            imodpoly_tol=1e-3,
            imodpoly_max_iter=100,
            normalisation="vector-0to1",
            plot=False,
            save_path=None,
            alex_data=False
        )

        if scale_unirradiated and "Unirradiated" in folder:
            y *= 0.5
            label += " (scaled × 0.5)"

        y_offset = y + i * (np.max(y) - np.min(y)) * offset_step
        plt.plot(x, y_offset, label=label, linewidth=1.5)

    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Offset Intensity (a.u.)")
    plt.title("Overlay of Preprocessed Raman Spectra")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# === Main Execution ===
def main():
    input_files = get_input_file()

    if not input_files:
        print("[!] No file(s) selected.")
        return

    # If multiple files selected, plot overlay
    if isinstance(input_files, (list, tuple)) and len(input_files) > 1:
        overlay_multiple_spectra(input_files)
        return

    # === Single-file full pipeline ===
    input_file = input_files[0] if isinstance(input_files, (list, tuple)) else input_files
    if not os.path.isfile(input_file):
        print("[!] Invalid file selected.")
        return

    filename = os.path.splitext(os.path.basename(input_file))[0]
    print(f"[✓] Selected file: {filename}.csv")

    os.makedirs("output", exist_ok=True)

    x, y = preprocess(
        input_file,
        crop_min=cmin,
        crop_max=cmax,
        sg_window=11,
        sg_polyorder=10,
        imodpoly_order=2,
        imodpoly_tol=1e-3,
        imodpoly_max_iter=100,
        normalisation="vector-0to1",
        plot=True,
        save_path=f"output/{filename}_processed.csv",
        alex_data=False
    )

    CENTER_SHIFT_LIMIT = 30

    y_fit_total, fitted_peaks, peak_params = fit_peaks_regionwise(x, y, REGIONS, center_tolerance=CENTER_SHIFT_LIMIT)

    plot_and_report(
        x, y,
        y_fit_total, fitted_peaks, peak_params,
        annotate=False,
        stagger_labels=True,
        font_size=9,
        label_offset=0.05,
        show_components=True,
        show_text_plot=True,
        save_curve_path=f"output/{filename}_fitted_curve.csv",
        save_params_path=f"output/{filename}_peak_parameters.csv",
        show=True
    )

if __name__ == "__main__":
    main()