# === Raman Spectrum Analysis Pipeline (Region-Based Fitting) ===
import os
import argparse
import tkinter as tk
from tkinter import filedialog

from preprocessing import preprocess
from curve_fitting import fit_peaks_regionwise
from analysis_plotting import plot_and_report

# === Region Definitions ===
# Format: (start, end, [ (model, amp, center, width), ... ])
REGIONS = [
    (400, 650, [("pvoigt", 0.05, 420, 5),("pvoigt", 0.2, 550, 5),("pvoigt", 0.1, 590, 5)]),
    (650, 1000, [("pvoigt", 0.05, 720, 5),("lorentz", 0.6, 770, 5), ("lorentz", 1, 790, 5), ("lorentz",0.3, 860, 2),("lorentz", 0.3, 890, 2),("lorentz", 0.3, 960, 5),("lorentz", 0.6, 975, 5),]),
    (1300, 1700, [("lorentz", 0.4, 1400, 5), ("lorentz",0.4,1600,5)])
]

# === File Input Handling ===
def choose_file_dialog():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Raman CSV File",
        filetypes=[("CSV files", "*.csv")]
    )

def get_input_file():
    parser = argparse.ArgumentParser(description="Raman Curve Fitting Pipeline (Region-Based)")
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

    # Output directory
    os.makedirs("output", exist_ok=True)

    # === Step 1: Preprocessing ===
    x, y = preprocess(
        input_file,
        crop_min=150,
        crop_max=2000,
        sg_window=31,
        sg_polyorder=3,
        imodpoly_order=8,
        imodpoly_tol=1e-3,
        imodpoly_max_iter=100,
        normalisation="vector-0to1",
        plot=True,
        save_path=f"output/{filename}_processed.csv",
        alex_data = True
    )

    # === Step 2: Region-Based Curve Fitting ===
    # Allow peaks to shift ± this many cm⁻¹ from initial guess

    CENTER_SHIFT_LIMIT = 35

    y_fit_total, fitted_peaks, peak_params = fit_peaks_regionwise(x, y, REGIONS, center_tolerance=CENTER_SHIFT_LIMIT)


    # === Step 3: Plot and Report ===
    plot_and_report(
        x, y,
        y_fit_total, fitted_peaks, peak_params,
        annotate=True,
        stagger_labels=True,
        font_size=9,
        label_offset=0.05,
        show_components=True,
        show_text_plot=True,
        save_curve_path=f"output/{filename}_fitted_curve.csv",
        save_params_path=f"output/{filename}_peak_parameters.csv"
    )

if __name__ == "__main__":
    main()
