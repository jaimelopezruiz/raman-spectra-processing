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
    (170, 1600, [("voigt", 0.1, 186, 10), ("voigt", 0.1, 266, 10), ("voigt", 0.1, 435, 5), 
                 ("voigt", 0.2, 500, 5), ("voigt", 0.2, 535, 5), ("voigt", 0.1, 580, 1), 
                 ("voigt", 0.1, 660, 1), ("lorentz", 0.4, 767, 1), ("lorentz", 0.4, 790, 1), ("lorentz", 0.4, 795, 1), ("gauss", 0.5, 770, 5), 
                 ("voigt", 0.3, 849, 2), ("voigt", 0.3, 940, 2), ("voigt", 0.3, 923, 2), ("gauss", 0.3, 870, 10), ("voigt", 0.3, 870, 2), 
                 ("bwf", 0.13, 1080, 10, 1), ("lorentz", 0.1, 1415, 2), ("voigt", 0.01, 1520, 2)]),
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
        crop_min=170,
        crop_max=4000,
        sg_window=11,
        sg_polyorder=10,
        imodpoly_order=5,
        imodpoly_tol=1e-3,
        imodpoly_max_iter=100,
        normalisation="vector-0to1",
        plot=True,
        save_path=f"output/{filename}_processed.csv",
        alex_data = False
    )

    # === Step 2: Region-Based Curve Fitting ===
    # Allow peaks to shift ± this many cm⁻¹ from initial guess

    CENTER_SHIFT_LIMIT = 30

    y_fit_total, fitted_peaks, peak_params = fit_peaks_regionwise(x, y, REGIONS, center_tolerance=CENTER_SHIFT_LIMIT)


    # === Step 3: Plot and Report ===
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
        show = True
    )

if __name__ == "__main__":
    main()
