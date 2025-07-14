# === Raman Spectrum Analysis Pipeline ===
import os
import argparse
import tkinter as tk
from tkinter import filedialog

from preprocessing import preprocess
from curve_fitting import fit_peaks
from analysis_plotting import plot_and_report

# === Peak Definitions ===
# Format: (model, amp_guess, center_guess, width_guess)
PEAKS = [
    ("lorentz", 0.5, 520, 5),
    ("gauss", 0.7, 560, 10),
    ("pvoigt", 0.40, 790, 10),
    ("pvoigt", 0.40, 880, 10),
    ("pvoigt", 0.60, 930, 10),
    ("gauss", 0.2, 1400, 10),
    ("gauss", 0.1, 1600, 10),
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

    # Output directory
    os.makedirs("output", exist_ok=True)

    # === Step 1: Preprocessing ===
    x, y = preprocess(
        input_file,
        crop_min=150,
        crop_max=2000,
        sg_window=11,
        sg_polyorder=3,
        imodpoly_order=8,
        imodpoly_tol=1e-3,
        imodpoly_max_iter=100,
        normalisation="vector-0to1",
        plot=True,
        save_path=f"output/{filename}_processed.csv"
    )

    # === Step 2: Curve Fitting ===
    y_fit_total, fitted_peaks, peak_params = fit_peaks(x, y, PEAKS)

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
