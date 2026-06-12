# Multi Spectra Comparison
#
# Stacked overlay of several spectra with temperature labels extracted from
# the filenames (intended for the in-situ annealing series).
# Usage:
#   python multi_spectra_comparison.py --input FILE_OR_DIR [...] [--preprocess]
#   python multi_spectra_comparison.py            (interactive file dialog)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox


# Paul Tol's muted colorblind-friendly palette
PAUL_TOL_MUTED = [
    "#CC6677", "#332288", "#DDCC77", "#117733",
    "#88CCEE", "#882255", "#44AA99", "#999933", "#AA4499",
]


def choose_file_dialog(multiple=True):
    root = tk.Tk()
    root.withdraw()
    if multiple:
        return filedialog.askopenfilenames(
            title="Select Raman CSV Files",
            filetypes=[("Spectrum files", "*.csv *.txt"), ("CSV files", "*.csv"), ("Text files", "*.txt")]
        )
    else:
        return [filedialog.askopenfilename(
            title="Select a Raman CSV File",
            filetypes=[("Spectrum files", "*.csv *.txt"), ("CSV files", "*.csv"), ("Text files", "*.txt")]
        )]


def parse_args():
    parser = argparse.ArgumentParser(description="Raman Spectra Plotter")
    parser.add_argument("--input", nargs="*", help="Path(s) to CSV files or directories")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run files through the preprocessing pipeline before plotting "
                             "(skips the yes/no dialog)")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Plot raw data as-is (skips the yes/no dialog)")
    return parser.parse_args()


def get_input_files(args):
    if args.input:
        all_files = []
        for path in args.input:
            if os.path.isdir(path):
                files_in_dir = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.lower().endswith((".csv", ".txt"))
                ]
                all_files.extend(files_in_dir)
            else:
                all_files.append(path)
        return all_files
    else:
        return list(choose_file_dialog(multiple=True))


def ask_preprocess():
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(
        "Preprocessing",
        "Run files through the preprocessing pipeline before plotting?\n\n"
        "Yes = baseline correction, normalise\n"
        "No  = plot raw data as-is"
    )
    root.destroy()
    return result


def load_processed_spectrum(filepath):
    # Auto-detect separator
    data = pd.read_csv(filepath, sep=None, engine="python",
                       header=None, skiprows=0)
    # If first row is non-numeric, treat it as a header and drop it
    try:
        float(str(data.iloc[0, 0]).strip())
    except ValueError:
        data = data.iloc[1:].reset_index(drop=True)
    x = data.iloc[:, 0].astype(float).values
    y = data.iloc[:, 1].astype(float).values
    return x, y


def extract_temperature_label(filename):
    name = os.path.basename(filename)

    match = re.search(r"\b(\d+)[Cc]\b", name)
    if "RT" in name.upper():
        return "RT"
    elif match:
        temp = match.group(1)
        return f"{temp} °C"  # Unicode degree symbol
    else:
        return ""  # fallback if not found


def main():
    args = parse_args()
    file_paths = get_input_files(args)
    if not file_paths:
        print("[!] No file(s) selected.")
        return

    if args.preprocess:
        run_preprocessing = True
    elif args.no_preprocess:
        run_preprocessing = False
    else:
        run_preprocessing = ask_preprocess()

    if run_preprocessing:
        from preprocessing import preprocess

    plt.figure(figsize=(12, 6))
    plt.gca().set_prop_cycle(color=PAUL_TOL_MUTED)

    for i, file in enumerate(file_paths):
        folder = os.path.basename(os.path.dirname(file))
        name = os.path.splitext(os.path.basename(file))[0]
        label = f"{folder} {name}"

        if run_preprocessing:
            x, y = preprocess(
                file,
                crop_min=170, crop_max=2000,
                imodpoly_order=5, imodpoly_tol=1e-3, imodpoly_max_iter=100,
                normalisation="vector-0to1",
                plot=False, save_path=None,
                convert_wavelength_to_shift=False
            )
        else:
            x, y = load_processed_spectrum(file)

        # Shift to zero-minimum and scale to [0, 1]
        y = y - y.min()
        if y.max() > 0:
            y = y / y.max()

        # Stack at whole-number offsets: bottom spectrum at 0, next at 1, etc.
        offset_index = (len(file_paths) - i - 1)
        y_offset = y + offset_index

        # Plot the spectrum
        plt.plot(x, y_offset, label=label, linewidth=1.5)

        # ---- Labeling: Position relative to the last valid point near 3800 cm⁻¹ ----
        anchor_x = 3800
        anchor_index = np.argmin(np.abs(x - anchor_x))  # closest index to the anchor
        if anchor_index >= len(y_offset):
            anchor_index = -1  # fallback

        y_anchor = y_offset[anchor_index]
        y_pos = y_anchor + 0.05  # small lift above the line
        x_pos = x[anchor_index]

        # Extract temperature string for label
        temperature = extract_temperature_label(file)
        plt.text(x_pos, y_pos, temperature, ha='left', fontsize=9, fontweight='bold')

    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Offset Intensity (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
