# Multi Spectra Comparison (Already Processed Data)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import tkinter as tk
from tkinter import filedialog


def choose_file_dialog(multiple=True):
    root = tk.Tk()
    root.withdraw()
    if multiple:
        return filedialog.askopenfilenames(
            title="Select Raman CSV Files",
            filetypes=[("CSV files", "*.csv")]
        )
    else:
        return [filedialog.askopenfilename(
            title="Select a Raman CSV File",
            filetypes=[("CSV files", "*.csv")]
        )]

def get_input_files():
    parser = argparse.ArgumentParser(description="Raman Spectra Plotter")
    parser.add_argument("--input", nargs="*", help="Path(s) to CSV files or directories")
    args = parser.parse_args()

    if args.input:
        all_files = []
        for path in args.input:
            if os.path.isdir(path):
                files_in_dir = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.lower().endswith(".csv")
                ]
                all_files.extend(files_in_dir)
            else:
                all_files.append(path)
        return all_files
    else:
        return list(choose_file_dialog(multiple=True))

# Use this to populate your list of files
file_paths = get_input_files()


def load_processed_spectrum(filepath):
    data = pd.read_csv(filepath)  # uses headers automatically
    x = data["Raman Shift (cm-1)"].values
    y = data["Processed Intensity"].values
    return x, y



# === Plotting ===
plt.figure(figsize=(12, 6))
offset_step = 1.2  # spacing between each curve

import re

def extract_temperature_label(filename):
    name = os.path.basename(filename)

    match = re.search(r"\b(\d+)[Cc]\b", name)
    if "RT" in name.upper():
        return "RT"
    elif match:
        temp = match.group(1)
        return f"{temp} °C"  # Unicode degree symbol
    else:
        return ""  # fallback if not found


for i, file in enumerate(file_paths):
    folder = os.path.basename(os.path.dirname(file))
    name = os.path.splitext(os.path.basename(file))[0]
    label = f"{folder} {name}"

    x, y = load_processed_spectrum(file)

    # Apply vertical offset to each curve
    offset_index = (len(file_paths) - i - 1)
    y_offset = y + offset_index * offset_step

    # Plot the spectrum
    plt.plot(x, y_offset, label=label, linewidth=1.5)

    # ---- Labeling: Position relative to the last valid point near 3500 cm⁻¹ ----
    anchor_x = 3800
    anchor_index = np.argmin(np.abs(x - anchor_x))  # closest index to 3500 cm⁻¹
    if anchor_index >= len(y_offset):
        anchor_index = -1  # fallback

    y_anchor = y_offset[anchor_index]
    y_pos = y_anchor + 0.2  # small lift above the line
    x_pos = x[anchor_index]

    # Extract temperature string for label
    temperature = extract_temperature_label(file)
    plt.text(x_pos, y_pos, temperature, ha='left', fontsize=9, fontweight='bold')




plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Offset Intensity (a.u.)")
plt.title("Vertically Offset Overlay of Processed Raman Spectra")
#plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
