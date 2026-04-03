# === Raman Spectrum Analysis Pipeline (Region-Based Fitting) ===
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np

from tkinter import messagebox
from preprocessing import preprocess, _read_spectrum_table
from curve_fitting import fit_peaks_regionwise
from analysis_plotting import plot_and_report, apply_pub_style, PUB_FIGSIZE, PUB_DPI

# === Figure and display config ===
FIG_WIDTH = 6      # inches
FIG_HEIGHT = 4.5   # inches
LEGEND_OUTSIDE = True

# === Input-axis conversion ===
# Enable this only for datasets whose x-axis is wavelength and needs converting to Raman shift.
CONVERT_WAVELENGTH_TO_SHIFT = False

# === Region, cropping, and baseline definitions ===
CROP_MIN = 170
CROP_MAX = 2000
BASELINE_ORDER = 5

# Format: (start, end, [ (model, amp, center, width), ... ])
REGIONS = [
    (450, 550, [("lorentz", 0.2, 520, 2)]),
    (560, 580, [("lorentz", 0.05, 570, 1)]),
    (650, 1160, [("lorentz", 0.6, 760, 2), ("lorentz", 1, 790, 2),("lorentz", 1, 795, 2),("lorentz", 0.3, 960, 2)]),
    (1350, 1750, [("lorentz",0.1,1500,2),("lorentz",0.1,1700,2)])
]

# === File Input Handling ===
def choose_file_dialog():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilenames(
        title="Select Raman CSV File(s)",
        filetypes=[("Spectrum files", "*.csv *.txt"), ("CSV files", "*.csv"), ("Text files", "*.txt")]
    )
# === Overlaying Multiple Spectra ===       # Set normalisation to none to compare real relative intensities
def overlay_multiple_spectra(
    file_paths,
    crop_min=CROP_MIN, crop_max=CROP_MAX,
    scale_unirradiated=False,     # irrelevant with this method, but keep arg for compatibility
    figsize=None, run_preprocessing=True, show_legend=True
):
    PAUL_TOL_MUTED = [
        "#CC6677", "#332288", "#DDCC77", "#117733",
        "#88CCEE", "#882255", "#44AA99", "#999933", "#AA4499",
    ]

    fig, ax = plt.subplots(figsize=figsize or PUB_FIGSIZE, dpi=PUB_DPI)
    ax.set_prop_cycle(color=PAUL_TOL_MUTED)

    # 1) Load full + cropped (NO normalisation) to compute ratios
    spectra = []      # [(label, x_crop, y_crop, ratio, cropped_max, y_crop_min), ...]
    ratios = []

    for file in file_paths:
        folder = os.path.basename(os.path.dirname(file))
        name = os.path.splitext(os.path.basename(file))[0]
        label = f"{folder} {name}"

        if run_preprocessing:
            x_full, y_full = preprocess(
                input_path=file,
                crop_min=0,
                crop_max=4000,
                sg_window=11,
                sg_polyorder=10,
                imodpoly_order=BASELINE_ORDER,
                imodpoly_tol=1e-3,
                imodpoly_max_iter=100,
                normalisation="vector-0to1",
                plot=False,
                save_path=None,
                convert_wavelength_to_shift=CONVERT_WAVELENGTH_TO_SHIFT
            )
            full_max = np.max(y_full)

            x_crop, y_crop = preprocess(
                input_path=file,
                crop_min=crop_min,
                crop_max=crop_max,
                sg_window=11,
                sg_polyorder=10,
                imodpoly_order=BASELINE_ORDER,
                imodpoly_tol=1e-3,
                imodpoly_max_iter=100,
                normalisation="vector-0to1",
                plot=False,
                save_path=None,
                convert_wavelength_to_shift=CONVERT_WAVELENGTH_TO_SHIFT
            )
        else:
            x_full, y_full = _read_spectrum_table(file)
            full_max = np.max(y_full)
            mask = (x_full >= crop_min) & (x_full <= crop_max)
            x_crop, y_crop = x_full[mask], y_full[mask]


        cropped_max = np.max(y_crop)
        ratio = cropped_max / full_max if full_max > 0 else 0.0
        ratios.append(ratio)
        spectra.append((label, x_crop, y_crop, ratio, cropped_max, np.min(y_crop)))

    # 2) Find the max ratio -> that trace will hit 1.0 within its band
    max_ratio = max(ratios) if ratios else 1.0
    if max_ratio == 0:
        max_ratio = 1.0  # avoid divide-by-zero if all-zero (degenerate)

    # 3) Plot each: shift to non-negative, normalise by its own cropped max (shape kept),
    #    then scale by (ratio/max_ratio) so tallest -> 1. Finally offset by integer i.
    legend_handles, legend_labels = [], []
    for i, (label, x_crop, y_crop, ratio, cropped_max, y_min) in enumerate(spectra):
        # Make sure the cropped signal sits in [0, 1] *before* applying relative ratio scaling
        y_nonneg = y_crop - y_min
        # guard for flat spectra
        denom = (np.max(y_nonneg) if np.max(y_nonneg) > 0 else 1.0)
        y_unit = y_nonneg / denom                     # 0..1, preserves shape
        rel_scale = ratio / max_ratio                 # <= 1
        offset_index = len(spectra) - 1 - i
        y_band = offset_index + y_unit * rel_scale     # first file at top, last at bottom

        display_label = label if len(label) <= 40 else label[:37] + "..."
        h, = ax.plot(x_crop, y_band, linewidth=1.2, label=display_label)
        legend_handles.append(h)
        legend_labels.append(display_label)

    # 4) Styling + legend outside
    _ = apply_pub_style(
        ax,
        title="Overlay of Preprocessed Raman Spectra (Relative Cropped Height)",
        xlabel="Raman shift (cm$^{-1}$)",
        ylabel="Offset Intensity (a.u.)"
    )

    if show_legend:
        ncol = 1 if len(legend_handles) <= 14 else 2
        ax.legend(
            legend_handles, legend_labels,
            loc='upper left', bbox_to_anchor=(1.02, 1),
            borderaxespad=0, fontsize=7, frameon=False, ncol=ncol
        )
        fig.subplots_adjust(right=0.72)

    # Lock the y-limits to full integer bands
    top_band = len(spectra)
    ax.set_ylim(-0.05, top_band + 0.05)

    plt.tight_layout()
    plt.show()

def ask_preprocess():
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(
        "Preprocessing",
        "Run files through the preprocessing pipeline?\n\n"
        "Yes = denoise, baseline correction, normalise\n"
        "No  = plot/fit raw data as-is"
    )
    root.destroy()
    return result

def ask_show_legend():
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("Legend", "Show legend on the plot?")
    root.destroy()
    return result

# === Main Execution ===
def main():
    input_files = choose_file_dialog()

    if not input_files:
        print("[!] No file(s) selected.")
        return

    run_preprocessing = ask_preprocess()
    show_legend = ask_show_legend()
    figsize = (FIG_WIDTH, FIG_HEIGHT)

    if isinstance(input_files, (list, tuple)) and len(input_files) > 1:
        overlay_multiple_spectra(input_files, figsize=figsize, run_preprocessing=run_preprocessing, show_legend=show_legend)
        return

    input_file = input_files[0]
    if not os.path.isfile(input_file):
        print("[!] Invalid file selected.")
        return

    filename = os.path.splitext(os.path.basename(input_file))[0]
    print(f"[OK] Selected file: {filename}")

    os.makedirs("output", exist_ok=True)

    if run_preprocessing:
        x, y = preprocess(
            input_file,
            crop_min=CROP_MIN,
            crop_max=CROP_MAX,
            sg_window=11,
            sg_polyorder=10,
            imodpoly_order=BASELINE_ORDER,
            imodpoly_tol=1e-3,
            imodpoly_max_iter=100,
            normalisation="none",
            plot=True,
            save_path=f"output/{filename}_processed.csv",
            convert_wavelength_to_shift=CONVERT_WAVELENGTH_TO_SHIFT
        )
    else:
        x, y = _read_spectrum_table(input_file)

    CENTER_SHIFT_LIMIT = 100
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
        show=True,
        figsize=figsize,
        show_legend=show_legend,
        legend_outside=LEGEND_OUTSIDE,
        legend_ncol=1,
        legend_fontsize=6
    )



if __name__ == "__main__":
    main()
