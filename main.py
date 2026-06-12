# === Raman Spectrum Analysis Pipeline (Region-Based Fitting) ===
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Console summaries use σ/cm⁻¹; don't crash on non-UTF-8 consoles (e.g. cp1252)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from preprocessing import preprocess, _read_spectrum_table
from curve_fitting import fit_peaks_regionwise
from analysis_plotting import plot_and_report, apply_pub_style, PUB_FIGSIZE, PUB_DPI
from config import load_sample_config, find_matching_config

# === Figure and display config ===
FIG_WIDTH = 6      # inches
FIG_HEIGHT = 4.5   # inches
LEGEND_OUTSIDE = True

# === Input-axis conversion ===
# Enable this only for datasets whose x-axis is wavelength and needs converting to Raman shift.
CONVERT_WAVELENGTH_TO_SHIFT = False

# Per-sample region/peak configs live in params/configs/*.yaml; each peak
# carries its literature mode assignment (see README). The default config is
# the unirradiated reference sample.
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "params", "configs", "unirradiated.yaml")

# Fallback preprocessing/fitting settings, overridden by the config file.
CROP_MIN = 170
CROP_MAX = 2000
BASELINE_ORDER = 5
CENTER_SHIFT_LIMIT = 100


# === File Input Handling ===
def choose_file_dialog():
    import tkinter as tk
    from tkinter import filedialog
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
    figsize=None, run_preprocessing=True, show_legend=True, show=True,
    baseline_order=BASELINE_ORDER, save_fig_path=None
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
                imodpoly_order=baseline_order,
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
                imodpoly_order=baseline_order,
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
    if save_fig_path:
        fig.savefig(save_fig_path, bbox_inches="tight")
        print(f"[OK] Overlay figure saved to: {save_fig_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def ask_yes_no(title, message):
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(title, message)
    root.destroy()
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Physics-informed Raman spectrum preprocessing and region-wise peak fitting. "
                    "Run with no arguments for the interactive (file-dialog) mode."
    )
    parser.add_argument("inputs", nargs="*",
                        help="Input spectrum file(s) (.csv/.txt). One file -> fit; several -> overlay plot.")
    parser.add_argument("--config", default=None,
                        help="Per-sample fitting config YAML. If omitted, the config is "
                             "auto-detected from the input filename via the 'match' patterns "
                             f"in params/configs/*.yaml (fallback: {DEFAULT_CONFIG})")
    parser.add_argument("--output-dir", default="output", help="Directory for output CSVs (default: output)")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip baseline correction/normalisation; fit or plot raw data as-is")
    parser.add_argument("--no-legend", action="store_true", help="Hide the plot legend")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open plot windows (for scripted/CI runs)")
    parser.add_argument("--save-fig", default=None, metavar="PATH",
                        help="Save the main figure to PATH instead of/as well as showing it")
    return parser.parse_args()


# === Main Execution ===
def main():
    args = parse_args()
    interactive = not args.inputs

    if interactive:
        input_files = choose_file_dialog()
        if not input_files:
            print("[!] No file(s) selected.")
            return
        run_preprocessing = ask_yes_no(
            "Preprocessing",
            "Run files through the preprocessing pipeline?\n\n"
            "Yes = baseline correction, normalise\n"
            "No  = plot/fit raw data as-is"
        )
        show_legend = ask_yes_no("Legend", "Show legend on the plot?")
    else:
        input_files = args.inputs
        run_preprocessing = not args.no_preprocess
        show_legend = not args.no_legend

    show = not args.no_show
    figsize = (FIG_WIDTH, FIG_HEIGHT)

    # Resolve the fitting config: explicit --config wins; otherwise try to
    # auto-detect from the input filename via the configs' 'match' patterns.
    config_path = args.config
    if config_path is None:
        detected = find_matching_config(input_files[0]) if len(input_files) == 1 else None
        if detected:
            config_path = detected
            print(f"[OK] Auto-detected config from filename: {os.path.basename(detected)}")
        else:
            config_path = DEFAULT_CONFIG
            if len(input_files) == 1:
                print(f"[!] No config matched the input filename; using the default "
                      f"({os.path.basename(DEFAULT_CONFIG)}). Pass --config to override.")

    cfg = load_sample_config(config_path)
    pre_cfg = cfg["preprocessing"]
    crop_min = pre_cfg.get("crop_min", CROP_MIN)
    crop_max = pre_cfg.get("crop_max", CROP_MAX)
    baseline_order = pre_cfg.get("baseline_order", BASELINE_ORDER)
    normalisation = pre_cfg.get("normalisation", "none")
    center_tolerance = cfg["fitting"].get("center_tolerance", CENTER_SHIFT_LIMIT)
    if cfg["meta"].get("sample"):
        print(f"[OK] Using fitting config: {cfg['meta']['sample']} ({config_path})")

    if isinstance(input_files, (list, tuple)) and len(input_files) > 1:
        overlay_multiple_spectra(
            input_files, crop_min=crop_min, crop_max=crop_max,
            figsize=figsize, run_preprocessing=run_preprocessing,
            show_legend=show_legend, show=show,
            baseline_order=baseline_order, save_fig_path=args.save_fig
        )
        return

    input_file = input_files[0]
    if not os.path.isfile(input_file):
        print("[!] Invalid file selected.")
        return

    filename = os.path.splitext(os.path.basename(input_file))[0]
    print(f"[OK] Selected file: {filename}")

    os.makedirs(args.output_dir, exist_ok=True)

    if run_preprocessing:
        # Note: no denoising — fitting is performed on unsmoothed,
        # baseline-corrected data (see preprocessing.preprocess docstring).
        x, y = preprocess(
            input_file,
            crop_min=crop_min,
            crop_max=crop_max,
            imodpoly_order=baseline_order,
            imodpoly_tol=1e-3,
            imodpoly_max_iter=100,
            normalisation=normalisation,
            plot=show,
            save_path=os.path.join(args.output_dir, f"{filename}_processed.csv"),
            convert_wavelength_to_shift=CONVERT_WAVELENGTH_TO_SHIFT
        )
    else:
        x, y = _read_spectrum_table(input_file)

    y_fit_total, fitted_peaks, peak_params, fit_stats = fit_peaks_regionwise(
        x, y, cfg["regions"], center_tolerance=center_tolerance
    )

    plot_and_report(
        x, y,
        y_fit_total, fitted_peaks, peak_params,
        fit_stats=fit_stats,
        annotate=False,
        stagger_labels=True,
        font_size=9,
        label_offset=0.05,
        show_components=True,
        show_text_plot=True,
        save_curve_path=os.path.join(args.output_dir, f"{filename}_fitted_curve.csv"),
        save_params_path=os.path.join(args.output_dir, f"{filename}_peak_parameters.csv"),
        save_stats_path=os.path.join(args.output_dir, f"{filename}_fit_statistics.csv"),
        save_fig_path=args.save_fig,
        show=show,
        figsize=figsize,
        show_legend=show_legend,
        legend_outside=LEGEND_OUTSIDE,
        legend_ncol=1,
        legend_fontsize=6
    )



if __name__ == "__main__":
    main()
