# === Raman Spectrum Analysis Pipeline ===
from preprocessing import preprocess
from curve_fitting import fit_peaks
from analysis_plotting import plot_and_report

# === User Input ===
input_file = "input/Dark Grey.csv"           # Path to your raw CSV
save_processed = "output/processed.csv"          # Processed spectrum output (optional)
save_fitted_curve = "output/fitted_curve.csv"    # Fitted curve output
save_fitted_params = "output/fitted_params.csv"  # Fitted peak parameter table

# === Preprocessing Settings ===
crop_min = 150
crop_max = 2000
sg_window = 11
sg_polyorder = 3
imodpoly_order = 8
imodpoly_tol = 1e-3
imodpoly_max_iter = 100
normalisation = "vector"  # or "max"

# === Peak Fitting Definitions ===
peak_groups = [
    {"model": ["gaussian", "voigt"], "center": [770, 900], "window": 80},
    {"model": "gaussian", "center": 1350, "window": 60},
    {"model": "gaussian", "center": 1600, "window": 60},
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
    bounds=None  # Optional: add fit parameter bounds here
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

print("\n Pipeline complete.")
