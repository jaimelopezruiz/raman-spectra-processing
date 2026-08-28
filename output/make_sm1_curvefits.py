"""SM.1 — supplementary curve-fit figure: the five irradiated conditions, each
with data + total fit + components and a residual panel underneath, annotated
with per-region R2/RMSE (reduced chi2 deliberately NOT shown: not emitted, and
meaningless under curve_fit's absolute_sigma=False error model).

Fit and residual are drawn only inside the fitted regions (outside them the
total-fit curve is zero-padding, not a model). Style follows
analysis_plotting.apply_pub_style / make_figures.py.

Run from the repo root:  .venv\\Scripts\\python.exe output\\make_sm1_curvefits.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.stdout.reconfigure(errors="replace")

# runnable as `python output/make_sm1_curvefits.py` from the repo root: sys.path[0]
# is output/, so the repo modules need adding explicitly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import preprocess
from curve_fitting import fit_peaks_regionwise
from config import load_sample_config, find_matching_config
from analysis_plotting import apply_pub_style, PUB_DPI

OUT = "output"
TOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

# panel order = survey order (least -> most retained damage), Ne last
SAMPLES = [
    ("Si 0.25 dpa / 750 °C", "input/3 Si 750 0.25.csv"),
    ("Si 2.5 dpa / 750 °C", "input/4 Si 750 2.5.csv"),
    ("Si 0.25 dpa / 300 °C", "input/2 Si 300 0.25.csv"),
    ("Si 2.5 dpa / 300 °C", "input/1 Si 300 2.5.csv"),
    ("Ne 2.5 dpa / 300 °C", "input/0 Ne 300 2.5.csv"),
]

fig = plt.figure(figsize=(8.0, 21.0), dpi=PUB_DPI)
# nested gridspecs: tight fit/residual pairing inside each block, generous
# spacing BETWEEN condition blocks so titles never collide with the plot above
outer = GridSpec(len(SAMPLES), 1, figure=fig, hspace=0.42)

for i, (label, path) in enumerate(SAMPLES):
    cfg = load_sample_config(find_matching_config(path))
    pre = cfg["preprocessing"]
    x, y = preprocess(path, 170, 2000,
                      imodpoly_order=pre.get("baseline_order", 5),
                      imodpoly_tol=1e-3, imodpoly_max_iter=100,
                      normalisation=pre.get("normalisation", "vector-0to1"),
                      plot=False, save_path=None, convert_wavelength_to_shift=False)
    y_fit, fitted_peaks, peak_params, fit_stats, _ = fit_peaks_regionwise(
        x, y, cfg["regions"],
        center_tolerance=cfg["fitting"].get("center_tolerance", 100))

    inner = outer[i].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(inner[0])
    ax_res = fig.add_subplot(inner[1], sharex=ax)

    ax.plot(x, y, color="black", lw=1.0, label="processed data")
    first_fit = True
    for st in fit_stats:
        start, end = (float(v) for v in st["region"].split("-"))
        m = (x >= start) & (x <= end)
        ax.plot(x[m], y_fit[m], "--", color=TOL[1], lw=1.1,
                label="total fit" if first_fit else None)
        res = y[m] - y_fit[m]
        ax_res.plot(x[m], res, color="black", lw=0.7)
        first_fit = False
    for _, y_peak in fitted_peaks:
        ax.plot(x, np.where(y_peak > 1e-4, y_peak, np.nan), ":", lw=0.7,
                color=TOL[0], alpha=0.8)
    ax.plot([], [], ":", lw=0.7, color=TOL[0], label="components")

    stats_txt = "\n".join(
        f"{st['region']} cm$^{{-1}}$:  R$^2$ = {st['R2']:.3f},  RMSE = {st['RMSE']:.4f}"
        for st in fit_stats)
    ax.annotate(stats_txt, xy=(0.985, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6))
    ax.annotate(f"({chr(97 + i)})", xy=(0.012, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=13, weight="bold")

    apply_pub_style(ax, xlabel="", ylabel="Intensity (a.u.)")
    ax.set_title(label, fontsize=11, weight="bold", pad=4)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    plt.setp(ax.get_xticklabels(), visible=False)

    ax_res.axhline(0, color=TOL[1], ls="--", lw=0.7)
    ax_res.set_ylabel("Residual", fontsize=9)
    ax_res.tick_params(axis="both", labelsize=9, direction="out", length=4)
    ax_res.grid(True, ls="--", lw=0.4, alpha=0.3)
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)
    ax_res.set_xlabel("Raman shift (cm$^{-1}$)" if i == len(SAMPLES) - 1 else "",
                      fontsize=12)

out_path = os.path.join(OUT, "fig_SM1_curvefits.png")
fig.savefig(out_path, bbox_inches="tight", dpi=PUB_DPI)
plt.close(fig)
print(f"[OK] {out_path}")
