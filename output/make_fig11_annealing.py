"""Main-text Figure 9 (step-wise annealing overlays) + the full-series
supplementary versions.  Ticket C12; band labels removed in C21.

Panels, matching the manuscript caption's lettering:
  (a) Au-implanted single-crystal 6H-SiC        (Experiment 2)
  (b) Ne-implanted polycrystalline RB-SiC       (Experiment 3)
  (c) Si-implanted polycrystalline RB-SiC, 2.5 dpa / 300 C   (Experiment 3)
  (d) Si-implanted polycrystalline RB-SiC, 2.5 dpa / 750 C   (Experiment 3)

Style, from the C4 overlay template and the reviewer comments on the old figure:
  * stacked at integer offsets, LOW temperature at the BOTTOM, increasing upward,
    with the room-temperature references as the bottom (before) and top (after)
    rows — the same reading direction as Figure 8(a), which is what comment [68]
    ("reverse the order … as in Fig. 10") asks for. The caption has since been
    flipped to match ("temperature increases from bottom to top").
  * right-hand-side trace labels, no legend.
  * larger fonts throughout (comment [67]).
  * main-text panels trimmed to the temperatures the Section 3.2 text actually
    discusses (comment [70]); the full series go to the supplementary.

C21: THE VIBRATIONAL-BAND LABELS ARE GONE. The row of "Si-Si / Si-Si / Si-C TO /
Si-C LO / C-C" text that C12 drew along the top of each panel (the second half
of comment [68]) is removed, and that is the ONLY change from the C12 build —
same layout, fonts, colours, offsets, trace labels, panel headers and figure
sizes. The labels sat inside the axes, below `top + 0.95`, so the tight bounding
box is unaffected and every raster keeps its published pixel size. The head-room
above the top trace is deliberately left as it was rather than being tightened,
because closing it would move every trace and stop the rebuild being a pure
deletion.

RT-AFTER AVAILABILITY (comment [69], "3. in b): is there RT after?"):
  Au    yes (spectra 30-37, "RT after anneal")
  Ne    NO  — the series ends at the 1200 C hold (spectrum 037); the methods text
        for Experiment 3 already says so ("…at room temperature post-annealing
        without the furnace glass, except for the Ne-implanted specimen").
  Si 2.5/300  yes (spectra 057-058)
  Si 2.5/750  yes (spectra 028-030)
  The script asserts this, so a future data drop that adds a Ne RT-after spectrum
  makes the assertion fail rather than silently changing the figure.

Writes: output/fig_9_annealing_4panel.png       (main text)
        output/fig_annealing_overlay_au.png     (panel (a) alone)
        output/fig_annealing_overlay_si300.png  (panel (c) alone)
        output/fig_SM_annealing_overlay_{au,ne,si300,si750}_full.png
Run from the repo root:  .venv\\Scripts\\python.exe output\\make_fig11_annealing.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(errors="replace")

from preprocessing import preprocess
from analysis_plotting import apply_pub_style, PUB_DPI
from annealing_series import (SERIES, index_series, stage_order, stage_label,
                              usable_spectra)

OUT = "output"
# Paul Tol muted palette as elsewhere in the repo, but with its pale grey
# replaced by a dark neutral: these overlays cycle through up to 22 traces and
# #BBBBBB is too faint to read as a spectrum.
TOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#444444"]
CROP = (170.0, 2000.0)
TRACE_AMP = 0.9        # trace height per unit offset: 0.9 leaves a clear gap
LABEL_GAP = 25.0       # cm-1 from the trace end to its right-hand-side label

# Temperatures discussed in Section 3.2 for each specimen, so the main-text
# panels show the evidence for the sentences rather than every hold:
#   Au    plateau (100) / recovery onset (500) / the 700-900 transition / recovered (1200)
#   Ne    plateau (100) / extended 400 hold / the abrupt 800-900 recovery / the 1200 480 cm-1 feature
#   Si    plateau (100) / pre- and post-onset (700, 800) / chi minimum (1000) / chi tick-up (1200)
PANELS = [
    ("(a)", "au", [100, 500, 700, 900, 1200]),
    ("(b)", "ne", [100, 400, 800, 900, 1200]),
    ("(c)", "si300", [100, 700, 800, 1000, 1200]),
    ("(d)", "si750", [100, 700, 800, 900, 1100]),
]
RT_AFTER_EXPECTED = {"au": True, "ne": False, "si300": True, "si750": True}


def norm01(y):
    y = y - y.min()
    return y / y.max() if y.max() > 0 else y


def load(path, convert_wavelength):
    x, y = preprocess(path, CROP[0], CROP[1], imodpoly_order=5, imodpoly_tol=1e-3,
                      imodpoly_max_iter=100, normalisation="vector-0to1",
                      plot=False, save_path=None,
                      convert_wavelength_to_shift=convert_wavelength)
    return x, norm01(y)


def entries(key, temps=None, report=False):
    """[(label, x, y), ...] bottom-to-top for one series.

    One spectrum per stage: the first acquired that passes annealing_series'
    usability tests (full spectral coverage, and a SiC spot rather than the
    residual free-Si matrix). Anything dropped is printed once, when `report`.
    """
    spec = SERIES[key]
    stages, _ = index_series(key)
    have_after = "RT after" in stages
    assert have_after == RT_AFTER_EXPECTED[key], (
        f"{key}: RT-after present={have_after}, expected {RT_AFTER_EXPECTED[key]} "
        "— the data set changed, revisit the Figure 9 caption and comment [69]")
    out = []
    for stage in stage_order(stages):
        if temps and isinstance(stage, int) and stage not in temps:
            continue
        kept, rejected = usable_spectra(stages[stage], spec["convert_wavelength"], CROP)
        if report:
            # only the rejections that actually changed which spectrum is shown,
            # i.e. those acquired before the one selected
            cut = stages[stage].index(kept[0]) if kept else len(stages[stage])
            for p, why in rejected:
                if stages[stage].index(p) < cut:
                    print(f"  [dropped] {key} {stage_label(stage)}: "
                          f"{os.path.basename(p)[:44]} — {why}")
        if not kept:
            print(f"  [SKIP] {key} {stage_label(stage)}: no usable spectrum")
            continue
        x, y = load(kept[0], spec["convert_wavelength"])
        out.append((stage_label(stage), x, y))
    return out


def draw_overlay(ax, ent, trace_fs=11):
    """Stacked overlay with right-hand-side labels; returns the top offset."""
    for i, (label, x, y) in enumerate(ent):
        ax.plot(x, TRACE_AMP * y + i, lw=1.0, color=TOL[i % len(TOL)])
        ax.text(x.max() + LABEL_GAP, i + TRACE_AMP * y[-1], label, ha="left",
                va="center", fontsize=trace_fs, color=TOL[i % len(TOL)], weight="bold")
    top = len(ent) - 1 + TRACE_AMP
    apply_pub_style(ax, xlabel="Raman shift (cm$^{-1}$)",
                    ylabel="Intensity + offset (a.u.)")
    ax.set_yticks(range(len(ent)))
    # ticks stop at the data; the extra span to the right only holds the labels
    ax.set_xticks(np.arange(250, CROP[1] + 1, 250))
    ax.set_xlim(CROP[0], CROP[1] + 270)
    ax.set_ylim(-0.15, top + 0.95)
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    return top


def savefig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", dpi=PUB_DPI)
    plt.close(fig)
    from PIL import Image
    with Image.open(p) as im:
        print(f"[OK] {p}   {im.size[0]} x {im.size[1]} px @ {PUB_DPI} dpi")


# ============================ main-text Figure 9 =============================
print("Figure 9 (main text, trimmed):")
panel_entries = {}
fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), dpi=PUB_DPI)
for (letter, key, temps), ax in zip(PANELS, axes.ravel()):
    ent = entries(key, temps)
    panel_entries[key] = ent
    draw_overlay(ax, ent)
    ax.set_title(SERIES[key]["header"], fontsize=12, weight="bold")
    ax.annotate(letter, xy=(0.015, 0.995), xycoords="axes fraction", ha="left",
                va="top", fontsize=13, weight="bold")
    print(f"  {letter} {key}: {[e[0] for e in ent]}")
fig.subplots_adjust(wspace=0.20, hspace=0.26)
savefig(fig, "fig_9_annealing_4panel.png")

# single-panel versions of (a) and (c), kept because the master value table and
# the earlier supplementary batch reference these filenames
for key in ("au", "si300"):
    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=PUB_DPI)
    draw_overlay(ax, panel_entries[key])
    savefig(fig, f"fig_annealing_overlay_{key}.png")

# ======================== supplementary: full series =========================
print("\nSupplementary full series:")
for _, key, _ in PANELS:
    ent = entries(key, report=True)
    fig, ax = plt.subplots(figsize=(8.5, 1.05 + 0.62 * len(ent)), dpi=PUB_DPI)
    draw_overlay(ax, ent, trace_fs=10)
    savefig(fig, f"fig_SM_annealing_overlay_{key}_full.png")
    print(f"  {key}: {len(ent)} traces  {[e[0] for e in ent]}")
