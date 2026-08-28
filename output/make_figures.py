"""Generate reference figures + their data CSVs for the paper values, into
output/. Matches the repo publication style (analysis_plotting.apply_pub_style,
PUB_DPI, Paul Tol muted palette). These are REFERENCE figures tied to the master
value table; regenerate from the saved CSVs.
"""
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.stdout.reconfigure(errors="replace")

# runnable as `python output/make_figures.py` from the repo root: sys.path[0] is
# output/, so the repo modules need adding explicitly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import preprocess
from curve_fitting import fit_peaks_regionwise
from config import load_sample_config, find_matching_config
from derived_quantities import (chi_integrated, lo_band_position,
                                line_through_two_points, STRESS_COEFF_6H, STRAIN_COEFF_6H)
from cross_spectrum_ratios import crystallinity_index
from analysis_plotting import apply_pub_style, PUB_DPI

OUT = "output"
os.makedirs(OUT, exist_ok=True)
TOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

# survey conditions ordered least→most retained damage (physics axis)
SURVEY = [
    ("Unirradiated", "input/Unirradiated.csv", None, None),
    ("Si 0.25 dpa\n750 °C", "input/3 Si 750 0.25.csv", 0.25, 750),
    ("Si 2.5 dpa\n750 °C", "input/4 Si 750 2.5.csv", 2.5, 750),
    ("Si 0.25 dpa\n300 °C", "input/2 Si 300 0.25.csv", 0.25, 300),
    ("Si 2.5 dpa\n300 °C", "input/1 Si 300 2.5.csv", 2.5, 300),
    ("Ne 2.5 dpa\n300 °C", "input/0 Ne 300 2.5.csv", 2.5, 300),
]
FTO_WIN = (783.0, 802.0)


def load(path, order=5, norm="vector-0to1", convert_wavelength=False):
    return preprocess(path, 170, 2000, imodpoly_order=order, imodpoly_tol=1e-3,
                      imodpoly_max_iter=100, normalisation=norm, plot=False,
                      save_path=None, convert_wavelength_to_shift=convert_wavelength)


def fit_survey(path):
    cfg = load_sample_config(find_matching_config(path))
    pre = cfg["preprocessing"]
    x, y = load(path, pre.get("baseline_order", 5), pre.get("normalisation", "vector-0to1"))
    _, _, pp, _, _ = fit_peaks_regionwise(x, y, cfg["regions"],
                                          center_tolerance=cfg["fitting"].get("center_tolerance", 100))
    return x, y, pp


def savefig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", dpi=PUB_DPI)
    plt.close(fig)
    print(f"[OK] {p}")


# ---- C21 font bump for Figure 6 only ---------------------------------------
# One global factor applied to every font in the composite and nothing else.
# apply_pub_style hard-codes 12 pt axis titles and 10 pt ticks, so those are
# rescaled here after it runs; the per-call sizes (tick labels 9, the FWHM
# annotation 7, the legend 8, the panel letters 13) carry the factor inline.
# C21 asked for ~1.2x; that collides. Panel (a)'s six two-line category tick
# labels are the binding constraint — "Unirradiated" runs into "Si 0.25 dpa"
# from 1.11x upward. Measured gap between adjacent tick-label boxes:
#   1.00 -> 3.51 pt | 1.06 -> 1.94 pt | 1.10 -> 0.58 pt | 1.11 -> -0.68 pt
# 1.10 is the largest strictly collision-free factor but reads as touching in
# print. C22 ships 1.06, the largest journal-safe factor (JLR decision).
FS_FIG6 = 1.06


def _scale_fonts(ax, fs):
    if fs == 1.0:
        return
    ax.xaxis.label.set_size(12 * fs)
    ax.yaxis.label.set_size(12 * fs)
    ax.tick_params(axis="both", labelsize=10 * fs)


def savefig_exact(fig, name, target, tol=1):
    """Save with a tight bbox at exactly `target` pixels.

    Bigger fonts change what a tight bbox crops, so the C18 figsize no longer
    lands on 2170x810. The figure size is solved for instead (a couple of
    iterations converge), and any residual pixel or two is padded/cropped in
    white at the edges — which cannot move or rescale a single glyph, so the
    fonts stay at the size they were asked for.
    """
    w0, h0 = fig.get_size_inches()
    for _ in range(8):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=PUB_DPI)
        with Image.open(buf) as im:
            w, h = im.size
        if abs(w - target[0]) <= tol and abs(h - target[1]) <= tol:
            break
        fig.set_size_inches(fig.get_size_inches() * np.array(
            [target[0] / w, target[1] / h]))
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if img.size != tuple(target):
        canvas = Image.new("RGB", tuple(target), "white")
        canvas.paste(img, ((target[0] - img.width) // 2,
                           (target[1] - img.height) // 2))
        img = canvas
    p = os.path.join(OUT, name)
    img.save(p)
    plt.close(fig)
    print(f"[OK] {p}   {img.size[0]} x {img.size[1]} px "
          f"(figsize solved {w0:.2f}x{h0:.2f} -> "
          f"{fig.get_size_inches()[0]:.2f}x{fig.get_size_inches()[1]:.2f} in)")


# ---------- collect survey quantities (χ integrated ± baseline band; FTO FWHM; crystallinity) ----------
rows = []
for label, path, dpa, T in SURVEY:
    chis = [chi_integrated(*load(path, o))["chi"] for o in (4, 5, 6)]
    cr = [crystallinity_index(*load(path, o, "none")) for o in (4, 5, 6)]
    x, y, pp = fit_survey(path)
    cand = [p for p in pp if FTO_WIN[0] <= p["mu"] <= FTO_WIN[1]]
    # a resolvable sharp FTO = a narrow peak (FWHM < 20); the broad disordered band is not
    sharp = [p for p in cand if p["FWHM"] < 20]
    fto = max(sharp, key=lambda p: p["Relative_Intensity"]) if sharp else None
    rows.append(dict(label=label, dpa=dpa, T=T,
                     chi=chis[1], chi_lo=min(chis), chi_hi=max(chis),
                     cryst=cr[1], cryst_lo=min(cr), cryst_hi=max(cr),
                     fto_fwhm=(fto["FWHM"] if fto else np.nan),
                     fto_fwhm_err=(fto["FWHM_err"] if fto else np.nan),
                     lo=lo_band_position(x, y)))
df = pd.DataFrame(rows)
chi_floor = df.loc[df.label == "Unirradiated", "chi"].iloc[0]
df["cryst_retained"] = 100 * df["cryst"] / df.loc[df.label == "Unirradiated", "cryst"].iloc[0]
df["cryst_ret_lo"] = 100 * df["cryst_lo"] / df.loc[df.label == "Unirradiated", "cryst_hi"].iloc[0]
df["cryst_ret_hi"] = 100 * df["cryst_hi"] / df.loc[df.label == "Unirradiated", "cryst_lo"].iloc[0]
df.to_csv(os.path.join(OUT, "fig_survey_values.csv"), index=False)

xt = np.arange(len(df))

# ===== Fig: survey χ (integrated) with baseline band =====
fig, ax = plt.subplots(figsize=(7, 4), dpi=PUB_DPI)
yerr = [df["chi"] - df["chi_lo"], df["chi_hi"] - df["chi"]]
ax.errorbar(xt, df["chi"], yerr=yerr, fmt="o", color=TOL[0], capsize=3, ms=6, lw=1.2)
ax.axhline(chi_floor, ls="--", lw=0.8, color=TOL[6], label=f"pristine floor ≈ {chi_floor:.3f} (estimator artefact)")
apply_pub_style(ax, xlabel="", ylabel="χ  (integrated  A$_D$/A$_{TO}$)")
ax.set_xticks(xt); ax.set_xticklabels(df["label"], fontsize=10)
ax.legend(fontsize=8, frameon=False)
savefig(fig, "fig_survey_chi.png")

# ===== Fig: crystallinity retained % =====
fig, ax = plt.subplots(figsize=(7, 4), dpi=PUB_DPI)
# NB the band is one-sided upward for the irradiated bars BY CONSTRUCTION:
# order 5 (the plotted nominal, = the master-table value) gives the minimum C
# of the three baseline orders for every irradiated sample, so the nominal sits
# on the band's lower edge. Whiskers anchor exactly at the bar tops.
yerr = [df["cryst_retained"] - df["cryst_ret_lo"], df["cryst_ret_hi"] - df["cryst_retained"]]
ax.bar(xt, df["cryst_retained"], yerr=yerr, color=TOL[2], alpha=0.85, width=0.6,
       error_kw=dict(capsize=5, capthick=1.4, lw=1.4, ecolor="k"))
apply_pub_style(ax, xlabel="", ylabel="SiC crystallinity retained  (% of pristine)")
ax.set_xticks(xt); ax.set_xticklabels(df["label"], fontsize=10)
savefig(fig, "fig_crystallinity.png")

# ===== Fig 6a: FTO FWHM vs condition (only sharp-FTO conditions resolvable) =====
# C21: back to the C18 layout exactly. `fs` scales every font in the panel by one
# factor and nothing else — it is 1.0 for the standalone reference panel (which
# therefore still reproduces its C18 bytes) and FS_FIG6 for the composite.
def draw_fwhm(ax, fs=1.0):
    res = df[df["fto_fwhm"].notna()]
    ax.errorbar(res.index, res["fto_fwhm"], yerr=res["fto_fwhm_err"], fmt="s",
                color=TOL[1], capsize=3, ms=7, lw=1.2)
    for i, r in df[df["fto_fwhm"].isna()].iterrows():
        ax.annotate("no resolvable\nsharp FTO", (i, 5.8), ha="center", va="bottom",
                    fontsize=7 * fs, color=TOL[5])
        ax.plot(i, 5.7, marker="x", color=TOL[5], ms=8)
    apply_pub_style(ax, xlabel="", ylabel="FTO FWHM (cm$^{-1}$)")
    _scale_fonts(ax, fs)      # before set_xticklabels: that call must win on x
    ax.set_xticks(xt); ax.set_xticklabels(df["label"], fontsize=9 * fs)


# ===== Fig 9b/18: LO position vs irradiation T, 2 doses, DoF=0 slopes =====
# Literature reference restored 2026-08-12 (C18): the main-doc caption promises
# "the dashed line reproduces neutron-irradiated CVD 3C-SiC data from Koyanagi
# et al.", which the C9 regeneration had dropped.
# PROVENANCE. In the previous draft (old Figure 18, docx media/image19.png) the
# comparison was NOT an overlay: panel (a) was Koyanagi's own published figure
# pasted in whole, panel (b) was our two-point plot. That figure prints its LO
# trend on the axes as y = 0.0666x + 904.8 (left axis, cm⁻¹). NB the descending
# grey dashed line on that figure is LINEAR SWELLING on the RIGHT axis, not a
# Raman shift — only the ascending red line is the LO trend. Their red LO points
# span ~230-770 °C (line drawn 223-766 °C), so we redraw over 230-770 °C only,
# no extrapolation. An independent digitisation of that raster's red content
# gives 63.4e-3 / 905.4 (marker pixels bias the slope low), consistent with the
# printed equation and ruling out the ~77e-3 that a back-calculation from the
# stale "63 %" ratio implied — see the C18 note in paper-tracker.md.
# The legend says "(2018)" and not "[44]" deliberately: Mendeley renumbers the
# bibliography, so a bracketed number baked into the raster would go stale.
KOY_SLOPE, KOY_INTERCEPT = 0.0666, 904.8      # cm⁻¹ °C⁻¹, cm⁻¹ (as printed by Koyanagi)
KOY_RANGE = (230.0, 770.0)                    # °C, extent of their own LO data


def draw_lo(ax, save_csv=False, fs=1.0):
    lo_rows = []
    for dpa, col, mk in [(2.5, TOL[1], "o"), (0.25, TOL[0], "s")]:
        # Si-only: the LO-vs-T doses are Si 2.5 dpa and Si 0.25 dpa (Ne excluded)
        sub = df[(df.dpa == dpa) & df.label.str.startswith("Si") & (df["T"].notna())].sort_values("T")
        Ts = sub["T"].to_numpy(); los = sub["lo"].to_numpy()
        ax.errorbar(Ts, los, yerr=1.0, fmt=mk, color=col, capsize=3, ms=7,
                    label=f"{dpa} dpa (data ±1 cm⁻¹)")
        ln = line_through_two_points(Ts[0], los[0], 1.0, Ts[1], los[1], 1.0)
        tt = np.array([250, 800])
        ax.plot(tt, ln["intercept"] + ln["slope"] * tt, "--", color=col, lw=1.1)
        # slopes are NOT annotated on the lines (JLR C8) — they go in the caption:
        if save_csv:
            print(f"[caption] LO slope {dpa} dpa: ({ln['slope']*1e3:.1f} ± {ln['u_slope']*1e3:.1f})"
                  f" x10-3 cm-1/degC, intercept {ln['intercept']:.0f} cm-1")
        lo_rows += [dict(dpa=dpa, T=Ts[0], LO=los[0]), dict(dpa=dpa, T=Ts[1], LO=los[1])]
    # literature comparison: dotted grey, visually distinct from our dashed fits
    tk = np.array(KOY_RANGE)
    ax.plot(tk, KOY_INTERCEPT + KOY_SLOPE * tk, ls=":", color="0.45", lw=1.7,
            label="Koyanagi et al. (2018),\nneutron-irradiated CVD 3C-SiC")
    if save_csv:
        print(f"[caption] Koyanagi reference line: omega = {KOY_INTERCEPT} + "
              f"{KOY_SLOPE*1e3:.1f}e-3 T, drawn {KOY_RANGE[0]:.0f}-{KOY_RANGE[1]:.0f} degC "
              f"({KOY_INTERCEPT + KOY_SLOPE*KOY_RANGE[0]:.1f} -> "
              f"{KOY_INTERCEPT + KOY_SLOPE*KOY_RANGE[1]:.1f} cm-1); "
              f"our 2.5 dpa slope is {48.5/(KOY_SLOPE*1e3)*100:.0f} +- "
              f"{3.1/(KOY_SLOPE*1e3)*100:.0f} % of it")
    apply_pub_style(ax, xlabel="Irradiation temperature (°C)",
                    ylabel="Folded-LO band position (cm$^{-1}$)")
    ax.legend(fontsize=8 * fs, frameon=False, loc="lower right")
    _scale_fonts(ax, fs)
    if save_csv:
        pd.DataFrame(lo_rows).to_csv(os.path.join(OUT, "fig_lo_vs_T.csv"), index=False)


# the two panels also exist standalone (reference figures, not placed in either
# document). Built exactly as in C18, fs=1.0, so they reproduce their C18 bytes.
fig, ax = plt.subplots(figsize=(7, 4), dpi=PUB_DPI)
draw_fwhm(ax)
savefig(fig, "fig_fto_fwhm.png")

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=PUB_DPI)
draw_lo(ax, save_csv=True)
savefig(fig, "fig_lo_vs_T.png")

# ===== composite main Figure 6: (a) FTO FWHM + (b) LO vs irradiation T =======
# C21: the C18 build with every font multiplied by FS_FIG6 and nothing else —
# same layout, same 2170x810 raster, same legend placement, same Koyanagi
# element. The doc frame (6.318 x 2.358 in) is unchanged, so the bump raises the
# printed sizes by the same factor.
fig, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(13, 4.4), dpi=PUB_DPI)
draw_fwhm(ax9a, fs=FS_FIG6)
draw_lo(ax9b, fs=FS_FIG6)
for ax_, letter in [(ax9a, "(a)"), (ax9b, "(b)")]:
    ax_.annotate(letter, xy=(0.02, 0.98), xycoords="axes fraction", ha="left",
                 va="top", fontsize=13 * FS_FIG6, weight="bold")
fig.subplots_adjust(wspace=0.22)
savefig_exact(fig, "fig_6_fwhm_lo.png", (2170, 810))

# ===== Fig 12 (consolidating 17): χ vs annealing T — (a) Au 2.5e15, (b) Si 2.5 dpa/300 °C =====
# Au series recomputed 2026-07-29 with the same integrated estimator (annealing_chi.py
# --convert-wavelength on "input/Annealing/Au 2.5e15 RT"); RT endpoints
# from au_chi_RT.csv (narrow-window zoom scans excluded — D window not covered).
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=PUB_DPI, sharey=False)


# EXCLUDED spectra (kept in the CSVs; dropped from the figure with reason):
# 900C (2) is SIGNAL-FREE — raw counts flat at the ~1353-count dark level, no
# Raman bands at all (cf. 900C (1)/800C/1000C which all show the TO band), so
# vector-0to1 amplifies pure noise and its chi=0.703 is a noise ratio, not
# repeat-spot scatter. Inspected 2026-07-29 (C8).
CHI_EXCLUDE = ["900C Si 300 2,5dpa --Spectrum--052--Spec.Data 1 (2).txt"]


def chi_panel(ax, csv, label):
    ann = pd.read_csv(os.path.join(OUT, csv))
    ann = ann[~ann["file"].isin(CHI_EXCLUDE)]
    yerr = [ann["chi"] - ann["chi_min"], ann["chi_max"] - ann["chi"]]
    ax.errorbar(ann["temp_C"], ann["chi"], yerr=yerr, fmt="o", color=TOL[0], capsize=2,
                ms=5, lw=1, label="χ (integrated) ± baseline band")
    dup = ann[ann.duplicated("temp_C", keep=False)]
    ax.scatter(dup["temp_C"], dup["chi"], facecolors="none", edgecolors=TOL[1], s=90,
               lw=1.3, label="repeat-spot scatter (D5)")
    apply_pub_style(ax, xlabel="Annealing temperature (°C)",
                    ylabel="χ  (integrated  A$_D$/A$_{TO}$)")
    ax.annotate(label, xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top",
                fontsize=12, weight="bold")


chi_panel(axa, "au_chi_vs_T.csv", "(a)")
rt = pd.read_csv(os.path.join(OUT, "au_chi_RT.csv"))
for stage, T, mk in [("RT before", 25, "D"), ("RT after", 1290, "^")]:
    sub = rt[rt.stage == stage]
    ax_yerr = [sub["chi"] - sub["chi_min"], sub["chi_max"] - sub["chi"]]
    axa.errorbar([T] * len(sub), sub["chi"], yerr=ax_yerr, fmt=mk, mfc="none",
                 color=TOL[2], capsize=2, ms=6, lw=1, label=stage)
# ONE legend for the whole figure, in (a)'s empty upper-right corner — the
# series encodings are identical in both panels, so (b) carries none (JLR C8).
axa.legend(fontsize=8, frameon=False, loc="upper right")
# unified header style: ion, dose (dpa), irradiation temperature. 8 dpa = peak
# damage of the non-plateau Au profile (SRIM 7.82 dpa @ 415 nm) — the caption
# carries that nuance (JLR C9); RT confirmed per methods §2.1.
axa.set_title("Au, 8 dpa, RT", fontsize=11, weight="bold")

chi_panel(axb, "si300_2.5_chi_vs_T.csv", "(b)")
axb.set_title("Si, 2.5 dpa, 300 °C", fontsize=11, weight="bold")
savefig(fig, "fig_annealing_chi_vs_T.png")

# ===== Fig 13 (consolidating 19): stress map, 0.25 dpa 750 °C =====
# JLR's peak positions (manual WiRE cursor read-offs, stated in §2.3.4) on
# their spatial grid (Stress Mapping.xlsx Sheet1, rows 30-35 x cols 11-14 =
# cross-shaped point map; grid step 2.5 µm confirmed via §2.2.3, JLR
# 2026-07-29). Discrete points only — NO interpolated
# surface (the published "compressive island" was an interpolation artefact;
# real min −11 MPa, below the ±326 MPa floor). Per-point u(ωᵢ) comes from
# output/stress_point_fits.csv (fit_stress_points.py: Lorentzian fit of each
# point spectrum, rank-order matched to the Excel positions).
MAP_PTS = [  # (excel_row, excel_col, peak_pos)
    (30, 12, 785.14),
    (31, 11, 786.0), (31, 12, 788.7), (31, 13, 788.75), (31, 14, 785.0),
    (32, 11, 786.9), (32, 12, 786.93), (32, 13, 788.72), (32, 14, 786.93),
    (33, 12, 786.95), (34, 12, 786.95), (35, 12, 787.71),
]
w0, u_w0 = 788.72, 0.44
STEP_UM = 2.5                                  # map step (methods §2.2.3)
rows_, cols_, POS = (np.array(v) for v in zip(*MAP_PTS))
gx = (cols_ - cols_.min()) * STEP_UM           # µm, y up
gy = (rows_.max() - rows_) * STEP_UM
pf = pd.read_csv(os.path.join(OUT, "stress_point_fits.csv"))
u_map = dict(zip(pf["excel_pos"].round(2), pf["u_wi"]))
u_wi = np.array([u_map[round(p, 2)] for p in POS])
sigma = -STRESS_COEFF_6H * (POS - w0)          # σ = -k·Δω ; downshift → tensile (+)
u_sig = STRESS_COEFF_6H * np.hypot(u_wi, u_w0)
floor = STRESS_COEFF_6H * 2 * u_w0
below = np.abs(sigma) < floor

fig, (axm, axs) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=PUB_DPI,
                               gridspec_kw={"width_ratios": [1, 1.3]})
# (a) discrete colored markers over JLR's optical image. This recreates his
# heatmap-over-optical presentation WITHOUT the cubic-griddata interpolation
# that invented the "compressive island". Pixel positions = his clicked
# red-dot coordinates ("Stress Map Pt 2/find points.py" output), same reading
# order as MAP_PTS (top point, two rows L->R, vertical tail). Physical scale
# from the 2.5 µm grid step = mean adjacent-column pixel spacing (~50 px).
res = ~below
PT2 = r"data/stress_map"
pix = pd.read_csv(os.path.join(PT2, "stress_map_points_pixels.csv"))
img = plt.imread(os.path.join(PT2, "Irr Region of Interest Marked.png"))
px_x, px_y = pix["x"].to_numpy(), pix["y"].to_numpy()
px_per_um = np.mean(np.concatenate([np.diff(np.sort(px_x[1:5])),
                                    np.diff(np.sort(px_x[5:9]))])) / STEP_UM
axm.imshow(img)
sc = axm.scatter(px_x[res], px_y[res], c=sigma[res], cmap="Reds", s=170, marker="o",
                 vmin=0, vmax=1400, edgecolors="k", linewidths=0.8, zorder=3)
axm.scatter(px_x[below], px_y[below], facecolors="white", edgecolors="0.35", s=170,
            marker="s", linewidths=1.4, zorder=3,
            label=f"below detection floor (±{floor:.0f} MPa) → ≈ 0")
# no max-value label on the image — every placement collides with a marker on
# this dense cross; the max is visible in (b) and quoted in the caption
print(f"[caption] max tensile point {sigma.max():.0f} MPa at "
      f"({px_x[int(np.argmax(sigma))]:.0f}, {px_y[int(np.argmax(sigma))]:.0f}) px")
# 5 µm scale bar, bottom left
bar_px = 5 * px_per_um
x0, y0 = img.shape[1] * 0.06, img.shape[0] * 0.94
axm.plot([x0, x0 + bar_px], [y0, y0], color="k", lw=3, solid_capstyle="butt")
axm.annotate("5 µm", ((2 * x0 + bar_px) / 2, y0 - 6), ha="center", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
# legend outside, below the image, so it never covers image content
axm.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.01))
axm.annotate("(a)", xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top",
             fontsize=12, weight="bold",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
axm.set_axis_off()
cb = fig.colorbar(sc, ax=axm, shrink=0.85, pad=0.04)
cb.set_label("σ (MPa)", fontsize=9)
fig.subplots_adjust(wspace=0.35)

# (b) sorted magnitudes with error bars + floor band
order = np.argsort(sigma)
idx = np.arange(len(POS))
axs.axhspan(-floor, floor, color=TOL[6], alpha=0.3, label=f"± detection floor ({floor:.0f} MPa)")
axs.errorbar(idx, sigma[order], yerr=u_sig[order], fmt="o", color=TOL[1], capsize=3, ms=6,
             label="tensile stress ± stat (per point)")
axs.axhline(0, color="k", lw=0.6)
apply_pub_style(axs, xlabel="map point (sorted by σ)", ylabel="In-plane stress σ (MPa)")
axs.legend(fontsize=8, frameon=False, loc="upper left")
axs.annotate("(b)", xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top",
             fontsize=12, weight="bold")
# (caption content — formula, u(ω₀), per-point u range, k systematic — lives in
# the manuscript caption, not in the image; per-point u range for it:
print(f"[caption] per-point stat u(σ) = ±{u_sig.min():.0f}–{u_sig.max():.0f} MPa")
savefig(fig, "fig_stress_map.png")
pd.DataFrame(dict(map_row=rows_ - rows_.min(), map_col=cols_ - cols_.min(),
                  x_um=gx, y_um=gy,
                  peak_pos=POS, delta_omega=POS - w0, sigma_MPa=sigma,
                  u_wi=u_wi, u_sigma_MPa=u_sig, below_floor=below,
                  strain_pct=STRAIN_COEFF_6H * (POS - w0))
            ).to_csv(os.path.join(OUT, "fig_stress_0.25dpa750.csv"), index=False)


def norm01(y):
    y = y - y.min()
    return y / y.max() if y.max() > 0 else y


# ===== §3.1.1 survey overlay (C14) ==========================================
# Landscape ~1.6:1, larger fonts, right-hand-side trace labels (no legend), and
# the band / crystalline-feature annotations of an earlier version of this
# figure restored (that raster is a working file, not part of the repo).
#
# STACKING: ascending disorder BOTTOM→TOP, i.e. unirradiated at the bottom and
# Ne 2.5 dpa / 300 °C at the top — the same order as the caption and as the
# earlier figure. SURVEY is already in that order, so it is NOT reversed here.
SURVEY_AMP = 0.9                # trace height per unit offset; 0.9 keeps a gap
# Vibrational-band brackets, drawn above the top (most disordered) trace.
SURVEY_BANDS = [("Si-Si", 170.0, 320.0), ("Si-Si", 480.0, 595.0),
                ("Si-C", 730.0, 990.0), ("C-C", 1270.0, 1680.0)]
# Crystalline-SiC features, labelled on the bottom (unirradiated) trace.
# (text, x, horizontal alignment, y offset from the local trace maximum)
SURVEY_FEATURES = [("FTA", 215.0, "center", 0.05),
                   ("Crystalline SiC", 555.0, "center", 0.05),
                   ("FTO", 805.0, "left", -0.13),
                   ("FLO", 975.0, "left", 0.04),
                   ("2$^{nd}$ Order SiC", 1520.0, "center", 0.05)]


def _local_max(x, y, x0, halfwidth=30.0):
    m = (x >= x0 - halfwidth) & (x <= x0 + halfwidth)
    return float(np.max(y[m])) if m.any() else 0.0


def _bracket(ax, x0, x1, y, label, fontsize):
    ax.plot([x0, x0, x1, x1], [y - 0.09, y, y, y - 0.09], lw=1.0, color="0.3",
            clip_on=False)
    ax.text(0.5 * (x0 + x1), y + 0.05, label, ha="center", va="bottom",
            fontsize=fontsize, color="0.15", weight="bold")


def survey_overlay(name="fig_survey_overlay.png"):
    ent = [(label.replace("\n", " "), *load(path)) for label, path, _, _ in SURVEY]
    fig, ax = plt.subplots(figsize=(11, 6.9), dpi=PUB_DPI)   # ~1.6:1 landscape
    xmax = 0.0
    for i, (label, x, y_raw) in enumerate(ent):
        y = SURVEY_AMP * norm01(y_raw) + i
        ax.plot(x, y, lw=1.0, color=TOL[i % len(TOL)])
        ax.text(x.max() + 25, y[-1], label, ha="left", va="center", fontsize=12,
                color=TOL[i % len(TOL)], weight="bold")
        xmax = max(xmax, float(x.max()))
    # crystalline features on the unirradiated trace (offset 0, drawn first)
    x0, y0 = ent[0][1], SURVEY_AMP * norm01(ent[0][2])
    for text, xpos, ha, dy in SURVEY_FEATURES:
        ax.text(xpos, _local_max(x0, y0, xpos) + dy, text, ha=ha, va="bottom",
                fontsize=10, color="0.15")
    top = len(ent) - 1 + SURVEY_AMP
    for label, a, b in SURVEY_BANDS:
        _bracket(ax, a, b, top + 0.20, label, 13)
    apply_pub_style(ax, xlabel="Raman shift (cm$^{-1}$)", ylabel="Intensity (a.u.)")
    ax.set_yticks(range(len(ent)))
    # ticks stop at the data; the extra span to the right only holds the labels
    ax.set_xticks(np.arange(250, 2001, 250))
    ax.set_xlim(170, xmax + 350)
    ax.set_ylim(-0.12, top + 0.62)
    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", dpi=PUB_DPI)
    plt.close(fig)
    with Image.open(p) as im:
        print(f"[OK] {p}   {im.size[0]} x {im.size[1]} px @ {PUB_DPI} dpi")


survey_overlay()

# Annealing-series overlays (main-text Fig 11 + the full supplementary versions)
# live in output/make_fig11_annealing.py: they need the RT-before/RT-after
# reference rows and the per-series filename conventions that
# output/annealing_series.py resolves, which this script does not carry.

print("\nAll figures + data CSVs written to output/.")
