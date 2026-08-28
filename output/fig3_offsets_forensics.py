"""Are the four Figure 3 traces stacked by display offsets, or are those their
as-acquired baselines?  Ticket C17(b).

The caption claims "All spectra are shown as-acquired (no post-processing)", yet
the four traces sit at ~325 / ~331 / ~336 / ~342 a.u. The underlying single-crystal
spectra are not on this machine (searched 2026-08-06: no .opj/.opju/.ogg/.wip/.wdf
anywhere under the user profile, and no single-crystal exports in the repo or in
any of the authors' working folders), so the only evidence available is the
published raster itself.

This script recovers the four traces from `output/fig_3_source_image4.png` by
colour, calibrates pixels to the plotted y axis from the tick marks, and measures
each trace's baseline in windows where every trace is flat. It then tests the
stack against what an as-acquired stack would have to look like.

Writes: output/fig3_trace_baselines.csv, output/fig3_offsets_forensics.png
Run from the repo root:  .venv\\Scripts\\python.exe output\\fig3_offsets_forensics.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(errors="replace")

OUT = "output"
SRC = os.path.join("data", "figure_sources", "fig_3_source_image4.png")
# x axis calibration from the major tick marks measured on the axis line
X0_PX, X_PER_CM1 = 125.0, (834.0 - 125.0) / 2200.0
UNIRR_SCALE = 0.25          # the legend's "Unirradiated (x0.25)"

TRACES = [                  # legend order, top entry first
    ("Si 0.25 dpa 750 °C", "blue", lambda r, g, b: (b > 120) & (b > r + 60) & (b > g + 60)),
    ("Si 2.5 dpa 300 °C", "green", lambda r, g, b: (g > 120) & (g > r + 50) & (g > b + 50)),
    ("Ne 2.5 dpa 300 °C", "red", lambda r, g, b: (r > 120) & (r > g + 70) & (r > b + 70)),
    ("Unirradiated (x0.25)", "black", None),
]
# windows where every trace is flat (no band, no peak) — chosen from the figure
FLAT_WINDOWS = [(1650, 1750), (1750, 1850), (1850, 1990)]
PLOT_WINDOW = (1600, 2000)


def px_to_cm1(x):
    return (x - X0_PX) / X_PER_CM1


def cm1_to_px(v):
    return X0_PX + v * X_PER_CM1


im = Image.open(SRC)
flat = Image.new("RGB", im.size, (255, 255, 255))
flat.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
a = np.asarray(flat).astype(int)
H, W, _ = a.shape
r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
grey = (abs(r - g) < 45) & (abs(g - b) < 45) & (abs(r - b) < 45)
dark = grey & (a.sum(axis=2) < 480)

# ---------------------------------------------------------------- y calibration
# y tick marks point OUTWARD, i.e. they are the short dark runs immediately left
# of the spine at x = 124/125
tick_rows = []
band = dark[:, 112:124]
for y in range(60, 650):
    if band[y].sum() >= 10:
        tick_rows.append(y)
groups, cur = [], [tick_rows[0]]
for y in tick_rows[1:]:
    if y - cur[-1] <= 2:
        cur.append(y)
    else:
        groups.append(sum(cur) / len(cur))
        cur = [y]
groups.append(sum(cur) / len(cur))
LABELS = [355, 350, 345, 340, 335, 330, 325]          # top to bottom
if len(groups) != len(LABELS):
    print(f"[!] found {len(groups)} y ticks, expected {len(LABELS)}: {groups}")
    raise SystemExit(1)
slope, intercept = np.polyfit(groups, LABELS, 1)
resid = np.array(LABELS) - (slope * np.array(groups) + intercept)
print("y calibration from the tick marks:")
for yp, lab in zip(groups, LABELS):
    print(f"   y={yp:6.1f} px  ->  {lab}")
print(f"   {slope:.6f} a.u./px  (1 a.u. = {abs(1/slope):.2f} px)"
      f"   max residual {np.abs(resid).max():.4f} a.u.")


def px_to_val(y):
    return slope * y + intercept


# ------------------------------------------------------------- trace extraction
# The legend's line swatches are the SAME colours as the traces and sit at
# x = 598-720 px, i.e. 1467-1846 cm-1 — right inside the flat windows used for the
# baselines. Blank the whole legend box out of every colour mask.
LEGEND_BOX = (596, 999, 0, 178)
EXCLUDE = np.zeros((H, W), bool)
EXCLUDE[LEGEND_BOX[2]:LEGEND_BOX[3] + 1, LEGEND_BOX[0]:LEGEND_BOX[1] + 1] = True


def trace_y(mask, x, ylo=60, yhi=652):
    """Centre row of the trace in column x, or nan."""
    ys = np.where(mask[ylo:yhi, x] & ~EXCLUDE[ylo:yhi, x])[0]
    return ylo + float(np.median(ys)) if len(ys) else np.nan


# Black ink that is NOT the unirradiated trace: the in-plot labels, the band
# brackets and the legend. Measured in C16.
BLACK_TEXT_BOXES = [
    (169, 229, 44, 72), (268, 329, 44, 72), (373, 429, 44, 72),      # band labels
    (158, 246, 75, 124), (254, 342, 75, 124), (349, 455, 75, 124),   # brackets
    (546, 595, 172, 200), (509, 705, 202, 257),                      # C-C label + bracket
    (322, 371, 523, 544), (411, 459, 523, 544),                      # FTO, FLO
    (153, 199, 545, 565), (583, 716, 558, 580),                      # FTA, 2nd Order SiC
    (596, 999, 0, 178),                                              # legend
]
BLACK_OK = dark.copy()
for bx0, bx1, by0, by1 in BLACK_TEXT_BOXES:
    BLACK_OK[by0:by1 + 1, bx0:bx1 + 1] = False
BLACK_OK[650:, :] = False            # x axis line and everything below
BLACK_OK[:, :127] = False            # y axis spine and its labels


def black_trace_y(x, mode="median"):
    """The unirradiated trace with every black annotation masked out.

    `median` gives the line centre (right for the flat baseline); `top` gives the
    topmost pixel, which is the only meaningful reading where the line is near
    vertical, i.e. on the folded-TO/LO spikes.
    """
    ys = np.where(BLACK_OK[400:650, x])[0]
    if not len(ys):
        return np.nan
    ys = ys + 400
    if mode == "top":
        return float(ys.min())
    keep = ys[ys >= ys.max() - 8]        # bottom cluster only
    return float(np.median(keep))


xs = np.arange(int(cm1_to_px(60)), int(cm1_to_px(2000)) + 1)
curves = {}
for name, colour, pred in TRACES:
    if pred is None:
        y = np.array([black_trace_y(x) for x in xs])
    else:
        m = pred(r, g, b)
        y = np.array([trace_y(m, x) for x in xs])
    curves[colour] = px_to_val(y)

# ------------------------------------------------------------------ baselines
rows = []
print("\nBaseline of each trace in the flat windows (plotted a.u.):")
for name, colour, _ in TRACES:
    vals = {}
    for w in FLAT_WINDOWS:
        m = (px_to_cm1(xs) >= w[0]) & (px_to_cm1(xs) <= w[1])
        v = curves[colour][m]
        vals[w] = (float(np.nanmedian(v)), float(np.nanstd(v)))
    txt = "   ".join(f"{w[0]}-{w[1]}: {v[0]:7.3f} ± {v[1]:.3f}" for w, v in vals.items())
    print(f"  {name:22s} {txt}")
    rows.append(dict(trace=name, colour=colour,
                     **{f"baseline_{w[0]}_{w[1]}": vals[w][0] for w in FLAT_WINDOWS},
                     **{f"scatter_{w[0]}_{w[1]}": vals[w][1] for w in FLAT_WINDOWS}))

df = pd.DataFrame(rows)
ref = f"baseline_{FLAT_WINDOWS[-1][0]}_{FLAT_WINDOWS[-1][1]}"
base = df[ref].to_numpy()

# ------------------------------------------------------------------- the tests
print(f"\n--- Test 1: are the separations a constant step? (window "
      f"{FLAT_WINDOWS[-1][0]}-{FLAT_WINDOWS[-1][1]} cm-1) ---")
order = np.argsort(base)
for i in range(len(order) - 1):
    lo, hi = order[i], order[i + 1]
    print(f"  {df.trace[hi]:22s} - {df.trace[lo]:22s} = {base[hi] - base[lo]:6.3f} a.u.")
steps = np.diff(np.sort(base))
print(f"  steps {np.round(steps, 3)}  mean {steps.mean():.3f}"
      f"  spread {steps.max() - steps.min():.3f} a.u."
      f"  -> {'CONSTANT' if steps.max() - steps.min() < 0.5 else 'NOT a constant step'}")

print("\n--- Test 2: does the stack survive undoing the x0.25 on the unirradiated? ---")
u = df.index[df.colour == "black"][0]
print(f"  plotted unirradiated baseline           {base[u]:8.3f} a.u.")
print(f"  same baseline with the x0.25 undone     {base[u]/UNIRR_SCALE:8.3f} a.u."
      f"   (i.e. the raw level it would represent)")
others = base[df.colour != "black"]
print(f"  the three irradiated baselines          {np.round(others, 3)}")
print(f"  ratio raw-unirradiated / mean-irradiated = "
      f"{(base[u]/UNIRR_SCALE)/others.mean():.2f}x")

print("\n--- Test 3: peak height above baseline, and the SCALE-INVARIANT ratio h/B ---")
# h/B is unchanged by ANY multiplicative rescale (including the stated x0.25) but
# is driven down by an additive pedestal, so it is the one quantity that can tell
# a scaled spectrum from an offset one.
heights, ratios = [], []
for i, (name, colour, _) in enumerate(TRACES):
    top = (np.array([black_trace_y(x, "top") for x in xs]) if colour == "black"
           else None)
    peak = px_to_val(np.nanmin(top)) if top is not None else np.nanmax(curves[colour])
    h = peak - base[i]
    heights.append(h)
    ratios.append(h / base[i])
    print(f"  {name:22s} peak {peak:7.2f}  baseline {base[i]:7.2f}"
          f"  h = {h:6.3f} a.u.   h/B = {h/base[i]:.4f}")
df["peak_height"] = heights
df["h_over_B"] = ratios

print("\n--- Test 4: the same ratio in this project's own RAW spectra ---")
from preprocessing import _read_spectrum_table, wavelength_to_shift
RAW = [("Unirradiated RB-SiC (survey)", "input/Unirradiated.csv", True),
       ("Si 0.25 dpa 750 °C (survey)", "input/3 Si 750 0.25.csv", True),
       ("Si 2.5 dpa 300 °C (survey)", "input/1 Si 300 2.5.csv", True),
       ("Ne 2.5 dpa 300 °C (survey)", "input/0 Ne 300 2.5.csv", True),
       ("Ne RT, through furnace glass",
        "input/Annealing/Ne 2.5dpa 300C/02 REFEL Ne 300 2,5dpa--Spectrum--002--Spec.Data 1RT.txt", False)]
raw_ratio = {}
for label, path, conv in RAW:
    xr, yr = _read_spectrum_table(path)
    if conv:
        xr = wavelength_to_shift(xr, 532, False)
    o = np.argsort(xr)
    xr, yr = xr[o], yr[o]
    B = float(np.median(yr[(xr >= 1850) & (xr <= 1990)]))
    h = float(np.max(yr[(xr >= 170) & (xr <= 2000)]) - B)
    raw_ratio[label] = h / B
    print(f"  {label:32s} baseline {B:9.1f}  h {h:8.1f}   h/B = {h/B:.3f}")
u_raw = raw_ratio["Unirradiated RB-SiC (survey)"]
irr_raw = np.mean([v for k, v in raw_ratio.items() if "Unirradiated" not in k and "glass" not in k])
print(f"\n  In raw data the UNIRRADIATED sample has the HIGHEST h/B: "
      f"{u_raw:.3f} vs {irr_raw:.3f} mean for the irradiated ones "
      f"(ratio {u_raw/irr_raw:.2f}x).")
u_fig = ratios[[t[1] for t in TRACES].index("black")]
irr_fig = np.mean([ratios[i] for i, t in enumerate(TRACES) if t[1] != "black"])
print(f"  In Figure 3 the unirradiated trace gives h/B = {u_fig:.4f} vs "
      f"{irr_fig:.4f} for the irradiated ones (ratio {u_fig/irr_fig:.2f}x).")
print(f"  h/B is invariant under the stated x0.25, so this is a like-for-like "
      f"comparison.\n  Figure-3 ratios are {irr_raw/irr_fig:.0f}x (irradiated) and "
      f"{u_raw/u_fig:.0f}x (unirradiated) smaller than the raw spectra.")

print("\n--- Test 5: implied additive pedestal, if the true h/B matched raw data ---")
for i, (name, colour, _) in enumerate(TRACES):
    target = u_raw if colour == "black" else irr_raw
    b_true = heights[i] / target
    print(f"  {name:22s} would need baseline {b_true:6.2f} a.u. for h/B = {target:.2f}"
          f"  ->  pedestal {base[i] - b_true:7.2f} a.u. "
          f"({100*(base[i]-b_true)/base[i]:.1f}% of the plotted baseline)")

print("\n--- Test 6: do adjacent traces ever come close, across the WHOLE range? ---")
# The one test that separates a chosen stack from an accidental one. Four spectra
# of similar shape, plotted on a common axis with whatever backgrounds they
# happened to have, would sooner or later approach or cross each other. A stack
# whose gaps never close is a stack somebody chose.
stack = [t for t in TRACES]
ordered = sorted(range(len(TRACES)), key=lambda i: base[i])   # bottom to top
for k in range(len(ordered) - 1):
    lo, hi = ordered[k], ordered[k + 1]
    gap = curves[TRACES[hi][1]] - curves[TRACES[lo][1]]
    good = np.isfinite(gap)
    cm = px_to_cm1(xs)[good]
    gg = gap[good]
    j = int(np.argmin(gg))
    print(f"  {TRACES[hi][0]:22s} over {TRACES[lo][0]:22s}"
          f"  gap min {gg.min():6.3f} a.u. at {cm[j]:6.0f} cm-1"
          f"   median {np.median(gg):6.3f}   max {gg.max():6.3f}")
    print(f"    {'':22s}      min gap is {100*gg.min()/max(heights[lo], heights[hi]):.0f}%"
          f" of the taller trace's own height; crossings: {int((gg <= 0).sum())} px")

df.to_csv(os.path.join(OUT, "fig3_trace_baselines.csv"), index=False)
print(f"\n[OK] {OUT}/fig3_trace_baselines.csv")

# ---------------------------------------------------------------------- figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
COL = {"blue": "#3333cc", "green": "#22aa22", "red": "#dd2222", "black": "#222222"}
for i, (name, colour, _) in enumerate(TRACES):
    ax1.plot(px_to_cm1(xs), curves[colour], lw=0.8, color=COL[colour], label=name)
    m = (px_to_cm1(xs) >= PLOT_WINDOW[0]) & (px_to_cm1(xs) <= PLOT_WINDOW[1])
    ax2.plot(px_to_cm1(xs)[m], curves[colour][m], lw=0.9, color=COL[colour])
    ax2.axhline(base[i], ls="--", lw=0.7, color=COL[colour])
for w in FLAT_WINDOWS:
    ax2.axvspan(w[0], w[1], color="0.85", alpha=0.35)
ax1.set_xlabel("Raman shift (cm$^{-1}$)")
ax1.set_ylabel("Plotted intensity (a.u.)")
ax1.legend(fontsize=7, frameon=False)
ax1.set_title("traces recovered from the raster\n(unirradiated drawn from its lower "
              "envelope, so its sharp peaks read short here)", fontsize=9)
ax2.set_xlabel("Raman shift (cm$^{-1}$)")
ax2.set_ylabel("Plotted intensity (a.u.)")
ax2.set_title("flat tail, with the measured baselines", fontsize=10)
for ax in (ax1, ax2):
    ax.grid(alpha=0.3, ls=":")
fig.tight_layout()
p = os.path.join(OUT, "fig3_offsets_forensics.png")
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {p}")
