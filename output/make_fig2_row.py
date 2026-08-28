"""Fig 2 rebuild (C11, split per JLR): TWO separate images, joined manually.

  fig_2_ab.png   — (a) Ne + (b) Si cropped 1:1 from the manuscript raster
                   (docx media/image2.png top row; black strips and old (c)
                   discarded, old label ghosts whitened). Native resolution,
                   no rescaling. Letters (a)/(b) top-left.
  fig_2_c_au.png — (c) Au regenerated from the on-disk SRIM damage profile
                   (3 MeV, 2.5e15 cm-2) in matching style: left/bottom spines
                   only, black Total curve, title above box, Depth (um) 0-1.6.

NB the old Fig 2 axes were already correct um units (the mislabelled exports
were only the standalone SRIM_profile.png files).

Requires output/fig2_source_image2.png (extracted from the master docx media/).
Run from the repo root: .venv\\Scripts\\python.exe output\\make_fig2_row.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.stdout.reconfigure(errors="replace")

OUT = "output"
SRC = os.path.join("data", "figure_sources", "fig2_source_image2.png")
AU_CSV = r"data/srim/Au 3 MeV - 2.5e15 ions cm-2/Damage.csv"
DPI = 150

img = np.asarray(Image.open(SRC).convert("RGB"))
A = img[28:454, 0:519].copy()      # (a) Ne panel, native pixels
B = img[28:454, 527:1046].copy()   # (b) Si panel
# whiten the descenders of the old strip "(a)"/"(b)" labels
A[0:26, 0:200] = 255
B[0:26, 0:200] = 255

# ---- part 1: (a)+(b), native 1:1 pixels ----
ph = A.shape[0]
gap = 14
canvas = np.full((ph, A.shape[1] + gap + B.shape[1], 3), 255, dtype=np.uint8)
canvas[:, :A.shape[1]] = A
canvas[:, A.shape[1] + gap:] = B
fig = plt.figure(figsize=(canvas.shape[1] / DPI, ph / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(canvas)
ax.set_axis_off()
for x_frac, letter in [(0.012, "(a)"), (A.shape[1] / canvas.shape[1] + 0.012, "(b)")]:
    ax.annotate(letter, xy=(x_frac, 0.985), xycoords="axes fraction", ha="left",
                va="top", fontsize=13, weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
p1 = os.path.join(OUT, "fig_2_ab.png")
fig.savefig(p1, dpi=DPI)
plt.close(fig)

# ---- part 2: (c) Au, matching style ----
fig, axc = plt.subplots(figsize=(519 / DPI, 480 / DPI), dpi=DPI)
fig.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.16)
d = pd.read_csv(AU_CSV, header=None).to_numpy(float)
axc.plot(d[:, 0] / 1000.0, d[:, 1], color="black", lw=1.3, label="Total")
axc.set_xlim(0, 1.6)
axc.set_ylim(0, 8.2)
axc.set_xticks(np.arange(0, 1.61, 0.2))   # 0.2 steps, matching (a)/(b)
axc.set_xticklabels([f"{t:.1f}" for t in np.arange(0, 1.61, 0.2)])
axc.set_xlabel("Depth (μm)", fontsize=11)
axc.set_ylabel("DPA", fontsize=11)
axc.set_title("Gold into SiC dpa", fontsize=11, pad=6)
axc.spines["top"].set_visible(False)
axc.spines["right"].set_visible(False)
axc.tick_params(axis="both", labelsize=9, direction="out", top=False, right=False)
axc.legend(fontsize=9, frameon=False, loc="upper right")
axc.annotate("(c)", xy=(0.02, 1.10), xycoords="axes fraction", ha="left", va="top",
             fontsize=13, weight="bold", annotation_clip=False)
p2 = os.path.join(OUT, "fig_2_c_au.png")
fig.savefig(p2, dpi=DPI)
plt.close(fig)

for p in (p1, p2):
    w, h = Image.open(p).size
    print(f"[OK] {p}  ({w} x {h} px at {DPI} dpi)")
