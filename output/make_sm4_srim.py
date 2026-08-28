"""SM.4 — SRIM damage/implantation profiles for the Au, He, Ni and Pb
irradiations (per Dong Liu comment [30]), from JLR's exported SRIM CSVs in
"data/srim/<condition>/{Damage,Ions}.csv"
(depth [nm], value; Damage = dpa at the folder's fluence, Ions = ion
concentration as atomic fraction — verified against range/straggle estimates).

Same-ion fluence variants are EXACT linear rescalings of one another (checked:
ratio 2.000 / 10.000 across the profile), so each panel plots one fluence and
the caption states the scaling. Dual axis (dpa left / at.% right) kept for
consistency with main-text Fig 2. Depth in nm (µm for the 93.6 MeV Pb panel).

Run from the repo root: .venv\\Scripts\\python.exe output\\make_sm4_srim.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(errors="replace")

# runnable as `python output/make_sm4_srim.py` from the repo root: sys.path[0] is
# output/, so the repo modules need adding explicitly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_plotting import apply_pub_style, PUB_DPI

OUT = "output"
BASE = r"data/srim"
TOL_BLUE, TOL_RED = "#4477AA", "#EE6677"

# (panel, title, folder plotted, depth unit, extra-fluence note for the stub,
#  empty-corner anchor for the peak annotation: axes-fraction (x, ha))
PANELS = [
    ("(a)", "Au 3 MeV", "Au 3 MeV - 2.5e15 ions cm-2", "nm",
     "2.5e15 cm^-2 plotted; 5e15 cm^-2 = x2", (0.98, "right")),
    ("(b)", "He 200 keV", "He 200 kV - 2.5e15 ions cm-2 10x100", "nm",
     "2.5e15 cm^-2 plotted; 5e15 cm^-2 = x2", (0.08, "left")),
    ("(c)", "Ni 1 MeV", "Ni 1 MeV - 2.5e15 ions cm-2 5x300", "nm",
     "2.5e15 cm^-2 plotted; 5e15 cm^-2 = x2", (0.98, "right")),
    ("(d)", "Pb 93.6 MeV", "Pb 93.6 MeV - 1e13 ions cm-2 50x300", "um",
     "1e13 cm^-2 plotted; 1e11-5e12 cm^-2 scale linearly", (0.08, "left")),
]


def load(folder, kind):
    a = pd.read_csv(os.path.join(BASE, folder, f"{kind}.csv"), header=None).to_numpy(float)
    return a[:, 0], a[:, 1]


fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=PUB_DPI)
stubs = []
for ax, (panel, title, folder, unit, flnote, (ax_x, ax_ha)) in zip(axes.ravel(), PANELS):
    xd, yd = load(folder, "Damage")
    xi, yi = load(folder, "Ions")
    scale = 1e-3 if unit == "um" else 1.0
    xlab = "Depth (µm)" if unit == "um" else "Depth (nm)"

    ax.plot(xd * scale, yd, color=TOL_BLUE, lw=1.4)
    apply_pub_style(ax, xlabel=xlab, ylabel="Damage (dpa)")
    ax.yaxis.label.set_color(TOL_BLUE)
    ax.tick_params(axis="y", colors=TOL_BLUE)

    ax2 = ax.twinx()
    ax2.plot(xi * scale, 100 * yi, color=TOL_RED, lw=1.2, ls="--")
    ax2.set_ylabel("Ion concentration (at.%)", fontsize=11, color=TOL_RED)
    ax2.tick_params(axis="y", labelsize=10, colors=TOL_RED)
    ax2.spines["top"].set_visible(False)

    ipk, jpk = int(np.argmax(yd)), int(np.argmax(yi))
    # peak note in the panel's empty corner (axes fraction), clear of both curves
    ax.annotate(f"peak {yd.max():.3g} dpa\n@ {xd[ipk]*scale:g} {xlab[7:-1]}",
                xy=(ax_x, 0.86), xycoords="axes fraction", ha=ax_ha, va="top",
                fontsize=8, color=TOL_BLUE)
    ax.annotate(panel, xy=(0.02, 0.98), xycoords="axes fraction", ha="left",
                va="top", fontsize=13, weight="bold")
    fluence = folder.split(" - ")[1].split(" ions")[0]
    ax.set_title(f"{title}, {fluence} ions cm$^{{-2}}$", fontsize=11, weight="bold")

    stubs.append(f"{panel} {title}, {flnote}. Peak damage {yd.max():.3g} dpa at "
                 f"{xd[ipk]*scale:g} {xlab[7:-1]}; peak ion concentration "
                 f"{100*yi.max():.3g} at.% at {xi[jpk]*scale:g} {xlab[7:-1]}.")

fig.tight_layout()
p = os.path.join(OUT, "fig_SM_srim_profiles.png")
fig.savefig(p, bbox_inches="tight", dpi=PUB_DPI)
plt.close(fig)
print(f"[OK] {p}\n\nCaption stubs:")
for s in stubs:
    print(" ", s)
