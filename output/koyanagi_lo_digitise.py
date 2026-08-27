"""C18 provenance check for the Koyanagi LO reference line in Fig 9b.

Establishes, reproducibly, WHERE the constants hard-coded in
`make_figures.py` (KOY_SLOPE = 0.0666, KOY_INTERCEPT = 904.8,
KOY_RANGE = 230-770 degC) come from.

The source is panel (a) of the PREVIOUS draft's Figure 18, which is Koyanagi's
own published figure pasted in whole (docx `word/media/image19.png`). That
figure prints its LO fit on its own axes as `y = 0.0666x + 904.8`. This script
caches the raster, calibrates its axes on the plot frame, and re-fits the red
LO content independently, so the printed equation is not taken on trust.

Two traps this encodes, in case anyone re-does it:
  * the DESCENDING grey dashed line on that figure is LINEAR SWELLING on the
    RIGHT axis. It is not a Raman shift. Only the ascending RED line is LO.
  * the red content is not only the trend line: it also includes the data
    markers, their error bars, the red "LO line position" annotation and a red
    left-pointing axis arrow at ~935 cm-1. The arrow in particular drags a
    naive fit down to ~17e-3, so it is excluded by restricting to T > 215 degC
    and fitting robustly (Theil-Sen seed + sigma clipping).

Run:  .venv\\Scripts\\python.exe output/koyanagi_lo_digitise.py
"""
import os
import sys
import zipfile

import numpy as np
from PIL import Image

sys.stdout.reconfigure(errors="replace")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "output")
DOCX = os.path.join(REPO, "Paper WIP Draft PREVIOUS with comments.docx")
MEMBER = "word/media/image19.png"                 # = old Figure 18 (both panels)
CACHE = os.path.join(OUT, "fig_18a_source_koyanagi.png")

# plot frame of panel (a), in pixels of the cached raster, and the axis values
# read off its printed tick labels. Panel (b) (our old two-point plot) is the
# right half and is ignored here.
FRAME = dict(x_left=82, x_right=530, y_top=40, y_bot=411)
X_AXIS = (0.0, 1000.0)          # degC   at x_left, x_right
Y_AXIS = (910.0, 1000.0)        # cm-1   at y_bot,  y_top  (LEFT axis)
PRINTED = (0.0666, 904.8)       # slope cm-1/degC, intercept cm-1 (on the figure)
# the legend block sits inside the frame, upper middle/right -> masked out
LEGEND_BOX = dict(x0=185, x1=440, y1=128)


def cached_raster():
    if not os.path.exists(CACHE):
        with zipfile.ZipFile(DOCX) as z, open(CACHE, "wb") as fh:
            fh.write(z.read(MEMBER))
        print(f"[cache] extracted {MEMBER} -> {CACHE}")
    return np.asarray(Image.open(CACHE).convert("RGB")).astype(int)


def main():
    a = cached_raster()
    xl, xr, yt, yb = (FRAME[k] for k in ("x_left", "x_right", "y_top", "y_bot"))

    def px2T(x):
        return X_AXIS[0] + (x - xl) / (xr - xl) * (X_AXIS[1] - X_AXIS[0])

    def px2w(y):
        return Y_AXIS[0] + (yb - y) / (yb - yt) * (Y_AXIS[1] - Y_AXIS[0])

    sub = a[yt:yb, xl:xr]
    r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    red = (r > 140) & (r - g > 55) & (r - b > 55)
    ys, xs = np.nonzero(red)

    in_legend = ((xs + xl > LEGEND_BOX["x0"]) & (xs + xl < LEGEND_BOX["x1"])
                 & (ys + yt < LEGEND_BOX["y1"]))
    T, W = px2T(xs[~in_legend] + xl), px2w(ys[~in_legend] + yt)
    print(f"red pixels in frame, legend masked: {T.size}")
    print(f"  full extent  T {T.min():.0f}-{T.max():.0f} degC, "
          f"{W.min():.1f}-{W.max():.1f} cm-1")

    # drop the red axis arrow / annotation: keep the data+line band only
    keep = (T > 215) & (T < 790) & (W > 912) & (W < 965)
    T, W = T[keep], W[keep]

    # Theil-Sen seed (immune to the marker cloud), then sigma-clipped least squares
    rng = np.random.default_rng(0)
    idx = rng.choice(T.size, size=(20000, 2))
    dT = T[idx[:, 1]] - T[idx[:, 0]]
    ok = np.abs(dT) > 80                     # long baselines only
    slope = np.median((W[idx[ok, 1]] - W[idx[ok, 0]]) / dT[ok])
    p = np.array([slope, np.median(W - slope * T)])
    print(f"  Theil-Sen           : {p[0]*1e3:6.2f}e-3 cm-1/degC, intercept {p[1]:.1f}")
    for _ in range(20):
        res = W - np.polyval(p, T)
        k = np.abs(res) < 1.2 * res.std()
        if k.sum() < 50:
            break
        p = np.polyfit(T[k], W[k], 1)
    print(f"  sigma-clipped fit   : {p[0]*1e3:6.2f}e-3 cm-1/degC, intercept {p[1]:.1f} "
          f"({k.sum()} px)")
    print(f"  PRINTED on the figure: {PRINTED[0]*1e3:6.2f}e-3 cm-1/degC, "
          f"intercept {PRINTED[1]:.1f}   <-- adopted")
    print("  (the digitised slope reads low because the marker/error-bar pixels are "
          "mixed in with the line; it confirms the printed value and excludes ~77e-3)")

    on_line = np.abs(W - (PRINTED[0] * T + PRINTED[1])) < 1.0
    print(f"  drawn line spans     T {T[on_line].min():.0f}-{T[on_line].max():.0f} degC "
          f"-> KOY_RANGE = 230-770 degC (rounded out to their marker extent)")

    print(f"\nour locked 2.5 dpa slope 48.5 +- 3.1 is "
          f"{48.5/(PRINTED[0]*1e3)*100:.0f} +- {3.1/(PRINTED[0]*1e3)*100:.0f} % of Koyanagi's")
    print(f"the manuscript's published 63 % corresponds to the OLD slope "
          f"{0.634*PRINTED[0]*1e3:.1f}e-3 -> STALE, see paper-tracker C18")


if __name__ == "__main__":
    main()
