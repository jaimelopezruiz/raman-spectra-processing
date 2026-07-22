"""Cross-spectrum area / intensity ratios relative to a pristine reference,
from BASELINE-CORRECTED RAW COUNTS (report §3/§4.3: total-area %, TO-area %,
D-area %, FTO-intensity %).

Why raw counts: comparing integrated areas between spectra is only meaningful on
a COMMON intensity scale. Per-spectrum normalisation (vector-0to1) rescales each
spectrum to its own maximum and therefore erases the very quantity being
compared — the change in area as the band shapes evolve. So every spectrum is
baseline-corrected (I-ModPoly) but NOT normalised, and areas are compared as
raw counts. (Direct integration is used, not summed fitted areas, because
raw-count curve-fitting does not converge with the pipeline's seed amplitudes —
the reason vector-0to1 exists for the fitting path.)

CAVEAT — semi-quantitative: raw-count comparison assumes equal light coupling
(laser power, focus, acquisition) across spectra. Where focus/power were not
locked (see report Fig 8) these ratios carry a coupling systematic that is NOT
captured by the baseline band; quote them as semi-quantitative and, if repeat
spectra per condition exist, use the spot-to-spot scatter as the error bar.

Uncertainty reported here = spread across I-ModPoly baseline orders (a
systematic band), which is the dominant *analysis* term; it does not include the
coupling systematic above.

Usage:
  python cross_spectrum_ratios.py --pristine input/Unirradiated.csv \
      "input/0 Ne 300 2.5.csv" "input/4 Si 750 2.5.csv" [...] \
      [--orders 4 5 6] [--out output/cross_ratios.csv]
"""
import argparse
import os
import sys

import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from preprocessing import preprocess

# Ratios are taken vs the pristine reference, so only bands present in pristine
# make sense here (the C-C D band is absent in pristine → use χ for that, not a
# vs-pristine ratio).
WINDOWS = {"total": (170, 2000), "TO": (700, 850)}
FTO_WINDOW = (783, 795)


def _integ(x, y, w):
    m = (x >= w[0]) & (x <= w[1])
    return float(np.trapezoid(y[m], x[m])) if m.sum() >= 2 else float("nan")


def _measure(path, order, crop):
    # Baseline-corrected RAW counts (normalisation='none').
    x, y = preprocess(path, crop[0], crop[1], imodpoly_order=order, imodpoly_tol=1e-3,
                      imodpoly_max_iter=100, normalisation="none", plot=False,
                      save_path=None, convert_wavelength_to_shift=False)
    m = {k: _integ(x, y, w) for k, w in WINDOWS.items()}
    fm = (x >= FTO_WINDOW[0]) & (x <= FTO_WINDOW[1])
    m["FTO_height"] = float(np.max(y[fm])) if fm.any() else float("nan")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="Sample spectra to compare to the pristine reference")
    ap.add_argument("--pristine", required=True, help="Pristine (unirradiated) reference spectrum")
    ap.add_argument("--orders", type=int, nargs="+", default=[4, 5, 6],
                    help="I-ModPoly baseline orders for the systematic band")
    ap.add_argument("--nominal", type=int, default=5)
    ap.add_argument("--crop", type=float, nargs=2, default=[170, 2000])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    keys = list(WINDOWS) + ["FTO_height"]
    ref = {o: _measure(args.pristine, o, args.crop) for o in args.orders}

    rows = []
    for s in args.samples:
        per_order = {o: _measure(s, o, args.crop) for o in args.orders}
        row = {"sample": os.path.basename(s)}
        for k in keys:
            vals = [per_order[o][k] / ref[o][k] for o in args.orders
                    if np.isfinite(per_order[o][k]) and ref[o].get(k)]
            if not vals:
                continue
            nom = per_order[args.nominal][k] / ref[args.nominal][k]
            row[f"{k}_pct"] = 100 * nom
            row[f"{k}_lo"] = 100 * min(vals)
            row[f"{k}_hi"] = 100 * max(vals)
        rows.append(row)

    print("Cross-spectrum ratios vs pristine (%), baseline-corrected RAW counts. "
          "Bands = baseline-order spread. SEMI-QUANTITATIVE (coupling caveat).\n")
    hdr = f"{'sample':16s}" + "".join(f"{k:>16}" for k in keys)
    print(hdr)
    for r in rows:
        line = f"{r['sample'][:16]:16s}"
        for k in keys:
            if f"{k}_pct" in r:
                line += f"{r[f'{k}_pct']:6.1f}[{r[f'{k}_lo']:.0f},{r[f'{k}_hi']:.0f}]".rjust(16)
            else:
                line += f"{'n/a':>16}"
        print(line)

    if args.out:
        import pandas as pd
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\n[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()
