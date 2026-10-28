"""G-band window ratio psi versus annealing temperature (ticket C27), the same
construction as annealing_chi.py's chi, applied to the G window instead of D.

For each spectrum in an annealing series:
  psi = integral(G-band window) / integral(TO-band window)   of the
        baseline-corrected, vector-0to1-normalised spectrum
        (derived_quantities.chi_g_integrated)

ADDITIVE, NOT A REDEFINITION: chi_integrated / CHI_D_WINDOW / CHI_TO_WINDOW and
every existing output/*chi_vs_T.csv are untouched. This script only reuses
annealing_chi.temp_from_name/collect (no re-implementation) and writes to new
CSV files.

CAVEAT (see derived_quantities.chi_g_integrated docstring): unlike chi's D
window, which sits on an essentially empty floor in the pristine spectrum, the
G window (1560-1600) sits in the tail of the very broad D-band lineshape, so
psi > 0 even with NO real G-band component. psi is NOT a presence/absence test
by itself -- it must be read against the no-G-band floor + spread built by
g_band_floor.py from the specimens ticket C26 already fit-confirmed to have no
G-band component. This script only computes and tabulates psi; the
presence/absence verdict is applied in output/C27_annealing_g_window_audit.md.

Usage:
  python annealing_chi_g.py "input/Annealing/Si 2.5dpa  300C" --out output/si300_2.5_chi_g_vs_T.csv
  python annealing_chi_g.py DIR [--orders 4 5 6] [--nominal 5]
         [--g-window 1560 1600] [--to-window 700 850] [--crop 170 2000]
"""
import argparse
import os
import sys

import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from preprocessing import preprocess
from derived_quantities import chi_g_integrated, CHI_G_WINDOW, CHI_TO_WINDOW
from annealing_chi import temp_from_name, collect


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Annealing-series directory (or a single spectrum)")
    ap.add_argument("--orders", type=int, nargs="+", default=[4, 5, 6],
                    help="I-ModPoly baseline orders for the systematic band (default 4 5 6)")
    ap.add_argument("--nominal", type=int, default=5, help="Baseline order for the reported psi")
    ap.add_argument("--g-window", type=float, nargs=2, default=list(CHI_G_WINDOW))
    ap.add_argument("--to-window", type=float, nargs=2, default=list(CHI_TO_WINDOW))
    ap.add_argument("--crop", type=float, nargs=2, default=[170, 2000])
    ap.add_argument("--convert-wavelength", action="store_true",
                    help="x-axis is wavelength in nm (532 nm excitation), not Raman "
                         "shift -- convert first (e.g. the Au 2.5e15 CSV exports)")
    ap.add_argument("--out", default=None, help="Write the psi-vs-T table to this CSV")
    args = ap.parse_args()

    rows, skipped = [], []
    for f in collect(args.path):
        T = temp_from_name(f)
        if T is None:
            skipped.append(os.path.basename(f))
            continue
        psis = {}
        for order in args.orders:
            try:
                x, y = preprocess(
                    f, args.crop[0], args.crop[1], imodpoly_order=order,
                    imodpoly_tol=1e-3, imodpoly_max_iter=100,
                    normalisation="vector-0to1", plot=False, save_path=None,
                    convert_wavelength_to_shift=args.convert_wavelength)
                psis[order] = chi_g_integrated(
                    x, y, tuple(args.g_window), tuple(args.to_window))["chi_g"]
            except Exception:
                psis[order] = float("nan")
        vals = [v for v in psis.values() if np.isfinite(v)]
        if not vals:
            continue
        nominal = psis.get(args.nominal)
        if nominal is None or not np.isfinite(nominal):
            nominal = float(np.median(vals))
        rows.append({
            "temp_C": T, "chi_g": nominal,
            "chi_g_min": min(vals), "chi_g_max": max(vals),
            "u_chi_g_baseline": (max(vals) - min(vals)) / 2.0,
            "file": os.path.basename(f),
        })

    rows.sort(key=lambda r: r["temp_C"])
    print(f"{'T(C)':>6} {'psi':>7} {'baseline band':>18}  file")
    for r in rows:
        print(f"{r['temp_C']:>6} {r['chi_g']:>7.4f}  [{r['chi_g_min']:.4f}, {r['chi_g_max']:.4f}]"
              f"   {r['file'][:44]}")
    if skipped:
        print(f"\n[skipped {len(skipped)} file(s) with no parseable temperature, e.g. {skipped[:3]}]")
    if args.out:
        import pandas as pd
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[OK] Wrote psi-vs-T table to: {args.out}")


if __name__ == "__main__":
    main()
