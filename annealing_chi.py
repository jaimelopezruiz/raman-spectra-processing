"""Chemical-disorder ratio χ versus annealing temperature, by direct window
integration (report §2.3.3, Figs 12/17), with a baseline-order systematic band.

For each spectrum in an annealing series:
  χ = ∫(D-band window) / ∫(TO-band window)   of the baseline-corrected,
      vector-0to1-normalised spectrum   (derived_quantities.chi_integrated)

A fixed peak-fit config cannot track the annealing series (amorphous→recovered)
without mis-decomposing recovered spectra and blowing χ up, so χ is computed by
integration instead. Because the I-ModPoly baseline order is the dominant
uncertainty, χ is recomputed at several orders and the spread is reported as the
systematic band (there is no fit covariance here).

Usage:
  python annealing_chi.py "input/Annealing/Si 2.5dpa  300C" --out output/si300_2.5_chi_vs_T.csv
  python annealing_chi.py DIR [--orders 4 5 6] [--nominal 5]
         [--d-window 1340 1470] [--to-window 700 850] [--crop 170 2000]

Temperature is parsed from each filename (leading "<T>C", "<T>" after "dpa ", or
"-<T>S/E"). Files with no parseable temperature (RT / glass references) are
listed and skipped — check the printed temperatures before trusting the curve.
"""
import argparse
import os
import re
import sys

import numpy as np

# χ/cm⁻¹ in console output shouldn't crash non-UTF-8 consoles (e.g. cp1252)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from preprocessing import preprocess
from derived_quantities import chi_integrated, CHI_D_WINDOW, CHI_TO_WINDOW


def temp_from_name(path):
    """Best-effort annealing temperature (°C) from a filename, else None."""
    b = os.path.basename(path)
    m = re.match(r"\s*(\d{2,4})\s*C\b", b)            # "100C ...", "1000C ..."
    if m:
        return int(m.group(1))
    m = re.search(r"dpa\s+(\d{2,4})\b", b)            # "...dpa 100--" (Si 750 series)
    if m and 50 <= int(m.group(1)) <= 1500:
        return int(m.group(1))
    m = re.search(r"-\s*(\d{2,4})\s*[SE ]", b)         # "-200S", "-300E", "-400 "
    if m and 50 <= int(m.group(1)) <= 1500:
        return int(m.group(1))
    m = re.search(r"(\d{2,4})C\b", b)                  # "... 100C anneal ..." (Au series)
    if m and 50 <= int(m.group(1)) <= 1500:
        return int(m.group(1))
    return None


def collect(path):
    """Spectra in a directory, preferring .txt (WITec native) to avoid the
    .csv/.txt duplicates in some folders. A single file is returned as-is."""
    if not os.path.isdir(path):
        return [path]
    names = os.listdir(path)
    txt = [f for f in names if f.lower().endswith(".txt")]
    csv = [f for f in names if f.lower().endswith(".csv")]
    return sorted(os.path.join(path, f) for f in (txt or csv))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Annealing-series directory (or a single spectrum)")
    ap.add_argument("--orders", type=int, nargs="+", default=[4, 5, 6],
                    help="I-ModPoly baseline orders for the systematic band (default 4 5 6)")
    ap.add_argument("--nominal", type=int, default=5, help="Baseline order for the reported χ")
    ap.add_argument("--d-window", type=float, nargs=2, default=list(CHI_D_WINDOW))
    ap.add_argument("--to-window", type=float, nargs=2, default=list(CHI_TO_WINDOW))
    ap.add_argument("--crop", type=float, nargs=2, default=[170, 2000])
    ap.add_argument("--convert-wavelength", action="store_true",
                    help="x-axis is wavelength in nm (532 nm excitation), not Raman "
                         "shift — convert first (e.g. the Au 2.5e15 CSV exports)")
    ap.add_argument("--out", default=None, help="Write the χ-vs-T table to this CSV")
    args = ap.parse_args()

    rows, skipped = [], []
    for f in collect(args.path):
        T = temp_from_name(f)
        if T is None:
            skipped.append(os.path.basename(f))
            continue
        chis = {}
        for order in args.orders:
            try:
                x, y = preprocess(
                    f, args.crop[0], args.crop[1], imodpoly_order=order,
                    imodpoly_tol=1e-3, imodpoly_max_iter=100,
                    normalisation="vector-0to1", plot=False, save_path=None,
                    convert_wavelength_to_shift=args.convert_wavelength)
                chis[order] = chi_integrated(
                    x, y, tuple(args.d_window), tuple(args.to_window))["chi"]
            except Exception:
                chis[order] = float("nan")
        vals = [v for v in chis.values() if np.isfinite(v)]
        if not vals:
            continue
        nominal = chis.get(args.nominal)
        if nominal is None or not np.isfinite(nominal):
            nominal = float(np.median(vals))
        rows.append({
            "temp_C": T, "chi": nominal,
            "chi_min": min(vals), "chi_max": max(vals),
            "u_chi_baseline": (max(vals) - min(vals)) / 2.0,
            "file": os.path.basename(f),
        })

    rows.sort(key=lambda r: r["temp_C"])
    print(f"{'T(C)':>6} {'chi':>7} {'baseline band':>18}  file")
    for r in rows:
        print(f"{r['temp_C']:>6} {r['chi']:>7.3f}  [{r['chi_min']:.3f}, {r['chi_max']:.3f}]"
              f"   {r['file'][:44]}")
    if skipped:
        print(f"\n[skipped {len(skipped)} file(s) with no parseable temperature, e.g. {skipped[:3]}]")
    if args.out:
        import pandas as pd
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[OK] Wrote χ-vs-T table to: {args.out}")


if __name__ == "__main__":
    main()
