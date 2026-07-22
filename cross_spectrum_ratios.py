"""SiC crystallinity vs damage (report §3/§4.3).

PRIMARY METRIC — within-spectrum SiC crystallinity index (coupling-invariant):

    C = (sharp folded-TO peak height) / (integrated Si-C TO-band area, 700-850)

Both terms come from the SAME baseline-corrected spectrum, so laser coupling
(focus, power, acquisition) cancels exactly. C is high when a sharp crystalline
folded-TO dominates and collapses toward the disordered-band floor as damage
increases. It is monotonic with damage and stable across baseline orders
(verified 2026-07-22), and "crystallinity retained" = C_sample / C_pristine
reproduces the report's ~-87.5% folded-TO intensity loss without a common-scale
assumption. THIS is the metric to quote for the crystalline-degradation story;
the chemical (C-C) disorder story is carried by chi (derived_quantities), also
within-spectrum. Both are coupling-immune.

SECONDARY / ILLUSTRATIVE ONLY — vs-pristine absolute ratios (total-area %,
TO-area %, FTO-height %) from BASELINE-CORRECTED RAW COUNTS. These compare
integrated areas between DIFFERENT spectra, so they assume equal light coupling
across spectra. Focus/power were not locked (report Fig 8), and empirically the
absolute total-area ratio does NOT order by damage (least-damaged Si 750 2.5 =
210%, more-damaged Si 300 2.5 = 58%) -> it is dominated by measurement coupling,
not disorder. DECISION 2026-07-22 (JLR): do NOT quote these as quantitative
disorder metrics; kept here as an illustrative aside with the coupling caveat.
The baseline-order band reported for them is the analysis term only and does NOT
include the coupling systematic.

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

# Absolute vs-pristine ratios (ILLUSTRATIVE ONLY — see module docstring). Only
# bands present in pristine make sense here (the C-C D band is absent in pristine
# → use χ for that, not a vs-pristine ratio).
WINDOWS = {"total": (170, 2000), "TO": (700, 850)}
FTO_WINDOW = (783, 795)

# Within-spectrum crystallinity index C = FTO peak height / TO-band integral.
# Same-spectrum ratio → laser coupling cancels. FTO window spans the sharp
# folded-TO (~785 pristine, up-shifting to ~788 irradiated); TO window is the
# whole Si-C region incl. the broad disordered band it collapses into.
CRYST_FTO_WINDOW = (783, 800)
CRYST_TO_WINDOW = (700, 850)


def _integ(x, y, w):
    m = (x >= w[0]) & (x <= w[1])
    return float(np.trapezoid(y[m], x[m])) if m.sum() >= 2 else float("nan")


def crystallinity_index(x, y, fto=CRYST_FTO_WINDOW, to=CRYST_TO_WINDOW):
    """Within-spectrum SiC crystallinity index ×100 (coupling-invariant).

    C = 100 · max(y in FTO window) / ∫(y over TO window). High = sharp
    crystalline folded-TO dominates; low = collapsed into the disordered band.
    Independent of laser coupling because numerator and denominator are from the
    same spectrum. NaN if either window is empty.
    """
    fm = (x >= fto[0]) & (x <= fto[1])
    a_to = _integ(x, y, to)
    if not fm.any() or not np.isfinite(a_to) or a_to == 0:
        return float("nan")
    return 100.0 * float(np.max(y[fm])) / a_to


def _measure(path, order, crop):
    # Baseline-corrected RAW counts (normalisation='none').
    x, y = preprocess(path, crop[0], crop[1], imodpoly_order=order, imodpoly_tol=1e-3,
                      imodpoly_max_iter=100, normalisation="none", plot=False,
                      save_path=None, convert_wavelength_to_shift=False)
    m = {k: _integ(x, y, w) for k, w in WINDOWS.items()}
    fm = (x >= FTO_WINDOW[0]) & (x <= FTO_WINDOW[1])
    m["FTO_height"] = float(np.max(y[fm])) if fm.any() else float("nan")
    m["cryst"] = crystallinity_index(x, y)   # within-spectrum (coupling-invariant)
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
    samples_po = {s: {o: _measure(s, o, args.crop) for o in args.orders}
                  for s in args.samples}

    def band(vals):
        return (100 * min(vals), 100 * max(vals))  # already-fractional inputs → %

    # === PRIMARY: within-spectrum crystallinity index (coupling-invariant) ===
    c_ref_nom = ref[args.nominal]["cryst"]
    c_ref_band = [ref[o]["cryst"] for o in args.orders if np.isfinite(ref[o]["cryst"])]
    cryst_rows = [{
        "sample": os.path.basename(args.pristine) + " (pristine)",
        "C_index": c_ref_nom, "C_lo": min(c_ref_band), "C_hi": max(c_ref_band),
        "retained_pct": 100.0, "retained_lo": 100.0, "retained_hi": 100.0,
    }]
    for s in args.samples:
        po = samples_po[s]
        cvals = [po[o]["cryst"] for o in args.orders if np.isfinite(po[o]["cryst"])]
        rvals = [po[o]["cryst"] / ref[o]["cryst"] for o in args.orders
                 if np.isfinite(po[o]["cryst"]) and ref[o]["cryst"]]
        if not cvals:
            continue
        cryst_rows.append({
            "sample": os.path.basename(s),
            "C_index": po[args.nominal]["cryst"], "C_lo": min(cvals), "C_hi": max(cvals),
            "retained_pct": 100 * po[args.nominal]["cryst"] / ref[args.nominal]["cryst"],
            "retained_lo": 100 * min(rvals), "retained_hi": 100 * max(rvals),
        })

    print("PRIMARY — within-spectrum SiC crystallinity index C = FTO height / TO-band "
          "integral (COUPLING-INVARIANT).\nBands = baseline-order spread. Lower C = more "
          "damage; 'retained' = C/C_pristine.\n")
    print(f"{'sample':22s}{'C index':>18}{'crystallinity retained %':>28}")
    for r in cryst_rows:
        print(f"{r['sample'][:22]:22s}"
              f"{r['C_index']:7.3f} [{r['C_lo']:.3f},{r['C_hi']:.3f}]".rjust(18)
              + f"{r['retained_pct']:7.1f} [{r['retained_lo']:.1f},{r['retained_hi']:.1f}]".rjust(28))

    # === SECONDARY (ILLUSTRATIVE ONLY): absolute vs-pristine ratios ===
    rows = []
    for s in args.samples:
        per_order = samples_po[s]
        row = {"sample": os.path.basename(s)}
        for k in keys:
            vals = [per_order[o][k] / ref[o][k] for o in args.orders
                    if np.isfinite(per_order[o][k]) and ref[o].get(k)]
            if not vals:
                continue
            row[f"{k}_pct"] = 100 * per_order[args.nominal][k] / ref[args.nominal][k]
            row[f"{k}_lo"], row[f"{k}_hi"] = band(vals)
        rows.append(row)

    print("\n" + "=" * 78)
    print("ILLUSTRATIVE ONLY — absolute vs-pristine ratios (%), RAW counts. "
          "COUPLING-DOMINATED,\nNON-MONOTONIC in damage — do NOT quote as quantitative "
          "disorder metrics (see docstring).\n")
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
        # merge crystallinity (primary) + absolute ratios (illustrative) per sample
        abs_by_name = {r["sample"]: r for r in rows}
        merged = []
        for cr in cryst_rows:
            base = os.path.basename(cr["sample"].replace(" (pristine)", ""))
            merged.append({**cr, **{k: v for k, v in abs_by_name.get(base, {}).items()
                                    if k != "sample"}})
        pd.DataFrame(merged).to_csv(args.out, index=False)
        print(f"\n[OK] Wrote {args.out}  (crystallinity index = primary; absolute ratios = illustrative)")


if __name__ == "__main__":
    main()
