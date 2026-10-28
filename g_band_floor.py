"""No-G-band psi floor + known-G-band psi calibration (ticket C27).

psi = integral(G window 1560-1600) / integral(TO window 700-850) is nonzero
even with no real G-band component (it sits in the broad D-band tail — see
derived_quantities.chi_g_integrated). To read an annealing-step psi value at
all, we need to know what psi looks like for spectra ticket C26 already
fit-confirmed to have NO G-band component, and for spectra it confirmed DO
have one. This script computes psi (baseline orders 4,5,6, nominal 5, same as
annealing_chi_g.py) for those six survey spectra and writes the two
calibration groups to output/g_band_floor.csv.

NO-G-BAND group (C26 §1-2, output/C26_dg_band_audit.md):
  Unirradiated            — no C-C band seeded/fitted at all
  Si 0.25 dpa / 750 C     — candidate 2nd-order-Si-C component collapsed to amp~0, flagged absent
  Si 2.5 dpa / 750 C      — region never fitted past 1450 cm-1; raw check: no rebound above noise
  Si 0.25 dpa / 300 C     — no G peak seeded (deliberate physics call); raw check: no rebound

G-BAND-PRESENT group (C26 §1-2):
  Si 2.5 dpa / 300 C      — fitted BWF G-band, SNR ~18 (amp/fit-RMSE)
  Ne 2.5 dpa / 300 C      — fitted BWF G-band, SNR ~24 (amp/fit-RMSE)

Usage: python g_band_floor.py [--out output/g_band_floor.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from preprocessing import preprocess
from derived_quantities import chi_g_integrated

NO_G = [
    ("Unirradiated", "input/Unirradiated.csv"),
    ("Si 0.25 dpa 750 C", "input/3 Si 750 0.25.csv"),
    ("Si 2.5 dpa 750 C", "input/4 Si 750 2.5.csv"),
    ("Si 0.25 dpa 300 C", "input/2 Si 300 0.25.csv"),
]
G_PRESENT = [
    ("Si 2.5 dpa 300 C", "input/1 Si 300 2.5.csv"),
    ("Ne 2.5 dpa 300 C", "input/0 Ne 300 2.5.csv"),
]
ORDERS = [4, 5, 6]
NOMINAL = 5
CROP = (170.0, 2000.0)


def psi_of(path, orders=ORDERS, nominal=NOMINAL):
    vals = {}
    for order in orders:
        x, y = preprocess(path, CROP[0], CROP[1], imodpoly_order=order,
                          imodpoly_tol=1e-3, imodpoly_max_iter=100,
                          normalisation="vector-0to1", plot=False, save_path=None,
                          convert_wavelength_to_shift=False)
        vals[order] = chi_g_integrated(x, y)["chi_g"]
    v = list(vals.values())
    return vals[nominal], min(v), max(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/g_band_floor.csv")
    args = ap.parse_args()

    rows = []
    for group, items in [("no_G_band", NO_G), ("G_band_present", G_PRESENT)]:
        for label, path in items:
            nominal, lo, hi = psi_of(path)
            rows.append({"group": group, "label": label, "file": path,
                        "psi": nominal, "psi_lo": lo, "psi_hi": hi})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    floor = df[df.group == "no_G_band"]
    floor_mean = floor["psi"].mean()
    floor_max = floor["psi_hi"].max()  # most conservative: highest no-G-band psi at any baseline order
    floor_spread = floor["psi"].std(ddof=1)
    print(f"\nno-G-band floor: mean psi = {floor_mean:.4f}, "
          f"sample std across the {len(floor)} specimens = {floor_spread:.4f}, "
          f"max across specimens+baseline orders = {floor_max:.4f}")
    print("(a step's psi is treated as a candidate G-band detection only if its own "
          "baseline-order band does not overlap this floor+spread — see C27 report)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()
