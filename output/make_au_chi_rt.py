"""Room-temperature chi endpoints for the Au annealing series -> output/au_chi_RT.csv.

WHY THIS EXISTS. Figure 10(a) plots the Au series' RT-before and RT-after chi
values, and Section 3.2 quotes them (RT before 1.56 [1.52, 1.74]; RT after
0.37-0.48). They were produced ad hoc in an earlier session and the CSV was the
only record: `annealing_chi.py` keys its rows by temperature and cannot emit a
"stage" column, so nothing in the repository could regenerate this file. A clean
clone therefore could not rebuild Figure 10(a). This script closes that gap.

ESTIMATOR. Identical to annealing_chi.py: crop 170-2000, I-ModPoly baseline at
orders 4/5/6, vector-0to1 normalisation, chi = integrated D band (1340-1470)
over integrated TO band (700-850). Order 5 is the reported value; the spread
across the three orders is the baseline band, and half of it is quoted as
u_chi_baseline. The Au files are wavelength-axis exports, so
convert_wavelength_to_shift is on, as everywhere else for this series.

SELECTION. Delegated to annealing_series.usable_spectra, the same rule the
figures use, rather than a hard-coded file list: spectra whose axis does not
span 200-1900 cm-1 are dropped. That removes the narrow-window exports (01, 33,
35-37: 217-1338 cm-1) and the stitched SpecS/DU970P exports (02, 34: from
308 cm-1), leaving 03 for RT-before and 30-32 for RT-after — which is exactly
the set the published CSV contains.

Run from the repo root:  .venv\\Scripts\\python.exe output\\make_au_chi_rt.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(errors="replace")

from preprocessing import preprocess
from derived_quantities import chi_integrated, CHI_D_WINDOW, CHI_TO_WINDOW
from annealing_series import SERIES, index_series, usable_spectra

OUT = "output"
NAME = "au_chi_RT.csv"
ORDERS = (4, 5, 6)
NOMINAL = 5
CROP = (170.0, 2000.0)
STAGES = ("RT before", "RT after")


def chi_for(path, order, convert_wavelength):
    x, y = preprocess(path, CROP[0], CROP[1], imodpoly_order=order,
                      imodpoly_tol=1e-3, imodpoly_max_iter=100,
                      normalisation="vector-0to1", plot=False, save_path=None,
                      convert_wavelength_to_shift=convert_wavelength)
    return chi_integrated(x, y, CHI_D_WINDOW, CHI_TO_WINDOW)["chi"]


def main():
    spec = SERIES["au"]
    stages, _ = index_series("au")
    rows = []
    for stage in STAGES:
        kept, rejected = usable_spectra(stages.get(stage, []),
                                        spec["convert_wavelength"], CROP)
        for p, why in rejected:
            print(f"  [dropped] {stage}: {os.path.basename(p)[:44]} — {why}")
        for p in kept:
            vals = [chi_for(p, o, spec["convert_wavelength"]) for o in ORDERS]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            nominal = chi_for(p, NOMINAL, spec["convert_wavelength"])
            rows.append({
                "stage": stage,
                "chi": nominal,
                "chi_min": min(vals),
                "chi_max": max(vals),
                "u_chi_baseline": (max(vals) - min(vals)) / 2.0,
                "file": os.path.basename(p),
            })
            print(f"  {stage:>10}  chi = {nominal:.4f}  "
                  f"[{min(vals):.4f}, {max(vals):.4f}]   {os.path.basename(p)[:44]}")

    path = os.path.join(OUT, NAME)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[OK] {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
