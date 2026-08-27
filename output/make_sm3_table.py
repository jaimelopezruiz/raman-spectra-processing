"""SM.3 — supplementary fitted-parameters table.

Concatenates the per-spectrum *_peak_parameters.csv files (regenerated with the
current pipeline: analytic Voigt/BWF areas, duplicate-voigt fix) into one
supplementary-ready table: per condition, per peak — assignment, centre / FWHM /
area each with 1-sigma, and bound/seed flags. Conditions ordered as the SM.1
panels (a)-(e), plus Unirradiated appended as the reference condition.

Also cross-checks that the R2/RMSE annotated in fig_SM1_curvefits.png (from
make_sm1_curvefits.py, which refits live) match the refreshed
*_fit_statistics.csv on disk.

Run from the repo root: .venv\\Scripts\\python.exe output\\make_sm3_table.py
"""
import os
import sys

import pandas as pd

sys.stdout.reconfigure(errors="replace")

OUT = "output"
# (panel, condition label, per-spectrum file stem) — SM.1 order, reference last
CONDITIONS = [
    ("(a)", "Si 0.25 dpa / 750 °C", "3 Si 750 0.25"),
    ("(b)", "Si 2.5 dpa / 750 °C", "4 Si 750 2.5"),
    ("(c)", "Si 0.25 dpa / 300 °C", "2 Si 300 0.25"),
    ("(d)", "Si 2.5 dpa / 300 °C", "1 Si 300 2.5"),
    ("(e)", "Ne 2.5 dpa / 300 °C", "0 Ne 300 2.5"),
    ("ref", "Unirradiated", "Unirradiated"),
]
KEEP = ["Peak", "Model", "Assignment", "Region (cm⁻¹)",
        "Center (cm⁻¹)", "Center 1σ (cm⁻¹)", "FWHM (cm⁻¹)", "FWHM 1σ (cm⁻¹)",
        "Area", "Area 1σ", "Flags"]

frames = []
for panel, label, stem in CONDITIONS:
    df = pd.read_csv(os.path.join(OUT, f"{stem}_peak_parameters.csv"))
    df = df[KEEP].copy()
    df.insert(0, "Condition", label)
    df.insert(1, "SM.1 panel", panel)
    frames.append(df)

table = pd.concat(frames, ignore_index=True)
table.to_csv(os.path.join(OUT, "sm_fitted_parameters.csv"), index=False,
             encoding="utf-8-sig")
print(f"[OK] output/sm_fitted_parameters.csv  ({len(table)} peak rows, "
      f"{len(CONDITIONS)} conditions)")

# --- cross-check: SM.1 annotations vs refreshed fit statistics ---
print("\nPer-region fit statistics (compare with fig_SM1_curvefits.png boxes):")
for panel, label, stem in CONDITIONS:
    fs = pd.read_csv(os.path.join(OUT, f"{stem}_fit_statistics.csv"))
    for _, r in fs.iterrows():
        print(f"  {panel} {label:22s} {r['region']:>12s}:  "
              f"R2 = {r['R2']:.3f}   RMSE = {r['RMSE']:.4f}")
