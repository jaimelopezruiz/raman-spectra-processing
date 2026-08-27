"""Per-point FTO fit uncertainties for the 0.25 dpa / 750 C stress map (Fig 13).

JLR's Excel peak positions (Stress Mapping.xlsx; manual WiRE cursor read-offs,
JLR 2026-07-29, stated in report §2.3.4) stay THE data — the map points are
fitted here only to extract a per-point statistical uncertainty u(w_i) on the
FTO centre (single Lorentzian + constant over a window around the FTO).

u(w0) POLICY (JLR 2026-07-29): the locked reference values u(w0) = 0.44 cm-1 /
floor ±326 MPa stand. THIS recipe applied to the same 4 valid unirradiated
references gives a scatter of 0.67 cm-1 (would imply floor ~±500 MPa) — recorded
here for transparency, deliberately NOT adopted; the reference scatter is
recipe-dependent and 0.44 is the published anchor.

Writes: output/stress_point_fits.csv  (one row per Excel map position)
Run from the repo root: .venv\\Scripts\\python.exe output\\fit_stress_points.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.stdout.reconfigure(errors="replace")

IRR_DIR = r"Temporal Claude Context/Irradiated Data Points Stress Mapping"
# Unirradiated reference point spectra, used ONLY for the transparency scatter
# quoted in the docstring (0.67 cm-1) — not for any published value. They live
# outside the repo; override with RAMAN_UNIRR_DIR if they move. Written via
# expanduser rather than a literal home path so no username is committed.
UNIRR_DIR = os.environ.get(
    "RAMAN_UNIRR_DIR",
    os.path.join(os.path.expanduser("~"), "Downloads", "OneDrive_1_22-07-2026"))
EXCEL_POS = [785.14, 786.0, 788.7, 788.75, 785.0, 786.9, 786.93, 788.72,
             786.93, 786.95, 786.95, 787.71]
WIN = (776.0, 800.0)


def lor(x, a, c, w, b):
    return a * w**2 / ((x - c)**2 + w**2) + b


def fit_fto(path):
    d = pd.read_csv(path, sep=r"\s+", header=None, engine="python").to_numpy(float)
    x, y = d[:, 0], d[:, 1]
    m = (x >= WIN[0]) & (x <= WIN[1])
    x, y = x[m], y[m]
    b0 = np.percentile(y, 10)
    c0 = x[np.argmax(y)]
    try:
        popt, pcov = curve_fit(lor, x, y, p0=[y.max() - b0, c0, 3.0, b0],
                               bounds=([0, WIN[0], 0.5, 0], [np.inf, WIN[1], 20, np.inf]),
                               maxfev=20000)
    except Exception:
        return dict(ok=False, center=np.nan, u_center=np.nan, fwhm=np.nan, snr=np.nan)
    perr = np.sqrt(np.diag(pcov))
    resid = y - lor(x, *popt)
    snr = popt[0] / max(np.std(resid), 1e-9)
    ok = (np.isfinite(perr[1]) and perr[1] < 1.0 and 1.0 < 2 * popt[2] < 30
          and snr > 5 and WIN[0] + 1 < popt[1] < WIN[1] - 1)
    return dict(ok=bool(ok), center=popt[1], u_center=perr[1],
                fwhm=2 * popt[2], snr=snr)


# --- unirradiated reference cross-check (expect u(w0) ~ 0.44 from 4 valid) ---
print("Unirradiated references:")
ref_centers = []
for f in sorted(glob.glob(os.path.join(UNIRR_DIR, "Unirr*.csv"))):
    r = fit_fto(f)
    print(f"  {os.path.basename(f):18s} center={r['center']:.2f}  u={r['u_center']:.3f}"
          f"  FWHM={r['fwhm']:.1f}  SNR={r['snr']:.0f}  {'VALID' if r['ok'] else 'EXCLUDED'}")
    if r["ok"]:
        ref_centers.append(r["center"])
print(f"  -> std of {len(ref_centers)} valid refs = {np.std(ref_centers, ddof=1):.3f} cm-1"
      f" (locked u(w0) = 0.44)\n")

# --- irradiated map points ---
fits = []
for f in sorted(glob.glob(os.path.join(IRR_DIR, "*.csv"))):
    r = fit_fto(f)
    r["file"] = os.path.basename(f)
    fits.append(r)
    print(f"  {r['file']:14s} center={r['center']:8.2f}  u={r['u_center']:.3f}"
          f"  FWHM={r['fwhm']:5.1f}  SNR={r['snr']:6.0f}  {'ok' if r['ok'] else 'WEAK/FAIL'}")

# --- rank-order assignment ---
# No estimator reproduces JLR's absolute positions (single-Lorentzian envelope
# centres sit ~2 cm-1 low on the disorder-dominated points; a two-component
# sharp Lorentzian sits ~2.5 high; raw apexes are numerically unstable), so a
# nearest-value match is impossible. Both lists ARE monotone by family — the 3
# sharp, near-reference spectra (Irr 2*) <-> the 3 near-788.7 positions, the
# disorder-dominated spectra <-> the shifted positions — so u(w_i) is assigned
# by rank: k-th lowest fitted centre -> k-th lowest Excel position. u(w_i) is
# the fit's centre uncertainty in THAT spectrum; the Excel position stays the
# data. The mapping affects only which u lands on which point, and u is similar
# within a family, so mis-pairing inside a cluster is inconsequential.
fits_sorted = sorted(fits, key=lambda r: r["center"])
rows = []
for pos, r in zip(sorted(EXCEL_POS), fits_sorted):
    rows.append(dict(excel_pos=pos, u_wi=r["u_center"], matched_file=r["file"],
                     fit_center=r["center"], fit_ok=r["ok"], source="rank-order fit"))

df = pd.DataFrame(rows)
df.to_csv("output/stress_point_fits.csv", index=False)
print("\nrank-matched (excel_pos <- file, u_wi):")
for r in rows:
    print(f"  {r['excel_pos']:7.2f} <- {r['matched_file']:14s} u_wi={r['u_wi']:.3f}"
          f" (fit c={r['fit_center']:.2f}{'' if r['fit_ok'] else ', broad-envelope'})")
print("\n[OK] output/stress_point_fits.csv")
