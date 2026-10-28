"""C-C region (D+G band) peak fit across the Au 2.5e15 RT annealing series.

Built for ticket C29 to check R24 ("Au-implanted specimen develops both D and
G bands after annealing") by fit, the same way C26/C27 checked the Si/Ne
series -- C27 could not run this check at all because no tracked peak-fit
config existed for the Au single-crystal spectra
(output/C27_annealing_g_window_audit.md, section D).

Uses params/configs/au_singlecrystal_cc.yaml (new, C29) -- a fixed two-peak
(D lorentz + G BWF) config covering only the C-C window [1150, 1750] cm^-1,
validated against the Au RT-after spectrum (see that config's header comment
and output/C29_au_cc_peakfit_audit.md). The SAME fixed peak list is fit to
every stage, exactly as C27's method B did for Si/Ne: a peak whose amplitude
collapses to ~0 or whose FWHM degenerates to a narrow "needle" is evidence of
absence, not a fit failure to work around.

File selection reuses output/annealing_series.usable_spectra (drops the
narrow-window/stitched Au exports and any free-Si-matrix spot), the same
filter output/make_au_chi_rt.py uses for the published RT chi endpoints --
this is additive and does not touch that script or its output.

Usage:
  .venv\\Scripts\\python.exe au_cc_peakfit.py
  .venv\\Scripts\\python.exe au_cc_peakfit.py --out output/au_cc_peakfit_vs_T.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from preprocessing import preprocess
from curve_fitting import fit_peaks_regionwise
from config import load_sample_config
from annealing_series import SERIES, index_series, usable_spectra, stage_order, stage_label

CONFIG_PATH = os.path.join("params", "configs", "au_singlecrystal_cc.yaml")

# Fit-quality bar, matching the confirmed-positive standard C26/C27 used
# (C27 audit, section C): a real band has amp several sigma from zero and a
# FWHM in the same range as the confirmed-positive survey fits, not a
# degenerate 2-9 cm^-1 needle.
MIN_SIGMA = 3.0
G_FWHM_RANGE = (40.0, 160.0)   # C26/C27 confirmed-positive G-bands: 90-113 cm^-1
D_FWHM_RANGE = (40.0, 320.0)   # C26 survey D-bands ranged 90-230 cm^-1 FWHM


def verdict(amp, amp_err, fwhm_val, fwhm_err, flags, fwhm_range):
    if "amp~0 (peak effectively absent)" in flags:
        return "absent (amp~0)"
    if "amp at upper bound (2x seed)" in flags:
        return "undetermined (amp pinned at fit bound, not converged)"
    drifted = "centre drifted" in flags or "centre at +/- tolerance bound" in flags
    if amp_err <= 0 or not np.isfinite(amp_err):
        return "undetermined (no error estimate)"
    sigma = amp / amp_err
    too_narrow = fwhm_val < fwhm_range[0]
    too_broad = fwhm_val > fwhm_range[1]
    poorly_constrained = (not np.isfinite(fwhm_err)) or (fwhm_err > 0.6 * max(fwhm_val, 1e-9))
    if drifted and sigma >= MIN_SIGMA and not too_narrow and not too_broad and not poorly_constrained:
        return "undetermined (centre drifted from the literature window -- not clearly the labelled band; see curve_fitting.py bound-flag docstring)"
    if sigma >= MIN_SIGMA and not too_narrow and not too_broad and not poorly_constrained:
        return "PRESENT (credible)"
    if sigma < 2.0:
        return "absent/undetermined (amp <2 sigma from 0)"
    if too_narrow:
        return f"undetermined (needle FWHM {fwhm_val:.1f}, not a real band)"
    if too_broad:
        return f"undetermined (FWHM {fwhm_val:.1f} railed broad/bound-like, not a resolved band)"
    return "undetermined (marginal: amp {:.1f} sigma, FWHM err {:.0%})".format(
        sigma, fwhm_err / max(fwhm_val, 1e-9))


def fit_one(path, cfg, convert_wavelength):
    pre = cfg["preprocessing"]
    x, y = preprocess(
        path, pre.get("crop_min", 170), pre.get("crop_max", 2000),
        imodpoly_order=pre.get("baseline_order", 5), imodpoly_tol=1e-3,
        imodpoly_max_iter=100, normalisation=pre.get("normalisation", "vector-0to1"),
        plot=False, save_path=None, convert_wavelength_to_shift=convert_wavelength,
    )
    center_tolerance = cfg["fitting"].get("center_tolerance", 30)
    _, _, peak_params, fit_stats, _ = fit_peaks_regionwise(x, y, cfg["regions"], center_tolerance)
    stats = fit_stats[0]
    by_band = {}
    for p in peak_params:
        key = "D" if "D-band" in p["assignment"] else "G" if "G-band" in p["assignment"] else None
        if key:
            by_band[key] = p
    return by_band, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/au_cc_peakfit_vs_T.csv")
    args = ap.parse_args()

    cfg = load_sample_config(CONFIG_PATH)
    spec = SERIES["au"]
    stages, skipped = index_series("au")
    if skipped:
        print(f"[!] {len(skipped)} Au file(s) with no parseable stage: {skipped}")

    rows = []
    header = (f"{'stage':>10} {'file':<46} {'R2':>6} {'RMSE':>7} | "
              f"{'D amp':>7}+-{'err':<6} {'D FWHM':>7}+-{'err':<6} {'D verdict':<28} | "
              f"{'G amp':>7}+-{'err':<6} {'G FWHM':>7}+-{'err':<6} {'G verdict':<28}")
    print(header)

    for stage in stage_order(stages):
        kept, rejected = usable_spectra(stages[stage], spec["convert_wavelength"], (200.0, 1900.0))
        for p, why in rejected:
            print(f"  [dropped] {stage_label(stage)}: {os.path.basename(p)[:44]} -- {why}")
        for path in kept:
            by_band, stats = fit_one(path, cfg, spec["convert_wavelength"])
            d, g = by_band.get("D"), by_band.get("G")
            d_verdict = verdict(d["amp"], d["amp_err"], d["FWHM"], d["FWHM_err"], d["flags"], D_FWHM_RANGE) if d else "not fit"
            g_verdict = verdict(g["amp"], g["amp_err"], g["FWHM"], g["FWHM_err"], g["flags"], G_FWHM_RANGE) if g else "not fit"
            row = {
                "stage": stage_label(stage), "file": os.path.basename(path),
                "R2": stats["R2"], "RMSE": stats["RMSE"],
                "D_amp": d["amp"], "D_amp_err": d["amp_err"],
                "D_FWHM": d["FWHM"], "D_FWHM_err": d["FWHM_err"],
                "D_center": d["mu"], "D_center_err": d["mu_err"],
                "D_flags": d["flags"], "D_verdict": d_verdict,
                "G_amp": g["amp"], "G_amp_err": g["amp_err"],
                "G_FWHM": g["FWHM"], "G_FWHM_err": g["FWHM_err"],
                "G_center": g["mu"], "G_center_err": g["mu_err"],
                "G_flags": g["flags"], "G_verdict": g_verdict,
            }
            rows.append(row)
            amp_rmse = d["amp"] / stats["RMSE"] if stats["RMSE"] else float("nan")
            g_amp_rmse = g["amp"] / stats["RMSE"] if stats["RMSE"] else float("nan")
            print(f"{stage_label(stage):>10} {os.path.basename(path)[:46]:<46} "
                  f"{stats['R2']:>6.3f} {stats['RMSE']:>7.4f} | "
                  f"{d['amp']:>7.4f}+-{d['amp_err']:<6.4f} {d['FWHM']:>7.1f}+-{d['FWHM_err']:<6.1f} {d_verdict:<28} | "
                  f"{g['amp']:>7.4f}+-{g['amp_err']:<6.4f} {g['FWHM']:>7.1f}+-{g['FWHM_err']:<6.1f} {g_verdict:<28}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n[OK] Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
