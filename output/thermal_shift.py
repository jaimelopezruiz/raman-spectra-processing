"""Si-C band positions at temperature vs at room temperature (ticket C13).

Puts numbers on the manuscript claim that spectra acquired at 1000 C "shifted to
lower wavenumbers relative to the room-temperature spectra, consistent with
thermal expansion": which Si-C features move down, and by how many cm-1 +- 1 sigma.

⚠ DATA SCOPE. The Experiment-1 dataset (single-crystal 6H-SiC, isothermal 1000 C
anneal in air; the ~11 spectra plotted as Figure 10a) is NOT in this repository or
anywhere on this machine — only the Origin raster survives. What IS available is
the in-situ step-wise annealing of Experiment 3 (`input/Annealing/`), which
measures the same physical effect on the polycrystalline RB-SiC specimens: a
spectrum recorded through the furnace window at each hold temperature, plus
room-temperature references before and (for the two Si specimens) after.

WHAT IS MEASURED. Two landmarks of the Si-C envelope, both decomposition-free,
because a fixed multi-peak decomposition cannot track amorphous→recovered spectra
(the locked project decision behind the integrated chi estimator):

  FTO apex     least-squares parabola through the points within +-PARAB_HW of the
               maximum in TO_SEARCH. Same philosophy as the locked
               `derived_quantities.lo_band_position` used for Fig 9b. u from a
               residual bootstrap, the I-ModPoly order 4/5/6 spread, and the
               scatter of repeat spectra where a stage has them.
  LO edge      high-frequency half-height edge of the Si-C envelope. The folded-LO
               mode is NOT separately resolved in these polycrystalline in-situ
               spectra (890-960 cm-1 is a plateau/shoulder, so an apex there
               tracks noise), but the half-height fall-off of the envelope — which
               the folded-LO dominates — is well defined and moves measurably.

A single Lorentzian + linear background is also fitted at the FTO apex and its
centre and FWHM are reported as DIAGNOSTICS only: the FWHM says whether the
landmark is a sharp folded-TO mode (FWHM ~15-25 cm-1 on this instrument: x50
long-working-distance objective, 600 l/mm grating, furnace window) or the broad
disordered Si-C band, and the apex-vs-Lorentzian gap shows how much the answer
depends on the recipe (a Lorentzian centre is pulled by the broad band underneath
— the same effect documented in fit_stress_points.py).

THREE COMPARISONS, answering different questions:
  (1) T vs RT-BEFORE is the literal form of the manuscript sentence, but the
      structure is not the same at both ends: between the RT-before spectrum and
      1000 C the specimen has also recovered. Reported, flagged.
  (2) T vs RT-AFTER is the same recovered structure measured hot and cold, so the
      difference is thermal only. Available for the two Si specimens.
  (3) dw/dT from a weighted straight line through the in-situ positions, so the
      1000 C number does not rest on one spectrum pair. Fitted twice: over the
      whole series, and over the recovered range only (+ the RT-after anchor).

Writes: output/thermal_shift_positions.csv  (every stage of every series)
        output/thermal_shift_summary.csv    (the quoted comparisons)
        output/fig_thermal_shift.png        (positions vs temperature)
Run from the repo root:  .venv\\Scripts\\python.exe output\\thermal_shift.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(errors="replace")

from preprocessing import preprocess
from analysis_plotting import apply_pub_style, PUB_DPI
from annealing_series import (SERIES, index_series, stage_order, stage_label,
                              usable_spectra)

OUT = "output"
TOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]
ORDERS = (4, 5, 6)
NOMINAL = 5
T_ROOM = 25.0          # room temperature, C
T_CLAIM = 1000         # the temperature the manuscript sentence quotes
RNG_SEED = 20260805    # bootstrap seed: fixed so the CSV is reproducible
N_BOOT = 300

# folded-TO: crystalline RT position ~789 cm-1, ~20 cm-1 lower at 1000 C; in
# disorder-dominated spectra the same region carries the broad Si-C band.
TO_SEARCH = (745.0, 812.0)
PARAB_HW = 8.0         # apex parabola half-window
LORENTZ_HW = 20.0      # diagnostic Lorentzian window, +- this about the apex
# A folded-TO landmark counts as a sharp mode only inside this FWHM range: above
# it the landmark is the broad disordered Si-C band; below it the diagnostic
# Lorentzian has locked onto a noise spike rather than a band.
SHARP_FWHM = (8.0, 30.0)
# Si-C envelope + the flat region above it, for the half-height LO edge.
BAND_WINDOW = (700.0, 960.0)
FLOOR_WINDOW = (1020.0, 1120.0)
EDGE_SEARCH = (900.0, 1010.0)
# Recovered range per specimen (abrupt recovery 800-900 C, Section 3.2).
FIT_FROM_C = {"ne": 900, "si300": 900, "si750": 800}
SERIES_KEYS = ("ne", "si300", "si750")


def lorentz_lin(x, a, c, w, b0, b1):
    return a * w ** 2 / ((x - c) ** 2 + w ** 2) + b0 + b1 * (x - x.mean())


def _apex_of(xs, ys, i, hw):
    """Parabola apex about index i, or the grid point if the parabola opens up."""
    w = (xs >= xs[i] - hw) & (xs <= xs[i] + hw)
    if w.sum() < 4:
        return float(xs[i])
    a2, a1, _ = np.polyfit(xs[w] - xs[i], ys[w], 2)
    if a2 >= 0:
        return float(xs[i])
    p = float(xs[i] - a1 / (2 * a2))
    return p if xs[i] - hw <= p <= xs[i] + hw else float(xs[i])


def fto_apex(x, y, rng):
    """Band-maximum position in TO_SEARCH, with a residual-bootstrap u.

    The bootstrap resamples the local parabola's residuals (i.e. the spectral
    noise) and refits, so u reflects how much noise can move the apex — including
    the case where noise moves which grid point is the maximum.
    """
    m = (x >= TO_SEARCH[0]) & (x <= TO_SEARCH[1])
    xs, ys = x[m], y[m]
    if xs.size < 6:
        return dict(apex=np.nan, u_apex=np.nan)
    i = int(np.argmax(ys))
    pos = _apex_of(xs, ys, i, PARAB_HW)
    w = (xs >= xs[i] - PARAB_HW) & (xs <= xs[i] + PARAB_HW)
    resid = ys[w] - np.polyval(np.polyfit(xs[w] - xs[i], ys[w], 2), xs[w] - xs[i])
    boots = []
    for _ in range(N_BOOT):
        yb = ys.copy()
        yb[w] = yb[w] + rng.choice(resid, size=int(w.sum()), replace=True)
        j = int(np.argmax(yb))
        boots.append(_apex_of(xs, yb, j, PARAB_HW))
    return dict(apex=pos, u_apex=float(np.std(boots, ddof=1)),
                prominence=float(ys[i] - max(ys[0], ys[-1])))


def fto_lorentzian(x, y, seed_pos):
    """Diagnostic single-Lorentzian centre + FWHM about the apex."""
    w = (x >= seed_pos - LORENTZ_HW) & (x <= seed_pos + LORENTZ_HW)
    xf, yf = x[w], y[w]
    if xf.size < 8:
        return dict(lorentz=np.nan, u_lorentz=np.nan, fwhm=np.nan)
    b0 = float(np.percentile(yf, 10))
    try:
        popt, pcov = curve_fit(
            lorentz_lin, xf, yf,
            p0=[max(yf.max() - b0, 1e-6), seed_pos, 8.0, b0, 0.0],
            bounds=([0, seed_pos - LORENTZ_HW, 0.5, -np.inf, -np.inf],
                    [np.inf, seed_pos + LORENTZ_HW, 4 * LORENTZ_HW, np.inf, np.inf]),
            maxfev=40000)
    except Exception:
        return dict(lorentz=np.nan, u_lorentz=np.nan, fwhm=np.nan)
    return dict(lorentz=float(popt[1]), u_lorentz=float(np.sqrt(pcov[1, 1])),
                fwhm=float(2 * popt[2]))


def lo_edge(x, y):
    """High-frequency half-height edge of the Si-C envelope (folded-LO region)."""
    band = (x >= BAND_WINDOW[0]) & (x <= BAND_WINDOW[1])
    flat = (x >= FLOOR_WINDOW[0]) & (x <= FLOOR_WINDOW[1])
    if band.sum() < 10 or flat.sum() < 10:
        return dict(edge=np.nan, u_edge=np.nan)
    band_max, floor = float(np.max(y[band])), float(np.median(y[flat]))
    noise = float(np.std(y[flat]))
    half = floor + 0.5 * (band_max - floor)
    m = (x >= EDGE_SEARCH[0]) & (x <= EDGE_SEARCH[1])
    xs, ys = x[m], y[m]
    cross = [i for i in range(len(xs) - 1) if ys[i] >= half > ys[i + 1]]
    if not cross:
        return dict(edge=np.nan, u_edge=np.nan)
    i = cross[-1]
    slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
    return dict(edge=float(xs[i] + (half - ys[i]) / slope),
                u_edge=float(1.5 * noise / max(abs(slope), 1e-9)))


def measure(path, convert_wavelength, rng):
    """One spectrum: both landmarks at the nominal baseline order, with the
    baseline-order (4/5/6) spread and the estimator gap folded into u(FTO)."""
    per_order = {}
    for o in ORDERS:
        x, y = preprocess(path, 170, 2000, imodpoly_order=o, imodpoly_tol=1e-3,
                          imodpoly_max_iter=100, normalisation="vector-0to1",
                          plot=False, save_path=None,
                          convert_wavelength_to_shift=convert_wavelength)
        a = fto_apex(x, y, rng)
        per_order[o] = dict(**a, **fto_lorentzian(x, y, a["apex"]), **lo_edge(x, y))
    nom = per_order[NOMINAL]

    def band(k):
        v = [per_order[o][k] for o in ORDERS]
        v = [q for q in v if np.isfinite(q)]
        return (max(v) - min(v)) / 2.0 if len(v) > 1 else 0.0

    gap = (abs(nom["lorentz"] - nom["apex"]) / 2.0
           if np.isfinite(nom["lorentz"]) and np.isfinite(nom["apex"]) else 0.0)
    fto = dict(center=nom["apex"], fwhm=nom["fwhm"], lorentz=nom["lorentz"],
               u_noise=nom["u_apex"], u_baseline=band("apex"), u_estimator=gap,
               u=float(np.sqrt(np.nansum([nom["u_apex"] ** 2, band("apex") ** 2, gap ** 2]))))
    lo = dict(center=nom["edge"], fwhm=np.nan, lorentz=np.nan,
              u_noise=nom["u_edge"], u_baseline=band("edge"), u_estimator=0.0,
              u=float(np.hypot(nom["u_edge"], band("edge"))))
    return fto, lo


def combine(records):
    """Mean over a stage's repeat spectra; u = within-spectrum u combined with the
    repeat scatter, which carries spot-to-spot and focus-drift effects no single
    spectrum's u can see."""
    good = [r for r in records if np.isfinite(r["center"]) and np.isfinite(r["u"])]
    if not good:
        return dict(center=np.nan, u=np.nan, n=0, spread=np.nan, fwhm=np.nan)
    c = np.array([r["center"] for r in good])
    u = np.array([r["u"] for r in good])
    spread = float(c.std(ddof=1)) if len(c) > 1 else 0.0
    return dict(center=float(c.mean()),
                u=float(np.hypot(np.sqrt((u ** 2).sum()) / len(u), spread)),
                n=len(good), spread=spread,
                fwhm=float(np.nanmean([r["fwhm"] for r in good])))


# ---------------------------------------------------------------- measurement
rng = np.random.default_rng(RNG_SEED)
rows = []
for key in SERIES_KEYS:
    spec = SERIES[key]
    stages, _ = index_series(key)
    print(f"\n=== {key}: {spec['header']} ===")
    print(f"  {'stage':>10}  {'FTO apex':>16} {'FWHM':>6} {'band':>6}"
          f"  {'LO half-height edge':>18}   n")
    for stage in stage_order(stages):
        # same usability filter as the Figure 11 script: full spectral coverage,
        # and a SiC spot rather than the residual free-Si matrix
        paths, rejected = usable_spectra(stages[stage], spec["convert_wavelength"],
                                         (200.0, 1900.0))
        for p, why in rejected:
            print(f"  [dropped] {stage_label(stage)}: {os.path.basename(p)[:44]} — {why}")
        if not paths:
            print(f"  {str(stage_label(stage)):>10}  no usable spectrum")
            continue
        ftos, los = [], []
        for p in paths:
            f, l = measure(p, spec["convert_wavelength"], rng)
            ftos.append(f)
            los.append(l)
        cf, cl = combine(ftos), combine(los)
        T = T_ROOM if isinstance(stage, str) else float(stage)
        for name, c in (("FTO", cf), ("LO edge", cl)):
            rows.append(dict(series=key, header=spec["header"], band=name,
                             stage=stage_label(stage), temp_C=T,
                             is_rt=isinstance(stage, str),
                             rt_kind=(stage if isinstance(stage, str) else ""),
                             center=c["center"], u_center=c["u"],
                             n_used=c["n"], n_files=len(paths),
                             repeat_spread=c["spread"], fwhm=c["fwhm"],
                             sharp=(name == "FTO" and np.isfinite(c["fwhm"])
                                    and SHARP_FWHM[0] <= c["fwhm"] <= SHARP_FWHM[1]),
                             files=";".join(os.path.basename(p) for p in paths)))
        kind = ("sharp" if (np.isfinite(cf["fwhm"])
                            and SHARP_FWHM[0] <= cf["fwhm"] <= SHARP_FWHM[1]) else "broad")
        print(f"  {str(stage_label(stage)):>10}  {cf['center']:8.2f} ± {cf['u']:4.2f}"
              f" {cf['fwhm']:6.1f} {kind:>6}  {cl['center']:8.2f} ± {cl['u']:4.2f}"
              f"   {len(paths)}")

pos = pd.DataFrame(rows)
pos.to_csv(os.path.join(OUT, "thermal_shift_positions.csv"), index=False)
print(f"\n[OK] {OUT}/thermal_shift_positions.csv")


# ------------------------------------------------------------------- summary
def get(series, band, stage_txt):
    s = pos[(pos.series == series) & (pos.band == band) & (pos.stage == stage_txt)]
    s = s[np.isfinite(s["center"])]
    return s.iloc[0] if len(s) else None


summary = []
print(f"\n=== Shift at {T_CLAIM} °C relative to room temperature ===")
for key in SERIES_KEYS:
    for band in ("FTO", "LO edge"):
        hot = get(key, band, f"{T_CLAIM} °C")
        if hot is None:
            print(f"  {key:6s} {band:8s} no {T_CLAIM} °C measurement")
            continue
        for kind, note in (("RT after", "thermal only (same recovered structure)"),
                           ("RT before", "thermal + recovery (structure differs)")):
            cold = get(key, band, kind)
            if cold is None:
                print(f"  {key:6s} {band:8s} no {kind} spectrum in this series")
                continue
            d = hot["center"] - cold["center"]
            ud = float(np.hypot(hot["u_center"], cold["u_center"]))
            summary.append(dict(series=key, header=SERIES[key]["header"], band=band,
                                comparison=f"{T_CLAIM} °C − {kind}",
                                hot_cm1=hot["center"], u_hot=hot["u_center"],
                                cold_cm1=cold["center"], u_cold=cold["u_center"],
                                delta_cm1=d, u_delta_cm1=ud, note=note))
            print(f"  {key:6s} {band:8s} {T_CLAIM} °C − {kind:10s} = "
                  f"{d:+7.1f} ± {ud:4.1f} cm-1   [{note}]")

print("\n=== dω/dT (weighted straight line through the in-situ positions) ===")
slopes = {}
for key in SERIES_KEYS:
    for band in ("FTO", "LO edge"):
        sub = pos[(pos.series == key) & (pos.band == band)]
        sub = sub[np.isfinite(sub["center"])]
        variants = [
            ("whole series", pd.concat([sub[sub.rt_kind == "RT before"], sub[~sub.is_rt]])),
            (f"≥{FIT_FROM_C[key]} °C + RT-after",
             pd.concat([sub[(~sub.is_rt) & (sub.temp_C >= FIT_FROM_C[key])],
                        sub[sub.rt_kind == "RT after"]])),
        ]
        for vname, use in variants:
            if len(use) < 3:
                print(f"  {key:6s} {band:8s} {vname:24s} only {len(use)} point(s) — skipped")
                continue
            T = use["temp_C"].to_numpy(float)
            w = use["center"].to_numpy(float)
            uw = np.clip(use["u_center"].to_numpy(float), 1e-3, None)
            p, cov = np.polyfit(T, w, 1, w=1.0 / uw, cov=True)
            slope, u_slope = float(p[0]), float(np.sqrt(cov[0, 0]))
            resid = w - np.polyval(p, T)
            r2 = 1 - (resid ** 2).sum() / ((w - w.mean()) ** 2).sum()
            d, ud = slope * (T_CLAIM - T_ROOM), u_slope * (T_CLAIM - T_ROOM)
            print(f"  {key:6s} {band:8s} {vname:24s} dω/dT = ({slope*1e3:+6.1f} ±"
                  f" {u_slope*1e3:4.1f})×10⁻³ cm⁻¹ °C⁻¹  n={len(use):2d} R²={r2:5.2f}"
                  f"  → {T_CLAIM} °C − RT = {d:+6.1f} ± {ud:4.1f} cm⁻¹")
            slopes[(key, band, vname)] = (slope, u_slope)
            summary.append(dict(series=key, header=SERIES[key]["header"], band=band,
                                comparison=f"dω/dT ({vname}) → {T_CLAIM} °C − RT",
                                hot_cm1=np.nan, u_hot=np.nan, cold_cm1=np.nan,
                                u_cold=np.nan, delta_cm1=d, u_delta_cm1=ud,
                                note=f"slope ({slope*1e3:+.1f}±{u_slope*1e3:.1f})e-3"
                                     f" cm-1/°C, n={len(use)}, R²={r2:.2f}"))

# Headline FTO temperature coefficient: weighted mean of the two Si specimens'
# recovered-range slopes (the Ne specimen is kept separate — its recovered range
# holds only 4 points and it has no RT-after anchor).
sel = [(k, slopes[k]) for k in slopes
       if k[1] == "FTO" and k[2].startswith("≥") and k[0] in ("si300", "si750")]
if len(sel) == 2:
    v = np.array([s[1][0] for s in sel])
    uv = np.array([s[1][1] for s in sel])
    wsum = float((1 / uv ** 2).sum())
    mean = float((v / uv ** 2).sum() / wsum)
    u_mean = float(np.sqrt(1 / wsum))
    chi2 = float((((v - mean) / uv) ** 2).sum())
    print(f"\n  HEADLINE  FTO dω/dT (Si specimens, weighted mean) = "
          f"({mean*1e3:+.1f} ± {u_mean*1e3:.1f})×10⁻³ cm⁻¹ °C⁻¹"
          f"  [χ²={chi2:.2f}, 1 DoF]  → {T_CLAIM} °C − RT = "
          f"{mean*(T_CLAIM-T_ROOM):+.1f} ± {u_mean*(T_CLAIM-T_ROOM):.1f} cm⁻¹")
    summary.append(dict(series="si300+si750", header="Si specimens, weighted mean",
                        band="FTO", comparison=f"dω/dT headline → {T_CLAIM} °C − RT",
                        hot_cm1=np.nan, u_hot=np.nan, cold_cm1=np.nan, u_cold=np.nan,
                        delta_cm1=mean * (T_CLAIM - T_ROOM),
                        u_delta_cm1=u_mean * (T_CLAIM - T_ROOM),
                        note=f"slope ({mean*1e3:+.1f}±{u_mean*1e3:.1f})e-3 cm-1/°C,"
                             f" chi2={chi2:.2f} on 1 DoF"))

pd.DataFrame(summary).to_csv(os.path.join(OUT, "thermal_shift_summary.csv"), index=False)
print(f"\n[OK] {OUT}/thermal_shift_summary.csv")

# ------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=PUB_DPI)
for ax, band, ylab in ((axes[0], "FTO", "Folded-TO band maximum (cm$^{-1}$)"),
                       (axes[1], "LO edge", "Si-C band half-height edge (cm$^{-1}$)")):
    for i, key in enumerate(SERIES_KEYS):
        sub = pos[(pos.series == key) & (pos.band == band)]
        sub = sub[np.isfinite(sub["center"])]
        ins = sub[~sub.is_rt]
        ax.errorbar(ins["temp_C"], ins["center"], yerr=ins["u_center"], fmt="o-",
                    ms=4, lw=1, capsize=2, color=TOL[i], label=SERIES[key]["header"])
        for kind, mk in (("RT before", "D"), ("RT after", "^")):
            r = sub[sub.rt_kind == kind]
            ax.errorbar(r["temp_C"], r["center"], yerr=r["u_center"], fmt=mk,
                        mfc="none", ms=7, lw=1, capsize=2, color=TOL[i])
    apply_pub_style(ax, xlabel="Measurement temperature (°C)", ylabel=ylab)
axes[0].legend(fontsize=8, frameon=False, loc="lower left")
axes[1].annotate("open diamond = RT before anneal\nopen triangle = RT after anneal",
                 xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8)
for ax, letter in zip(axes, ("(a)", "(b)")):
    ax.annotate(letter, xy=(0.02, 0.98), xycoords="axes fraction", ha="left",
                va="top", fontsize=12, weight="bold")
fig.subplots_adjust(wspace=0.28)
p = os.path.join(OUT, "fig_thermal_shift.png")
fig.savefig(p, bbox_inches="tight", dpi=PUB_DPI)
plt.close(fig)
print(f"[OK] {p}")
