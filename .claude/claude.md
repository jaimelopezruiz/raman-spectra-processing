# Project state — raman-spectra-processing

Physics-informed Raman preprocessing + curve-fitting pipeline for ion-irradiated
REFEL/RB-SiC (Oxford internship project, Jaime Lopez Ruiz & Daniel Collins).
Goal: make the repo **citable** alongside the report/paper. Peak assignments are
user-supplied (Sorieul, Chaâbane, Gorban et al. — report refs [26-28]), never
auto-detected: the fitter refines what the physics says is there.

## Architecture & logic (current)

```
main.py                CLI + interactive entry point
  └─ config.py         loads params/configs/*.yaml; auto-detects config from
                       input filename via per-config `match:` regexes
                       (`priority:` breaks ties; annealing.yaml = 10)
  └─ preprocessing.py  file parsing (incl. WITec [Data] sections), crop,
                       IModPoly baseline, normalisation. NO denoising by
                       default (see "Key decisions")
  └─ curve_fitting.py  region-wise scipy curve_fit; keeps pcov; 1σ errors on
                       center/FWHM/area/amp (propagated via models.py);
                       per-region R²/RMSE/residual variance
  └─ models.py         single source of truth: gauss/lorentz/voigt/bwf line
                       shapes, FWHM/area formulas, numeric error propagation,
                       np.trapz→trapezoid shim for numpy 2
  └─ analysis_plotting.py  fit figure + residual panel, ± errors in console/
                       text summaries, CSV export (params incl. raw amp/wid/q)

replot_from_csv.py           rebuilds figures from output CSVs; exact peak
                             reconstruction (incl. BWF) from raw params
multi_spectra_comparison.py  annealing-series overlay (renamed from
                             "Multi Spectra Comparision.py"; main() guard,
                             no longer executes at import)
```

Outputs per spectrum (to `output/`): `_processed.csv`, `_fitted_curve.csv`
(+ residual column), `_peak_parameters.csv` (center/FWHM/area ± 1σ,
assignment, raw amp/wid/q), `_fit_statistics.csv` (per-region R², RMSE).

CLI: `python main.py FILE [--config YAML] [--no-preprocess] [--no-legend]
[--no-show] [--save-fig PATH] [--output-dir DIR]`. No args → Tk dialogs
(original workflow). Multiple files → overlay plot instead of fitting.

## Per-sample configs (`params/configs/*.yaml`)

One YAML per sample = the "physics input": regions, peaks (model, seed
amp/center/width[/q]), literature `assignment` per peak, `match` patterns,
preprocessing + center_tolerance. Extracted from the old tuning scratchpad
`params/Regions.txt` (now `archive/Regions.txt`, kept verbatim).

| Config | Source entry in Regions.txt |
|---|---|
| unirradiated.yaml | "unirradiated peaks" (was also hardcoded in old main.py) |
| si_2.5dpa_750C.yaml | "Si - 2.5dpa 750C - PERFECT" |
| si_0.25dpa_750C.yaml | last "Si - 0.25dpa, 750C" (post-PERFECT refinement) |
| ne_2.5dpa_300C.yaml | "Ne = 2.5dpa, 300C - Perfect" |
| si_2.5dpa_300C.yaml | "Si = 2.5 dpa, 300C - Perfect" (same peaks as Ne) |
| annealing.yaml | "settings for annealing data" |
| si_0.25dpa_300C.yaml | none in Regions.txt — authored 2026-06-15 from report Table 5 + Figure 11 |

**(resolved) Si 0.25 dpa 300C** (`input/2 Si 300 0.25.csv`): config now exists
(`si_0.25dpa_300C.yaml`). Peak centres + assignments taken from report draft
Table 5 (Si-Si/Si-C/C-C breakdown), seed amps/widths read off Figure 11.
Auto-detects, fits at R² ≈ 0.995. The C-C sp3 ~1082 band is modelled as an
asymmetric BWF (positive q → right tail) so it spans the broad weak 1000–1300
hump; this stops the 1226 sp3 and 1418 D band up-shifting/pinning at the
bounds (they now land ~1239 and ~1440, near their Table-5 centres). Earlier
attempt with a plain gauss there left the D band pinned at ~1505; tightening
center_tolerance only moved the pin — the BWF was the fix.

**To verify (Jaime):** peaks at ~435, ~660–670, ~1200–1250 cm⁻¹ are marked
`assignment: "unassigned — verify against report"` in the configs.

## Key decisions (don't re-litigate)

1. **SavGol (11,10) was the identity transform** — degree-10 polynomial
   through 11 points = exact interpolation, so historical "denoising" did
   nothing. Decision: denoising OFF by default (sg_window/sg_polyorder=None);
   published numbers unaffected; documented in preprocessing.py + README.
   Do NOT "fix" it to polyorder=3 — that would change published numbers.
2. **Configs must use `normalisation: vector-0to1`.** Inputs are raw counts
   (1e3–1e5); seed amps are 0.05–1.2 with bounds 2×seed, so fitting raw
   counts cannot converge (verified: R² negative with "none", 0.95–0.996
   with vector-0to1). The old main.py's "none" was an overlay-session
   leftover. Implication: areas/heights are in normalized units → only
   relative intensities are physically meaningful.
3. **Baseline param removed from fits** — old ±1e-6-bounded offset was
   pinned to zero and degraded pcov conditioning.
4. **MIT license** (user-chosen). **params/*.xlsx stay tracked** (paper
   graph parameters extracted from them). Microscope images (.czi/.bmp/.png
   under input/) untracked but kept on disk, ignored via .gitignore.
5. Width conventions unchanged from original code: gauss wid=σ, lorentz
   wid=HWHM, voigt composite (Olivero–Longbothum FWHM), bwf FWHM =
   2|w|√(1+1/q²). Model math kept byte-identical to preserve results.

## Verification status (2026-06-12)

Tested in `.venv` (Python 3.14.5; pins in requirements.txt = tested versions):
- Unirradiated fit: R² 0.90–0.99/region; TO triplet 768.4/784.5/795.3 cm⁻¹,
  errors ≤0.1 cm⁻¹.
- Si 2.5dpa 750C (auto-detected): R² = 0.996 main region, BWF path OK.
- replot_from_csv reconstruction matches saved fit to ~5e-15 (incl. BWF).
- Auto-detection correct for all 9 test paths incl. annealing precedence.
- Windows console: main.py reconfigures stdout errors="replace" (cp1252
  can't print σ/cm⁻¹).

## Last steps / TODO

- [ ] Jaime: confirm "unassigned" peak assignments against the report table.
- [x] si_0.25dpa_300C.yaml authored from report Table 5 + Figure 11 (2026-06-15);
      C-C sp3 modelled as BWF so C-C centres sit near their table values, R²≈0.995.
- [ ] Add ORCID iDs to CITATION.cff; archive a release on Zenodo, add DOI.
- [ ] Methods sentence for the paper: fitted on unsmoothed baseline-corrected
      data; intensities normalized (vector-0to1) → relative units.
- [ ] Optionally regenerate output/ CSVs with the new pipeline (old tracked
      outputs lack uncertainty/stats columns). Old Unirradiated_* outputs
      were restored after a smoke test overwrote them.
- [ ] Commit everything (working tree uncommitted as of 2026-06-12; renames
      done with git mv; images/__pycache__ untracked via git rm --cached).
