# raman-spectra-processing

A physics-informed preprocessing and curve-fitting pipeline for Raman
spectra, developed at the University of Oxford to investigate the effects of
ion irradiation (Si and Ne, 0.25–2.5 dpa, 300–750 °C) and subsequent
annealing on reaction-bonded (REFEL) silicon carbide.

## Design philosophy

Peak attribution is **user-supplied, not automatic**. Each sample has a
version-controlled config (`params/configs/*.yaml`) listing the peaks to fit,
their line-shape models, initial parameters, and - crucially - the
literature mode assignment each peak encodes (Si–Si TA/LA/LO/TO, Si–C
TO/LO and disorder bands, C–C sp³/D/G bands, after Sorieul et al.,
Chaâbane et al. and Gorban et al.). The fitter only refines what the
physics says should be there; it does not invent peaks. This is deliberate:
unconstrained multi-peak fitting of broad, overlapping disorder bands
overfits easily and produces physically meaningless components.

## Pipeline

```
input spectrum (.csv/.txt, incl. WITec [Data]-section exports)
   │  preprocessing.py
   ├─ crop to analysis window (default 170–2000 cm⁻¹)
   ├─ IModPoly baseline subtraction (default order 5)
   ├─ normalisation (none / vector / min-max / max / vector-0to1)
   │      NOTE: no denoising - fitting is done on unsmoothed data (see below)
   │  curve_fitting.py  (+ models.py)
   ├─ region-wise least-squares fit of the config's peaks
   │    · scipy curve_fit with bounded centers (± center_tolerance)
   │    · models: gaussian, lorentzian, Voigt, Breit-Wigner-Fano
   │      (`pvoigt` is accepted as an alias fitted as a true Voigt)
   │    · 1σ parameter uncertainties from the covariance matrix,
   │      propagated to center, FWHM and area
   │    · per-region R², RMSE, residuals
   │  analysis_plotting.py
   └─ figure (fit + components + residual panel) and output CSVs
```

Outputs per spectrum (written to `output/`):

| File | Contents |
| --- | --- |
| `<name>_processed.csv` | baseline-corrected (and optionally normalised) spectrum |
| `<name>_fitted_curve.csv` | total fitted curve and residual on the data grid |
| `<name>_peak_parameters.csv` | per peak: model, assignment, center ± 1σ, FWHM ± 1σ, area ± 1σ, height, raw parameters (amp, wid, q), and a `Flags` column |
| `<name>_fit_statistics.csv` | per region: R², RMSE, residual variance, point/parameter counts |
| `<name>_run_metadata.json` | provenance: input file, config, sample, preprocessing/fitting settings, git commit, and package versions used |
| `<name>_derived.json` | chemical-disorder ratio χ — integrated-window (primary) and fit-based (cross-check, covariance-aware 1σ) — plus band areas and the per-band peak audit |

Uncertainties are 1σ values from the `curve_fit` covariance matrix (scaled
by the residual variance, since no per-point measurement errors are
available), propagated to derived quantities to first order.

### Fit-quality flags

Because the peak set is fixed by the config, an individual seeded peak can
occasionally be driven to a fit bound or collapse to zero amplitude - for
example a weak band that the data does not actually support. Such a component
is a fit artefact, not a refined physical band, and its reported centre, FWHM
and area should not be trusted. The pipeline detects these cases and records
them in the `Flags` column of `<name>_peak_parameters.csv` (and prints a
warning), with tags such as `amp~0 (peak effectively absent)`,
`width at upper/lower bound`, `centre at +/- tolerance bound`, `q at bound`,
and `centre drifted N cm-1 from seed`. The last one is deliberate: the ±
`center_tolerance` window is kept wide so genuinely shifted bands can be
followed, but any peak that then lands far from its literature seed is flagged
so its assignment label is never taken at face value. Always check the `Flags`
column before quoting a peak: a flagged peak usually means the config
over-specifies that region, or a band moved, rather than that a clean feature
was measured at the tabulated position.

### Chemical disorder χ

The chemical-disorder ratio χ = A(D-band) / A(TO-band) (report §2.3.3) is
recorded in `<name>_derived.json` two ways:

- **Integrated-window χ (primary).** χ = ∫(D window) / ∫(TO window) of the
  baseline-corrected, normalised spectrum — the C-C D band (~1340–1470 cm⁻¹)
  over the Si-C TO band (~700–850 cm⁻¹, *including* the broad disordered SiC
  that overlaps it). This is decomposition-free and robust: a *fixed* peak-fit
  cannot track a spectrum evolving from amorphous to recovered (it
  mis-decomposes recovered spectra and χ blows up), whereas the integral
  degrades gracefully. Windows are `CHI_D_WINDOW`/`CHI_TO_WINDOW` in
  `derived_quantities.py`. Its dominant uncertainty is the I-ModPoly **baseline
  order**, not a fit covariance — quantify it by re-integrating at several
  orders (`annealing_chi.py` reports this band).
- **Fit-based χ (cross-check).** Sum of the fitted D-region peak areas over the
  TO-region peak areas, with a covariance-aware 1σ (correlations between
  overlapping peaks *within* a band are included; the D and TO bands are
  effectively independent — measured correlation ≈ 0). Validated against a
  Monte-Carlo simulation in `tests/`.

For the annealing series (χ vs temperature, report Figs 12/17), use
`annealing_chi.py`, which integrates each spectrum and reports χ with a
baseline-order systematic band.

### A note on denoising

No smoothing is applied before fitting. Earlier versions of this pipeline
passed the data through a Savitzky–Golay filter with `window=11,
polyorder=10`, which is mathematically the **identity transform** (a
degree-10 polynomial through 11 points interpolates exactly) - so no
smoothing was ever applied in practice, and all previously reported numbers
are unaffected. The current code makes this explicit: denoising is off by
default and pre-smoothing is discouraged for fitting, since it correlates
adjacent noise points and biases parameter uncertainties low. SavGol can
still be enabled explicitly (`sg_window=`, `sg_polyorder=`) for display
figures.

### Reading the residual panel

Every fit figure has a lower panel showing the residual, defined as the
measured intensity minus the total fitted curve at each wavenumber. It is the
part of the spectrum the model did not reproduce, re-plotted on its own scale
so that small deviations hidden in the main panel become visible. The same
values are stored in the `residual` column of `<name>_fitted_curve.csv`, and
their spread is reported as the RMSE and residual variance in
`<name>_fit_statistics.csv`.

A good fit leaves residuals that scatter randomly around zero with no
structure, at an amplitude set by the measurement noise. Coherent structure
indicates the model is still missing something:

- A sharp up-then-down wiggle under a single peak means its centre is slightly
  off; a single bump at the apex means its height or width is off, or the line
  shape is wrong (for example a Gaussian where a Lorentzian fits better).
- A broad, smooth hump or dip over a region means a component is missing or
  mismodelled there.
- A residual that sits to one side of zero over a stretch, or grows toward a
  region edge, points to a baseline, normalisation, or peak-tail issue rather
  than noise.

A high R² with a clearly structured residual is less trustworthy than a
slightly lower R² with featureless noise, so it is worth checking the residual
panel before relying on a fit. Note that intensity beyond the last fitted peak
(and outside the fit region) is not modelled, so a gentle drift there is
expected and is not a defect.

### Goodness-of-fit statistics

`<name>_fit_statistics.csv` holds one row per fit region (fitting is done
region by region, so each region is scored independently). The columns are:

| Column | Definition | Units |
| --- | --- | --- |
| `region` | the wavenumber window that was fit | cm⁻¹ |
| `n_points` | number of data points in the window | count |
| `n_params` | free parameters fitted in the window (3 per Gaussian/Lorentzian/Voigt, 4 per BWF) | count |
| `dof` | degrees of freedom, `n_points − n_params` | count |
| `R2` | `1 − SS_res / SS_tot` | dimensionless |
| `RMSE` | `sqrt(SS_res / n_points)` | normalised intensity |
| `residual_variance` | `SS_res / (n_points − n_params)` | normalised intensity² |

where `SS_res = Σ(data − fit)²` is the summed squared residual and
`SS_tot = Σ(data − mean(data))²` is the total variance of the data in the
window.

**R² (coefficient of determination)** is the fraction of the spectrum's
variance the model reproduces: 1.0 is exact, 0 is no better than a flat line
at the mean. Two properties matter for disorder-band spectra. First, R² is
dominated by the tallest features, because they contribute most of `SS_tot`,
so a fit can exceed 0.99 while still missing a small but real low-intensity
band. Second, R² never decreases when peaks are added, so it does not on its
own justify the number of components. Read it together with the residual panel
and `n_params`.

**RMSE (root-mean-square error)** is the typical size of a single residual, in
the same units as the plotted intensity. With `vector-0to1` normalisation the
spectrum runs 0 to 1, so an RMSE of about 0.016 means the average miss is
roughly 1.6 percent of full scale. This is the most directly interpretable
number: compare it to the noise amplitude visible in the baseline. An RMSE
close to that noise means the real signal has been captured; an RMSE several
times larger means something is mismodelled.

**residual_variance** is `SS_res` divided by the degrees of freedom
(`n_points − n_params`) rather than by `n_points`, which makes it the unbiased
estimate of the residual variance; its square root is the standard error of
the regression. It is not only a diagnostic. The pipeline uses it to scale the
`curve_fit` covariance matrix, and the ± 1σ uncertainties reported for each
centre, FWHM and area follow from that scaling. A larger residual variance
therefore propagates into larger parameter uncertainties. Dividing by the
degrees of freedom rather than `n_points` is what prevents a fit with many
parameters from understating its own error.

**n_points and n_params** together give the degrees of freedom. A region with
few points relative to parameters can reach a high R² by overfitting noise,
and its uncertainties will be correspondingly large, so the ratio is worth
checking. A trustworthy fit combines a high R², an RMSE close to the noise
floor, a comfortable excess of points over parameters, and a residual panel
without large coherent structure. R² on its own is not sufficient.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Unix)
pip install -r requirements.txt
```

Regression tests that lock the line-shape math (so refactors cannot silently
change published numbers) live in `tests/`; run them with `pytest` from the
repo root.

## Usage

### Scriptable (reproducible) mode

```bash
# Fit one spectrum - the sample config is auto-detected from the filename:
python main.py "input/4 Si 750 2.5.csv" --no-show --save-fig output/si_2.5dpa_750C_fit.png

# ...or pass it explicitly:
python main.py "input/4 Si 750 2.5.csv" --config params/configs/si_2.5dpa_750C.yaml --no-show

# Overlay several spectra:
python main.py input/*.csv --no-show --save-fig output/overlay.png

# Fit raw data without preprocessing:
python main.py spectrum.csv --no-preprocess
```

Options: `--config` (sample YAML), `--output-dir`, `--no-preprocess`,
`--no-legend`, `--no-show`, `--save-fig`.

**Config auto-detection:** when `--config` is omitted, the input file path is
tested against each config's `match:` patterns (case-insensitive regexes);
e.g. `"4 Si 750 2.5.csv"` selects `si_2.5dpa_750C.yaml`, and anything under
`input/Annealing/` selects `annealing.yaml` (its `priority: 10` outranks the
survey configs, whose patterns also appear in annealing filenames). If
nothing matches, the pipeline warns and falls back to
`params/configs/unirradiated.yaml` - so always check the
`[OK] Using fitting config:` line, or pass `--config` explicitly.

### Interactive mode

`python main.py` with no arguments opens file-selection and yes/no dialogs
(the original workflow).

### Side scripts

- `multi_spectra_comparison.py` - stacked overlay of an annealing series
  with temperature labels parsed from filenames
  (`--input FILES_OR_DIRS --preprocess|--no-preprocess`, or interactive).
- `replot_from_csv.py` - rebuild the fitted-spectrum figure from the saved
  output CSVs without re-fitting (exact reconstruction, including BWF peaks,
  from the stored raw parameters).
- `annealing_chi.py` - chemical-disorder ratio χ vs annealing temperature by
  window integration, with an I-ModPoly baseline-order systematic band; writes a
  χ-vs-T CSV (report Figs 12/17). Robust to the amorphous→recovered evolution
  that defeats a fixed peak-fit config.
- `cross_spectrum_ratios.py` - total-area / TO-area / FTO-height ratios of
  several samples to a pristine reference, from baseline-corrected **raw counts**
  (normalisation is deliberately off — it would erase the area change). These are
  semi-quantitative: raw-count comparison assumes equal light coupling between
  spectra (see the Fig 8 focus caveat), so prefer the within-spectrum χ for the
  quantitative disorder story.

## Sample configs

One YAML per sample in `params/configs/`:

```yaml
sample: "Si 2.5 dpa 750C"
references: >-
  Peak assignments follow Table 1 of the accompanying report, after
  Sorieul et al., Chaâbane et al. and Gorban et al. (report refs [26-28]).
preprocessing: {crop_min: 170, crop_max: 2000, baseline_order: 5, normalisation: vector-0to1}
fitting: {center_tolerance: 100}
regions:
  - range: [170, 300]
    peaks:
      - {model: voigt, amp: 0.1, center: 186, width: 10, assignment: "Si-Si TA 180-209"}
      ...
```

`model` is one of `gauss`, `lorentz`, `voigt`, `bwf` (BWF additionally
requires `q`). `amp`/`center`/`width` are the initial guesses; centers are
bounded to ± `center_tolerance` during the fit. The seed amplitudes are on
the [0, 1] normalised-intensity scale, so the configs use
`normalisation: vector-0to1` - fitting raw counts with these seeds will not
converge (amplitudes are bounded to 2× their seed). `assignment` records the
literature phonon-mode attribution and is carried through to the output
CSVs and plots. Peaks marked `unassigned - verify against report` await
confirmation against the report's assignment table.

The historical peak-tuning notes these configs were extracted from are
preserved verbatim in `archive/Regions.txt` (and in git history).

**Width conventions:** `width` is the Gaussian σ (`FWHM = 2.3548·w`), the
Lorentzian HWHM (`FWHM = 2·w`), a Voigt composite parameter (FWHM via the
Olivero–Longbothum approximation), or the BWF width
(`FWHM = 2|w|·√(1+1/q²)`).

**Area conventions.** All peak areas are analytic and independent of the crop
window, so they are reproducible and comparable across models: `amp·σ·√(2π)`
(Gaussian), `amp·π·HWHM` (Lorentzian), `amp / max(profile)` (Voigt). The BWF
area is `π·amp·|w|·(1 − 1/q²)` - the integral of the Fano profile **with its
non-vanishing continuum floor `amp/q²` removed**. Without that subtraction, a
component that collapses onto the continuum (amplitude ≈ 0) would otherwise
report the largest "area" in the spectrum. For `|q| < 1` the BWF area is small
or negative, which is a real Fano antiresonance rather than an error.

**Optional per-peak bounds.** A peak may set `width_max` to lower its width
upper bound (the default is 100 in the width units above), and a BWF may set
`q_min` and `q_max` to bound its asymmetry parameter (defaults −100 and 100).
Peaks that omit these keys fit exactly as before. They are used to keep a peak
in a sensible shape when the unconstrained fit would otherwise drift into a
degenerate solution.

**A note on the BWF line shape.** The Breit-Wigner-Fano profile,
`I(x) = amp·(1 + s/q)² / (1 + s²)` with `s = (x − center)/width`, is singular
at `q = 0`: in that limit the peak flattens into a near-constant offset across
the fit region rather than a localised band. Because such an offset can lower
the residual slightly while contributing no real peak, an unconstrained BWF
fitting a weak, broad, or poorly-defined feature can collapse onto it and show
up as a straight-line shift of the total fit. The symptoms in the output are a
fitted `q` near zero, a width pinned at its lower bound, an amplitude near
zero, and an implausibly large FWHM. The fix is to bound `q` away from zero and
to one sign with `q_min` (or `q_max`), which holds the component as a genuine
asymmetric peak; `width_max` additionally keeps a broad BWF from growing to the
width bound. The Voigt and pseudo-Voigt profiles have no such singularity.

## Input data

`input/` contains the measured spectra: per-sample survey spectra
(`<n> <ion> <temp> <dose>.csv`) and the in-situ annealing series under
`input/Annealing/` (WITec exports with `[Data]` section markers are parsed
automatically). Microscope images (.czi/.bmp/.png) acquired alongside the
spectra are kept locally but not tracked in git; contact the authors for
the full imaging dataset. `params/*.xlsx` hold the collated fitted
parameters used for the report's analysis figures.

## Repository layout

```
main.py                     entry point (CLI + interactive)
preprocessing.py            file parsing, baseline, normalisation
curve_fitting.py            region-wise fitting + uncertainties + fit stats
models.py                   shared line-shape definitions and error propagation
derived_quantities.py       χ and other report metrics from a fit (covariance-aware)
analysis_plotting.py        figures, residual panel, CSV export
config.py                   sample-config (YAML) loader
multi_spectra_comparison.py annealing-series overlay
replot_from_csv.py          figure rebuild from saved outputs
annealing_chi.py            χ vs annealing temperature (window integration + baseline band)
cross_spectrum_ratios.py    raw-count area/intensity ratios vs pristine (semi-quantitative)
params/configs/             per-sample fitting configs (the physics input)
input/                      measured spectra
output/                     generated results (per-spectrum CSVs); git-ignored,
                            regenerate with `python main.py <file>`
archive/                    historical scratch scripts and tuning notes
```

## Citing

See `CITATION.cff`. Licensed under the MIT License (see `LICENSE`).
