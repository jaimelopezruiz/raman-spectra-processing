"""Shared indexing of the four step-wise annealing series (Fig 11 / SM.2-3 / C13).

One place that answers "which file is which temperature, and which files are the
room-temperature before/after references" for every series, because the four
folders use four different naming conventions and `annealing_chi.temp_from_name`
deliberately only covers the ones its CSVs needed (it misses, e.g., the Ne
100 C file whose temperature is encoded as a trailing "1-100").

Stage vocabulary
  "RT before"  room-temperature spectrum acquired before the anneal
  "RT after"   room-temperature spectrum acquired after the anneal
  <number>     in-situ spectrum at that hold temperature, in C

Nothing here changes annealing_chi.py: the published chi-vs-T CSVs stay
byte-identical.
"""
import os
import re
import sys

# runnable as `python output/annealing_series.py` from the repo root: sys.path[0]
# is output/, so the repo modules need adding explicitly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annealing_chi import temp_from_name

# --- series definitions -------------------------------------------------------
# `glass_pref` picks, where both exist, the room-temperature spectrum acquired
# through the same furnace window as the in-situ spectra, so an RT/high-T
# comparison is not also an optical-path comparison.
NE_DIR = os.path.join("input", "Annealing", "Ne 2.5dpa 300C")
SI300_DIR = os.path.join("input", "Annealing", "Si 2.5dpa  300C")
SI750_DIR = os.path.join("input", "Annealing", "Si 2.5dpa 750C")
AU_DIR = os.path.join("input", "Annealing", "Au 2.5e15 RT")

SERIES = {
    "au": dict(
        label="Au-implanted single-crystal 6H-SiC",
        header="Au, 8 dpa, RT",
        directory=AU_DIR, ext=".csv", convert_wavelength=True,
        rt_before=r"RT before anneal", rt_after=r"RT after anneal",
    ),
    "ne": dict(
        label="Ne-implanted polycrystalline RB-SiC",
        header="Ne, 2.5 dpa, 300 °C",
        directory=NE_DIR, ext=".txt", convert_wavelength=False,
        # spectra 001-003 precede the 100 C step; two carry an explicit "RT"
        rt_before=r"Spec\.Data 1RT", rt_after=None,
    ),
    "si300": dict(
        label="Si-implanted polycrystalline RB-SiC",
        header="Si, 2.5 dpa, 300 °C",
        directory=SI300_DIR, ext=".txt", convert_wavelength=False,
        rt_before=r"RT (No )?Window", rt_after=r"RT After",
    ),
    "si750": dict(
        label="Si-implanted polycrystalline RB-SiC",
        header="Si, 2.5 dpa, 750 °C",
        directory=SI750_DIR, ext=".txt", convert_wavelength=False,
        rt_before=r"RT Before", rt_after=r"RT End",
    ),
}


def _extra_temp(name):
    """Temperatures `annealing_chi.temp_from_name` does not reach.

    The Ne series encodes its 100 C step as a trailing "…Spec.Data 1-100.txt",
    which the shared parser's "-<T>[SE ]" pattern cannot match (the delimiter is
    the file extension). Kept here rather than in annealing_chi.py so the
    published chi-vs-T tables are untouched.
    """
    m = re.search(r"-(\d{2,4})\s*\.(?:txt|csv)$", name, re.I)
    if m and 50 <= int(m.group(1)) <= 1500:
        return int(m.group(1))
    return None


def index_series(key, root="."):
    """{stage: [paths]} for one series, stages sorted RT-before, T ascending, RT-after.

    Returns (stages, skipped) where `stages` is a dict keyed by "RT before",
    int temperature, or "RT after", each mapping to the list of matching files
    in acquisition (filename) order.
    """
    spec = SERIES[key]
    directory = os.path.join(root, spec["directory"])
    stages, skipped = {}, []
    for f in sorted(os.listdir(directory)):
        if not f.lower().endswith(spec["ext"]):
            continue
        path = os.path.join(directory, f)
        if spec["rt_before"] and re.search(spec["rt_before"], f, re.I):
            stages.setdefault("RT before", []).append(path)
            continue
        if spec["rt_after"] and re.search(spec["rt_after"], f, re.I):
            stages.setdefault("RT after", []).append(path)
            continue
        T = temp_from_name(f)
        if T is None:
            T = _extra_temp(f)
        if T is None:
            skipped.append(f)
            continue
        stages.setdefault(T, []).append(path)
    # Where the same stage was recorded both through the furnace window and
    # without it, put the through-window spectrum first: the in-situ spectra are
    # all through the window, so this keeps an RT/high-T comparison from also
    # being an optical-path comparison.
    for s in stages:
        stages[s].sort(key=lambda p: ("no glass" in os.path.basename(p).lower()
                                      or "no window" in os.path.basename(p).lower()))
    return stages, skipped


# --- spectrum selection ------------------------------------------------------
# RB-SiC is SiC islands in a residual free-silicon matrix, so an in-situ spot can
# land on the silicon rather than on a SiC island. Such a spectrum is a narrow
# crystalline-Si line at ~520 cm-1 with almost no Si-C band, and it carries no
# information about the SiC recovery these figures are about. It is rejected by
# the ratio of the Si-C TO band integral to the free-Si band integral.
#
# Empirically (all four series, every repeat) that ratio is 2.3-45 for spectra
# that sampled SiC, and only two first-acquired spectra fall below 2:
#   Ne  200 C  "-200S"  ratio 0.30  (the "-200E 5min" repeat of the same hold: 6.5)
#   Si750 500 C "500--Spectrum--010" ratio 1.34  (its 011 repeat: 6.7)
# Both are replaced by their same-hold repeat. No other stage's choice changes.
SIC_TO_WINDOW = (700.0, 850.0)
FREE_SI_WINDOW = (505.0, 540.0)
MIN_SIC_TO_SI = 2.0


def sic_to_si_ratio(path, convert_wavelength=False):
    """∫(Si-C TO) / ∫(free-Si) of the baseline-corrected spectrum, unnormalised.

    Unnormalised because this is a ratio of two bands of the same spectrum: the
    per-spectrum normalisation would only rescale both.
    """
    import numpy as np
    from preprocessing import preprocess
    x, y = preprocess(path, 170, 2000, imodpoly_order=5, imodpoly_tol=1e-3,
                      imodpoly_max_iter=100, normalisation="none", plot=False,
                      save_path=None, convert_wavelength_to_shift=convert_wavelength)

    def integral(window):
        m = (x >= window[0]) & (x <= window[1])
        return float(np.trapezoid(y[m], x[m])) if m.sum() >= 2 else float("nan")

    si = integral(FREE_SI_WINDOW)
    return integral(SIC_TO_WINDOW) / si if si else float("inf")


def usable_spectra(paths, convert_wavelength=False, need=(200.0, 1900.0),
                   min_ratio=MIN_SIC_TO_SI):
    """(kept, rejected) for one stage's files, in acquisition order.

    `rejected` holds (path, reason) so callers can print what was dropped rather
    than silently reshaping a figure. Two reasons: the axis does not cover `need`
    (the Au series' narrow-window stitched exports), or the spot sampled the
    free-Si matrix instead of a SiC island.
    """
    kept, rejected = [], []
    for p in paths:
        try:
            lo, hi = x_span(p, convert_wavelength)
        except Exception as exc:
            rejected.append((p, f"unreadable ({type(exc).__name__})"))
            continue
        if lo > need[0] or hi < need[1]:
            rejected.append((p, f"axis {lo:.0f}-{hi:.0f} cm-1 does not cover "
                                f"{need[0]:.0f}-{need[1]:.0f}"))
            continue
        r = sic_to_si_ratio(p, convert_wavelength)
        if r < min_ratio:
            rejected.append((p, f"free-Si matrix spot (TO/Si = {r:.2f} < {min_ratio})"))
            continue
        kept.append(p)
    return kept, rejected


def x_span(path, convert_wavelength=False):
    """(min, max) of a spectrum's Raman-shift axis, without preprocessing.

    Used to reject the narrow-window zoom scans in the Au series (the 1800 l/mm
    stitched exports stop near 1338 cm-1, so they cover neither the C-C region
    nor the chi D window — annealing_chi.py drops them for the same reason).
    """
    from preprocessing import _read_spectrum_table, wavelength_to_shift
    x, _ = _read_spectrum_table(path)
    if convert_wavelength:
        x = wavelength_to_shift(x, 532, False)
    return float(min(x)), float(max(x))


def pick_full_range(paths, convert_wavelength=False, need=(200.0, 1900.0)):
    """First usable path for a stage (see usable_spectra), else None."""
    kept, _ = usable_spectra(paths, convert_wavelength, need)
    return kept[0] if kept else None


def stage_order(stages):
    """Stage keys bottom-to-top for a stacked overlay: RT before, ascending T, RT after."""
    out = []
    if "RT before" in stages:
        out.append("RT before")
    out += sorted(k for k in stages if isinstance(k, int))
    if "RT after" in stages:
        out.append("RT after")
    return out


def stage_label(stage):
    return stage if isinstance(stage, str) else f"{stage} °C"


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(errors="replace")
    for key in SERIES:
        stages, skipped = index_series(key)
        print(f"\n=== {key}: {SERIES[key]['header']} ===")
        for s in stage_order(stages):
            names = [os.path.basename(p) for p in stages[s]]
            print(f"  {str(stage_label(s)):>10}  n={len(names)}  {names[0][:60]}")
        if skipped:
            print(f"  [no temperature parsed: {len(skipped)}] {skipped[:4]}")
