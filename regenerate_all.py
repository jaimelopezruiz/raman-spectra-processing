"""Regenerate every published result from the tracked inputs, in dependency order.

    .venv\\Scripts\\python.exe regenerate_all.py            # everything
    .venv\\Scripts\\python.exe regenerate_all.py --list     # show the steps only

WHY THIS EXISTS. The figure and table scripts in output/ are not independent:
make_sm3_table.py reads the per-spectrum *_peak_parameters.csv that main.py
writes, and make_figures.py reads the chi tables that annealing_chi.py and
make_au_chi_rt.py write. Running them in the wrong order fails with a bare
FileNotFoundError, and running them in an already-populated output/ hides the
problem entirely — which is how a clean clone came to be unable to rebuild
Figures 10 and SM.3 without anyone noticing. This driver encodes the order.

Every step is a separate subprocess, exactly as documented in the README, so
what runs here is what a reader would type by hand.

Figures 2 and 4 are built from rasters of the manuscript's own images, because
the data behind them is not ours to distribute (Fig 2 a/b) or is not in the
repository (Fig 4, single-crystal spectra). Those two rasters are shipped in
data/figure_sources/, so both figures still rebuild here without the .docx.

NOT INCLUDED (they need the manuscript .docx itself, which is not distributed):
  output/restore_published_figures.py recovery tool for output/ from the doc
  output/fig3_offsets_forensics.py    one-off forensics behind the Fig 4 caption
  output/koyanagi_lo_digitise.py      one-off digitisation behind the Fig 6b reference
"""
import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
REPO = os.path.dirname(os.path.abspath(__file__))

SURVEY = [
    "input/Unirradiated.csv",
    "input/0 Ne 300 2.5.csv",
    "input/1 Si 300 2.5.csv",
    "input/2 Si 300 0.25.csv",
    "input/3 Si 750 0.25.csv",
    "input/4 Si 750 2.5.csv",
]
AU_DIR = "input/Annealing/Au 2.5e15 RT"
SI300_DIR = "input/Annealing/Si 2.5dpa  300C"
# cross_spectrum_ratios writes one row per argument in the order given, and the
# published cross_ratios.csv is in the physics order (ascending retained damage)
# used by make_figures.SURVEY — not filename order.
CROSS_ORDER = [
    "input/3 Si 750 0.25.csv",
    "input/4 Si 750 2.5.csv",
    "input/2 Si 300 0.25.csv",
    "input/1 Si 300 2.5.csv",
    "input/0 Ne 300 2.5.csv",
]

# (label, argv) — order matters, see the module docstring
STEPS = (
    [(f"fit {os.path.basename(p)}", [PY, "main.py", p, "--no-show"]) for p in SURVEY]
    + [
        ("chi vs T, Au series",
         [PY, "annealing_chi.py", AU_DIR, "--convert-wavelength",
          "--out", "output/au_chi_vs_T.csv"]),
        ("chi at RT, Au series",
         [PY, "output/make_au_chi_rt.py"]),
        ("chi vs T, Si 2.5 dpa 300 C",
         [PY, "annealing_chi.py", SI300_DIR, "--out", "output/si300_2.5_chi_vs_T.csv"]),
        ("per-point stress-map fits", [PY, "output/fit_stress_points.py"]),
        ("cross-spectrum ratios + crystallinity index",
         [PY, "cross_spectrum_ratios.py", "--pristine", "input/Unirradiated.csv"]
         + CROSS_ORDER + ["--out", "output/cross_ratios.csv"]),
        ("in-situ thermal shift", [PY, "output/thermal_shift.py"]),
        ("Figures 3, 6, 10, 11 + SM.7, SM.8", [PY, "output/make_figures.py"]),
        ("Figure 9 + SM.2-SM.5", [PY, "output/make_fig11_annealing.py"]),
        ("Figure 2", [PY, "output/make_fig2_row.py"]),
        ("Figure 4", [PY, "output/make_fig3_wide.py"]),
        ("Figure SM.1", [PY, "output/make_sm1_curvefits.py"]),
        ("Table SM.1", [PY, "output/make_sm3_table.py"]),
        ("Figure SM.6", [PY, "output/make_sm4_srim.py"]),
        ("Fig 4 offsets forensics (audit)", [PY, "output/fig3_offsets_forensics.py"]),
    ]
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    args = ap.parse_args()

    if args.list:
        for i, (label, argv) in enumerate(STEPS, 1):
            print(f"{i:2d}. {label}\n    {' '.join(argv[1:])}")
        return 0

    os.makedirs(os.path.join(REPO, "output"), exist_ok=True)
    failures = []
    for i, (label, argv) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {label} ... ", end="", flush=True)
        t0 = time.time()
        r = subprocess.run(argv, cwd=REPO, capture_output=True, text=True,
                           errors="replace")
        if r.returncode == 0:
            print(f"ok ({time.time() - t0:.1f}s)")
        else:
            print("FAILED")
            print((r.stderr or r.stdout).strip()[-800:])
            failures.append(label)

    print()
    if failures:
        print(f"{len(failures)} step(s) failed: {', '.join(failures)}")
        return 1
    print(f"All {len(STEPS)} steps completed. Results are in output/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
