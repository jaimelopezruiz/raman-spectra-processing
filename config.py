"""Loader for per-sample fitting configs (params/configs/*.yaml).

Each config records, for one sample, the physics-informed initial peak
assignments (model, seed amplitude/center/width, literature mode label)
plus the preprocessing and fitting settings used to produce its published
fit. See README.md for the schema.
"""

import glob
import os
import re

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params", "configs")


def find_matching_config(input_path, config_dir=CONFIG_DIR):
    """Auto-detect the sample config for an input spectrum.

    Each config may declare `match:` — a list of case-insensitive regex
    patterns tested against the input file's full path (with forward
    slashes). If several configs match, the one with the highest
    `priority:` (default 0) wins; this lets e.g. the annealing config
    outrank the per-sample survey configs for files under input/Annealing/.

    Returns the config path, or None if no config matches.
    """
    haystack = os.path.abspath(input_path).replace("\\", "/")

    candidates = []
    for cfg_path in sorted(glob.glob(os.path.join(config_dir, "*.yaml"))):
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            continue
        patterns = raw.get("match") or []
        for pattern in patterns:
            if re.search(pattern, haystack, re.IGNORECASE):
                candidates.append((raw.get("priority", 0), cfg_path, raw.get("sample", cfg_path)))
                break

    if not candidates:
        return None

    candidates.sort(key=lambda c: -c[0])
    top_priority = candidates[0][0]
    top = [c for c in candidates if c[0] == top_priority]
    if len(top) > 1:
        names = ", ".join(c[2] for c in top)
        print(f"[!] Input matches several configs at the same priority ({names}); "
              f"using the first. Pass --config to disambiguate.")
    return top[0][1]


def load_sample_config(path):
    """Read a sample config YAML and return a dict with:
      - 'regions': [(start, end, [peak_dict, ...]), ...] for fit_peaks_regionwise
      - 'preprocessing': dict of preprocess() overrides (may be empty)
      - 'fitting': dict with e.g. center_tolerance (may be empty)
      - 'meta': everything else (sample, references, notes, ...)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "regions" not in raw:
        raise ValueError(f"[!] Config {path} must be a mapping with a 'regions' list.")

    regions = []
    for entry in raw["regions"]:
        start, end = entry["range"]
        peaks = entry["peaks"]
        for peak in peaks:
            missing = {"model", "amp", "center", "width"} - set(peak)
            if missing:
                raise ValueError(f"[!] Peak {peak} in {path} is missing keys: {sorted(missing)}")
            if peak["model"] == "bwf" and "q" not in peak:
                raise ValueError(f"[!] BWF peak {peak} in {path} needs an initial 'q'.")
        regions.append((start, end, peaks))

    meta = {k: v for k, v in raw.items() if k not in ("regions", "preprocessing", "fitting")}
    return {
        "regions": regions,
        "preprocessing": raw.get("preprocessing", {}) or {},
        "fitting": raw.get("fitting", {}) or {},
        "meta": meta,
    }
