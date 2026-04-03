import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ramanspy as rp
from ramanspy import preprocessing

def min_max_normalise_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def wavelength_to_shift(lambda_nm, lambda_exc_nm, microm):
    lambda_nm = np.array(lambda_nm, dtype=np.float64)
    if microm:
        lambda_exc_nm *= 1000
        lambda_nm *= 1000
    return 1e7 / lambda_exc_nm - 1e7 / lambda_nm

def _find_first_numeric_row(input_path, search_from=0, encoding="utf-8-sig"):
    """
    Starting from line `search_from`, return the index of the first line whose
    first whitespace-separated token parses as a float.  This skips any
    section markers, column-name rows, or other header text that follow a
    [Data] block.  Returns None if no such line is found.
    """
    try:
        with open(input_path, "r", encoding=encoding, errors="replace") as f:
            for i, line in enumerate(f):
                if i < search_from:
                    continue
                token = line.split()[0] if line.split() else ""
                try:
                    float(token)
                    return i
                except ValueError:
                    continue
    except OSError:
        pass
    return None

def _find_data_section(input_path, encoding="utf-8-sig"):
    """
    Look for a [Data] marker and return the index of the first numeric data
    row after it.  Falls back to searching the whole file if no marker exists.
    Returns None if no numeric rows are found.
    """
    try:
        with open(input_path, "r", encoding=encoding, errors="replace") as f:
            for i, line in enumerate(f):
                if line.strip().lower() == "[data]":
                    return _find_first_numeric_row(input_path, search_from=i + 1, encoding=encoding)
    except OSError:
        pass
    return None

def _read_spectrum_table(input_path):
    # --- Handle files with a [Data] section marker (e.g. WITec exports) ---
    for enc in ("utf-8-sig", "latin-1"):
        data_start = _find_data_section(input_path, encoding=enc)
        if data_start is not None:
            try:
                df = pd.read_csv(
                    input_path,
                    sep=r"\s+",
                    engine="python",
                    skiprows=data_start,
                    header=None,
                    encoding=enc,
                    encoding_errors="replace",
                )
                if df.empty or df.shape[1] < 2:
                    continue
                # Leading whitespace on data lines causes an all-NaN first column — drop it
                df = df.dropna(axis=1, how="all")
                if df.empty or df.shape[1] < 2:
                    continue
                xy_df = df.iloc[:, :2].copy()
                xy_df.iloc[:, 0] = pd.to_numeric(xy_df.iloc[:, 0], errors="coerce")
                xy_df.iloc[:, 1] = pd.to_numeric(xy_df.iloc[:, 1], errors="coerce")
                xy_df = xy_df.dropna()
                if not xy_df.empty:
                    return (
                        xy_df.iloc[:, 0].to_numpy(dtype=np.float64),
                        xy_df.iloc[:, 1].to_numpy(dtype=np.float64),
                    )
            except Exception:
                continue

    read_attempts = [
        {"sep": None, "header": 0, "engine": "python", "encoding": "utf-8-sig", "encoding_errors": "replace"},
        {"sep": None, "header": None, "engine": "python", "encoding": "utf-8-sig", "encoding_errors": "replace"},
        {"sep": ",", "header": 0, "encoding": "utf-8-sig", "encoding_errors": "replace"},
        {"sep": ",", "header": None, "encoding": "utf-8-sig", "encoding_errors": "replace"},
        {"sep": ",", "header": 0, "encoding": "latin-1"},
        {"sep": ",", "header": None, "encoding": "latin-1"},
        {"sep": r"\s+", "header": 0, "engine": "python", "encoding": "latin-1"},
        {"sep": r"\s+", "header": None, "engine": "python", "encoding": "latin-1"},
    ]
    known_x_names = {"#wave", "wave", "wavelength", "raman shift", "raman_shift", "wavenumber"}
    known_y_names = {"#intensity", "intensity", "counts", "signal"}
    best_candidate = None
    errors = []

    for options in read_attempts:
        try:
            df = pd.read_csv(input_path, **options)
        except Exception as exc:
            errors.append(f"{options}: {exc}")
            continue

        if df.empty or df.shape[1] < 2:
            continue

        columns = list(df.columns)
        normalised_names = {str(col).strip().lower(): col for col in columns}

        x_col = next((normalised_names[name] for name in known_x_names if name in normalised_names), columns[0])
        remaining_columns = [col for col in columns if col != x_col]
        y_col = next((normalised_names[name] for name in known_y_names if name in normalised_names and normalised_names[name] != x_col), None)
        if y_col is None:
            y_col = remaining_columns[0] if remaining_columns else None

        if y_col is None:
            continue

        xy_df = df[[x_col, y_col]].copy()
        xy_df[x_col] = pd.to_numeric(xy_df[x_col], errors="coerce")
        xy_df[y_col] = pd.to_numeric(xy_df[y_col], errors="coerce")
        xy_df = xy_df.dropna()

        if best_candidate is None or len(xy_df) > len(best_candidate["data"]):
            best_candidate = {
                "data": xy_df,
                "x_col": x_col,
                "y_col": y_col,
                "options": options,
            }

    if best_candidate is None or best_candidate["data"].empty:
        details = "; ".join(errors) if errors else "No parse attempt produced two numeric columns."
        raise ValueError(
            "[!] Could not load two numeric spectrum columns from the input file. "
            "Check whether the file is comma- or whitespace-delimited, whether it has a header row, "
            "and whether the x-axis units are what the pipeline expects (for example Raman shift vs wavelength). "
            f"Details: {details}"
        )

    df_xy = best_candidate["data"]
    x_values = df_xy[best_candidate["x_col"]].to_numpy(dtype=np.float64)
    y_values = df_xy[best_candidate["y_col"]].to_numpy(dtype=np.float64)

    if x_values.size == 0 or y_values.size == 0:
        raise ValueError(
            "[!] The loaded spectrum is empty after parsing numeric x/y columns. "
            "Check the input file format and confirm the x-axis units are correct."
        )

    return x_values, y_values

def preprocess(
    input_path,
    crop_min=170,
    crop_max=2000,
    sg_window=11,
    sg_polyorder=3,
    imodpoly_order=5,
    imodpoly_tol=1e-3,
    imodpoly_max_iter=1000,
    normalisation="vector-0to1",
    plot=True,
    save_path=None,
    convert_wavelength_to_shift=True,
    microm=False
):
    """
    Preprocesses a Raman spectrum with denoising, baseline removal, and normalisation.
    Returns x and y as numpy arrays.

    Normalisation options:
        - "vector"       → L2 norm
        - "vector-0to1"  → L2 norm + rescale to [0,1]
        - "min-max"      → rescale to [0,1]
        - "max"          → divide by max(y)
    """

    # === Load and clean CSV ===
    x_raw, y_raw = _read_spectrum_table(input_path)

    # Optional: Convert wavelength to Raman shift
    if convert_wavelength_to_shift:
        excitation_nm = 532
        x_raw = wavelength_to_shift(x_raw, excitation_nm, microm)

   # Sort raw arrays (after optional wavelength→shift conversion)
    sort_idx = np.argsort(x_raw)
    x_raw = x_raw[sort_idx]
    y_raw = y_raw[sort_idx]

    # Apply cropping on the arrays that will actually be used
    mask = (x_raw >= crop_min) & (x_raw <= crop_max)
    x_use = x_raw[mask]
    y_use = y_raw[mask]

    if x_use.size == 0:
        raise ValueError(
            f"[!] No data within [{crop_min}, {crop_max}] after conversion. "
            "Check units (cm^-1 vs nm) and convert_wavelength_to_shift/microm settings."
        )

    raw_spectrum = rp.Spectrum(y_use, x_use)

    # Save raw spectrum as CSV
    # raw_df = pd.DataFrame({'Wavenumber (cm^-1)': x_raw, 'Intensity (a.u.)': y_raw})
    # raw_df.to_csv("output/raw_spectrum_Tofix.csv", index=False)


    # === Apply preprocessing ===
    denoiser = preprocessing.denoise.SavGol(window_length=sg_window, polyorder=sg_polyorder)
    baseline = preprocessing.baseline.IModPoly(poly_order=imodpoly_order, tol=imodpoly_tol, max_iter=imodpoly_max_iter)
    s_denoised = denoiser.apply(raw_spectrum)
    s_baseline = baseline.apply(s_denoised)

    norm_mode = normalisation.lower()

    if norm_mode == "vector":
        normaliser = preprocessing.normalise.Vector()
        s_processed = normaliser.apply(s_baseline)

    elif norm_mode == "max":
        normaliser = preprocessing.normalise.MaxIntensity()
        s_processed = normaliser.apply(s_baseline)

    elif norm_mode == "vector-0to1":
        vec_norm = preprocessing.normalise.Vector().apply(s_baseline)
        y_scaled = min_max_normalise_array(vec_norm.spectral_data)
        s_processed = rp.Spectrum(y_scaled, vec_norm.spectral_axis)

    elif norm_mode == "min-max":
        y_scaled = min_max_normalise_array(s_baseline.spectral_data)
        s_processed = rp.Spectrum(y_scaled, s_baseline.spectral_axis)

    elif norm_mode == 'none':  
        print("No normalisation method will be applied")
        s_processed = s_baseline

    else:
        raise ValueError(f"[!] Unknown normalisation method: {normalisation}")

    x_proc = s_processed.spectral_axis
    y_proc = s_processed.spectral_data

    # === Plot raw vs processed ===
    if plot:
        plt.figure(figsize = (12, 6), dpi = 120)
        plt.subplot(1, 2, 1)
        rp.plot.spectra(raw_spectrum, title="Raw Spectrum")
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity")

        plt.subplot(1, 2, 2)
        rp.plot.spectra(s_processed, title="Processed Spectrum")
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Normalised Intensity")

        plt.tight_layout()
        plt.show()

    # === Save processed CSV ===
    if save_path:
        df_out = pd.DataFrame({
            "Raman Shift (cm-1)": x_proc,
            "Processed Intensity": y_proc
        })
        df_out.to_csv(save_path, index=False)
        print(f"[OK] Processed spectrum saved to: {save_path}")

    return x_proc, y_proc
