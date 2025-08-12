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
    alex_data=False,
    microm = True
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
    # df = pd.read_csv(input_path, delim_whitespace = True, header=None, skiprows=16, engine="python", encoding="latin1")   #For our annealing data
    df = pd.read_csv(input_path, delim_whitespace = True, header=None, skiprows=16, engine="python", encoding="latin1")   #we should use sep='\s+' instead of delim_whitespace
    # df.columns = df.columns.str.strip()
    x_col, y_col = df.columns[:2]

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna()

        # Check we are not dealing with empty csv / reading it wrong
    if df.shape[0] == 0:
        raise ValueError("[!] Loaded CSV is empty. Check delimiter or file format.")

    x_raw = df[x_col].values
    y_raw = df[y_col].values

    # Optional: Convert wavelength to Raman shift
    if alex_data:
        excitation_nm = 532
        x_raw = wavelength_to_shift(x_raw, excitation_nm, microm)

    df = df[(df[x_col] >= crop_min) & (df[x_col] <= crop_max)]
    df = df.sort_values(by=x_col)

    # Sort (again, safe in case conversion shuffled order)
    sort_idx = np.argsort(x_raw)
    x_raw = x_raw[sort_idx]
    y_raw = y_raw[sort_idx]


    raw_spectrum = rp.Spectrum(y_raw, x_raw)

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
        print(f"[✓] Processed spectrum saved to: {save_path}")

    return x_proc, y_proc
