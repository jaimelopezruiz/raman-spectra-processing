import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

def plot_and_report(x, y, y_fit_total, fitted_peaks, peak_params,
                    annotate=True, stagger_labels=True,
                    font_size=9, label_offset=0.05,
                    show_components=True, save_curve_path=None,
                    save_params_path=None, show=True):
    """
    Plot the processed spectrum, total fit, and optionally each fitted component.
    Also print a peak summary and optionally save data to CSV.

    Parameters:
        x (array): Raman shift axis
        y (array): Processed intensity values
        y_fit_total (array): Total fitted spectrum
        fitted_peaks (list): List of (x, y) for each component peak
        peak_params (list): List of dicts with fit parameters
        annotate (bool): Annotate peak positions
        stagger_labels (bool): Alternate label heights to avoid overlap
        font_size (int): Font size for annotations
        label_offset (float): Offset for peak label positioning
        show_components (bool): Plot individual fitted peaks
        save_curve_path (str or None): CSV output path for total fit
        save_params_path (str or None): CSV output path for parameters
        show (bool): Show plot (set False to skip during batch processing)
    """

    # === Plot full fit ===
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, color='black', label='Processed Data')
    plt.plot(x, y_fit_total, '--', color='red', label='Total Fit')

    if show_components:
        for i, (_, y_peak) in enumerate(fitted_peaks):
            model = peak_params[i]["model"]
            mu = peak_params[i]["mu"]
            plt.plot(x, y_peak, linestyle=':', label=f'Peak {i+1} ({model}, {mu:.1f})')

    if annotate:
        for i, row in enumerate(peak_params):
            mu = row["mu"]
            idx = np.abs(x - mu).argmin()
            height = y[idx]
            y_pos = height + (label_offset * (1.1 if (stagger_labels and i % 2) else 1.0))
            plt.axvline(mu, linestyle=":", color="gray", alpha=0.5)
            plt.text(mu, y_pos, f"{mu:.1f}", ha="center", va="bottom",
                     fontsize=font_size, bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.title("Fitted Raman Spectrum")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()

    # === Print summary to console ===
    print("\n--- Fitted Peak Parameters ---\n")
    for row in peak_params:
        print(f"Peak {row['peak']} ({row['model']})")
        print(f"  Center     = {row['mu']:.2f} cm⁻¹")
        print(f"  FWHM       = {row['FWHM']:.2f} cm⁻¹")
        print(f"  Area       = {row['Area']:.2f}")
        print(f"  Rel. Int.  = {row['Relative_Intensity']:.3f}")
        print("-" * 35)

    # === Save fitted curve ===
    if save_curve_path:
        df_curve = pd.DataFrame({
            "Raman Shift (cm-1)": x,
            "Fitted Intensity": y_fit_total
        })
        df_curve.to_csv(save_curve_path, index=False)
        print(f"[✓] Fitted curve saved to: {save_curve_path}")

    # === Save parameters ===
    if save_params_path:
        df_params = pd.DataFrame(peak_params)
        df_params = df_params[["peak", "model", "mu", "FWHM", "Area", "Relative_Intensity"]]
        df_params.columns = ["Peak", "Model", "Center (cm⁻¹)", "FWHM (cm⁻¹)", "Area", "Relative Intensity"]
        df_params.to_csv(save_params_path, index=False)
        print(f"[✓] Fitted parameters saved to: {save_params_path}")
