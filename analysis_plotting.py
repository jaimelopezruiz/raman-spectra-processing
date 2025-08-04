import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap


def plot_and_report(x, y, y_fit_total, fitted_peaks, peak_params,
                    annotate=True, stagger_labels=True,
                    font_size=9, label_offset=0.05,
                    show_components=True, save_curve_path=None,
                    save_params_path=None, show=True, show_text_plot=True):
    """
    Plot the processed spectrum with total fit and peak positions.
    Print and optionally show a monospace text plot with all peak parameters.

    Parameters:
        x, y: processed spectrum
        y_fit_total: total fitted curve
        fitted_peaks: list of (x, y) component peaks
        peak_params: list of dicts (mu, model, FWHM, Area, Relative_Intensity)
    """

        #=== Main plot with fitted peaks (cleaned + publication style) ===
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

    ax.plot(x, y, color='black', label='Processed Data', linewidth=1.2)
    ax.plot(x, y_fit_total, 'r--', label='Total Fit', linewidth=1.2)

    
    for _, y_peak in fitted_peaks:
        ax.plot(x, y_peak, linestyle=':', linewidth=1.0, label="_nolegend_")

    # if show_components:       # UNCOMMENT TO GET FULL LEGEND
    #     for i, (_, y_peak) in enumerate(fitted_peaks):
    #         model = peak_params[i]["model"]
    #         mu = peak_params[i]["mu"]
    #         if i < 6:  # show only top N components in legend
    #             ax.plot(x, y_peak, linestyle=':', linewidth=1.0, label=f'Peak {i+1} ({model}, {mu:.1f})')
    #         else:
    #             ax.plot(x, y_peak, linestyle=':', linewidth=1.0)

    if annotate:
        for row in peak_params:
            mu = row["mu"]
            ax.axvline(mu, linestyle="--", color="gray", alpha=0.4, linewidth=0.8)

    ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=12)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)

    ax.tick_params(axis='both', labelsize=10, direction='out', length=4)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    ax.set_title("Raman Spectrum with Fitted Peaks", fontsize=13, weight='bold')
    ax.legend(fontsize=9, frameon=False, loc='upper right', ncol=1)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()

        #=== Final labeled plot with staggered wavenumber annotations ===
    plt.figure(figsize=(12, 6), dpi=120)
    plt.plot(x, y, color='red', label='Processed Data')

    for i, row in enumerate(peak_params):
        mu = row["mu"]
        idx = np.abs(x - mu).argmin()
        y_offset = max(y) * (0.05 if i % 2 == 0 else 0.1)
        plt.axvline(x=mu, color='gray', linestyle='--', linewidth=1)
        plt.text(mu, y_offset, f"{mu:.1f}",
                 rotation=0, ha='center', va='bottom',
                 fontsize=9, color='black', fontweight='bold')

    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.title("Fitted Peak Centers (Wavenumbers)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


    # === Console summary ===
    print("\n--- Fitted Peak Summary ---\n")
    for row in peak_params:
        print(f"Peak {row['peak']} ({row['model']}):")
        print(f"  Center = {row['mu']:.2f} cm⁻¹")
        print(f"  FWHM   = {row['FWHM']:.2f} cm⁻¹")
        print(f"  Height = {row['Relative_Intensity']:.3f}")
        print(f"  Area   = {row['Area']:.3f}")
        print("-" * 35)

    # === Optional monospace plot of summary ===
    if show_text_plot:
        peak_text = "Fitted Peaks:\n\n"
        for row in peak_params:
            peak_text += (
                f"Peak {row['peak']} ({row['model']}):\n"
                f"  Center = {row['mu']:.2f} cm⁻¹\n"
                f"  FWHM   = {row['FWHM']:.2f} cm⁻¹\n"
                f"  Height = {row['Relative_Intensity']:.3f}\n"
                f"  Area   = {row['Area']:.3f}\n\n"
            )

        plt.figure(figsize=(6, 12), dpi=120)
        plt.axis("off")
        wrapped_text = textwrap.fill(peak_text, width=80)
        plt.text(0.01, 0.99, peak_text, fontsize=10, va='top', ha='left', family='monospace')
        plt.title("Peak Fit Summary", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # === Optional saving ===
    if save_curve_path:
        df_curve = pd.DataFrame({
            "Raman Shift (cm-1)": x,
            "Fitted Intensity": y_fit_total
        })
        df_curve.to_csv(save_curve_path, index=False)
        print(f"[✓] Fitted curve saved to: {save_curve_path}")

    if save_params_path:
        df_params = pd.DataFrame(peak_params)
        df_params = df_params[["peak", "model", "mu", "FWHM", "Area", "Relative_Intensity"]]
        df_params.columns = ["Peak", "Model", "Center (cm⁻¹)", "FWHM (cm⁻¹)", "Area", "Relative Intensity"]
        df_params.to_csv(save_params_path, index=False)
        print(f"[✓] Fitted parameters saved to: {save_params_path}")

