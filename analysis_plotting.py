import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Shared publication-style helpers ===
PUB_FIGSIZE = (8, 4.5)
PUB_DPI = 200

def apply_pub_style(ax, title=None, xlabel="Raman shift (cm$^{-1}$)", ylabel="Intensity (a.u.)", legend_kwargs=None):
    """Apply consistent, clean 'publication' styling to a matplotlib Axes.
    Returns legend kwargs with sensible defaults if not provided."""
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', labelsize=10, direction='out', length=4)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, weight='bold')
    if legend_kwargs is None:
        legend_kwargs = dict(fontsize=6, frameon=False, loc='upper right', ncol=1)
    return legend_kwargs


def _fmt_pm(value, err, decimals=2):
    """Format 'value ± err', tolerating missing errors."""
    if err is None or (isinstance(err, float) and not np.isfinite(err)):
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}f} ± {err:.{decimals}f}"


def plot_and_report(
    x, y, y_fit_total, fitted_peaks, peak_params,
    fit_stats=None,
    annotate=True, stagger_labels=True,
    font_size=9, label_offset=0.05,
    show_components=True, save_curve_path=None,
    save_params_path=None, save_stats_path=None,
    save_fig_path=None,
    show=True, show_text_plot=True,
    figsize=None, show_legend=True, legend_outside=True, legend_ncol=1, legend_fontsize=6
):
    """
    Plot the processed spectrum with total fit, per-peak components and a
    residual panel. Print a summary with 1-sigma uncertainties and optionally
    save the fitted curve, peak parameters and goodness-of-fit statistics.

    Args:
      - fit_stats: list of per-region dicts from fit_peaks_regionwise
      - figsize: (w, h) tuple; defaults to PUB_FIGSIZE
      - legend_outside: if True, place legend in a separate box on the right
      - legend_ncol: number of columns in the legend
      - legend_fontsize: legend text size
    """
    # === Main plot with fitted peaks + residual panel ===
    plot_figsize = figsize or PUB_FIGSIZE
    fig, (ax, ax_res) = plt.subplots(
        2, 1, sharex=True, figsize=plot_figsize, dpi=PUB_DPI,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05}
    )

    ax.plot(x, y, color='black', label='Processed Data', linewidth=1.2)
    ax.plot(x, y_fit_total, 'r--', label='Total Fit', linewidth=1.2)

    if show_components:
        for i, (_, y_peak) in enumerate(fitted_peaks):
            model = peak_params[i]["model"]
            mu = peak_params[i]["mu"]
            label = peak_params[i].get("assignment") or f'Peak {i+1}'
            ax.plot(x, y_peak, linestyle=':', linewidth=1.0, label=f'{label} ({model}, {mu:.1f})')

    if annotate:
        for row in peak_params:
            mu = row["mu"]
            ax.axvline(mu, linestyle="--", color="gray", alpha=0.4, linewidth=0.8)

    # Residual panel (data minus total fit)
    residual = np.asarray(y) - np.asarray(y_fit_total)
    ax_res.plot(x, residual, color='black', linewidth=0.8)
    ax_res.axhline(0, color='red', linestyle='--', linewidth=0.8)
    ax_res.set_ylabel("Residual", fontsize=10)
    ax_res.tick_params(axis='both', labelsize=9, direction='out', length=4)
    ax_res.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    ax_res.spines['top'].set_visible(False)
    ax_res.spines['right'].set_visible(False)

    # Apply shared style (x-label belongs to the residual panel)
    _ = apply_pub_style(ax, title="Raman Spectrum with Fitted Peaks",
                        xlabel="", ylabel="Intensity (a.u.)")
    ax_res.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=12)

    # Legend placement
    if show_legend:
        if legend_outside:
            ax.legend(
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0,
                fontsize=legend_fontsize,
                frameon=False,
                ncol=legend_ncol
            )
        else:
            ax.legend(loc='upper right', fontsize=legend_fontsize, frameon=False, ncol=legend_ncol)

    # (tight_layout is incompatible with the shared-axis gridspec; spacing is
    # set via height_ratios/hspace above)
    if save_fig_path:
        fig.savefig(save_fig_path, bbox_inches="tight")
        print(f"[OK] Figure saved to: {save_fig_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # === Final labeled plot with staggered wavenumber annotations ===
    if show:
        plt.figure(figsize=plot_figsize, dpi=120)
        plt.plot(x, y, color='red', label='Processed Data')

        for i, row in enumerate(peak_params):
            mu = row["mu"]
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
    print("\n--- Fitted Peak Summary (uncertainties are 1σ) ---\n")
    for row in peak_params:
        header = f"Peak {row['peak']} ({row['model']})"
        if row.get("assignment"):
            header += f" — {row['assignment']}"
        print(f"{header}:")
        print(f"  Center = {_fmt_pm(row['mu'], row.get('mu_err'))} cm⁻¹")
        print(f"  FWHM   = {_fmt_pm(row['FWHM'], row.get('FWHM_err'))} cm⁻¹")
        print(f"  Height = {row['Relative_Intensity']:.3f}")
        print(f"  Area   = {_fmt_pm(row['Area'], row.get('Area_err'), decimals=3)}")
        print("-" * 35)

    if fit_stats:
        print("\n--- Goodness of Fit ---\n")
        for stats in fit_stats:
            print(f"Region {stats['region']} cm⁻¹: "
                  f"R² = {stats['R2']:.4f}, RMSE = {stats['RMSE']:.4g} "
                  f"({stats['n_points']} pts, {stats['n_params']} params)")

    # === Optional monospace plot of summary ===
    if show_text_plot and show:
        peak_text = "Fitted Peaks (1σ uncertainties):\n\n"
        for row in peak_params:
            name = f"Peak {row['peak']} ({row['model']})"
            if row.get("assignment"):
                name += f"\n  {row['assignment']}"
            peak_text += (
                f"{name}:\n"
                f"  Center = {_fmt_pm(row['mu'], row.get('mu_err'))} cm⁻¹\n"
                f"  FWHM   = {_fmt_pm(row['FWHM'], row.get('FWHM_err'))} cm⁻¹\n"
                f"  Height = {row['Relative_Intensity']:.3f}\n"
                f"  Area   = {_fmt_pm(row['Area'], row.get('Area_err'), decimals=3)}\n\n"
            )

        plt.figure(figsize=(6, 12), dpi=120)
        plt.axis("off")
        plt.text(0.01, 0.99, peak_text, fontsize=10, va='top', ha='left', family='monospace')
        plt.title("Peak Fit Summary", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # === Optional saving ===
    if save_curve_path:
        df_curve = pd.DataFrame({
            "Raman Shift (cm-1)": x,
            "Fitted Intensity": y_fit_total,
            "Residual": residual
        })
        df_curve.to_csv(save_curve_path, index=False)
        print(f"[OK] Fitted curve saved to: {save_curve_path}")

    if save_params_path:
        df_params = pd.DataFrame(peak_params)
        column_order = [
            ("peak", "Peak"),
            ("model", "Model"),
            ("assignment", "Assignment"),
            ("region", "Region (cm⁻¹)"),
            ("mu", "Center (cm⁻¹)"),
            ("mu_err", "Center 1σ (cm⁻¹)"),
            ("FWHM", "FWHM (cm⁻¹)"),
            ("FWHM_err", "FWHM 1σ (cm⁻¹)"),
            ("Area", "Area"),
            ("Area_err", "Area 1σ"),
            ("Relative_Intensity", "Relative Intensity"),
            ("amp", "amp"),
            ("amp_err", "amp 1σ"),
            ("wid", "wid"),
            ("wid_err", "wid 1σ"),
            ("q", "q"),
            ("q_err", "q 1σ"),
        ]
        present = [(key, label) for key, label in column_order if key in df_params.columns]
        df_params = df_params[[key for key, _ in present]]
        df_params.columns = [label for _, label in present]
        df_params.to_csv(save_params_path, index=False)
        print(f"[OK] Fitted parameters saved to: {save_params_path}")

    if save_stats_path and fit_stats:
        df_stats = pd.DataFrame(fit_stats)
        df_stats.to_csv(save_stats_path, index=False)
        print(f"[OK] Goodness-of-fit statistics saved to: {save_stats_path}")
