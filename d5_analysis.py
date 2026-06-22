"""
=============================================================================
d5_analysis.py  —  D5 Deliverable: Accuracy Trade-Off Analysis
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

DELIVERABLE D5:
    "Brief analysis of how calibration/compression choices affect
    temperature accuracy."

ANALYSES:
    5a  YOUR ATP-2 calibration window sensitivity
    5b  YOUR ATP-3 compression ratio vs reconstruction accuracy
    5c  YOUR ATP-2 choice affects YOUR ATP-3 end-to-end accuracy

OUTPUTS:
    outputs/figures/d5_analysis.png    — full analysis figure
    outputs/d5_summary.csv            — all analysis results

HOW TO RUN:
    python d5_analysis.py
    python d5_analysis.py --nist_dir data/ --output outputs/
=============================================================================
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import median_filter, gaussian_filter1d

sys.path.insert(0, os.path.dirname(__file__))

from load_nist  import load_mat_layer
from calibrate  import (mean_offset_calibration, linear_calibration,
                         polynomial_calibration, piecewise_calibration,
                         remove_drift, rmse, mae)
from compress   import (delta_compress, delta_decompress,
                         wavelet_compress, wavelet_decompress,
                         compression_ratio, reconstruction_rmse)

NIST_DIR   = "data/"
OUTPUT_DIR = "outputs/"
FIGS_DIR   = "outputs/figures/"


def atp1_denoise(s):
    return gaussian_filter1d(
        median_filter(s.astype(np.float64), size=7), sigma=2.0)

def _rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))


def analysis_5a_window_sensitivity(T_den, T_tc):
    """
    5a: How does the calibration window fraction affect YOUR ATP-2 RMSE?
    Tests cal_fraction from 5% to 50%.
    """
    rows = []
    for frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for name, fn, kw in [
            ("linear",     linear_calibration,    {}),
            ("polynomial", polynomial_calibration, {"degree": 2}),
            ("piecewise",  piecewise_calibration,  {}),
        ]:
            T_c, _ = fn(T_den, T_tc, cal_fraction=frac, **kw)
            T_c    = remove_drift(T_c, T_tc)
            rows.append({
                "Analysis"     : "5a_window_sensitivity",
                "Method"       : name,
                "CalFraction"  : frac,
                "YOUR_ATP2_RMSE_C": round(_rmse(T_c, T_tc), 3),
            })
    return pd.DataFrame(rows)


def analysis_5b_compression_tradeoff(T_cal):
    """
    5b: Compression ratio vs reconstruction RMSE for YOUR ATP-3.
    """
    rows = []
    for step, label in [(None,"delta_lossless"),(0.05,"delta_0.05C"),
                         (0.1,"delta_0.1C"),(0.5,"delta_0.5C"),
                         (1.0,"delta_1.0C"),(2.0,"delta_2.0C")]:
        c = delta_compress(T_cal, quantize_step=step)
        r = delta_decompress(c)
        rows.append({
            "Analysis"          : "5b_compression_tradeoff",
            "Method"            : label,
            "Type"              : "Delta",
            "Ratio_x"           : round(compression_ratio(T_cal,c), 1),
            "YOUR_ATP3_Recon_C" : round(reconstruction_rmse(T_cal,r), 4),
        })
    for kf, label in [(0.30,"wavelet_30pct"),(0.20,"wavelet_20pct"),
                       (0.10,"wavelet_10pct"),(0.05,"wavelet_5pct"),
                       (0.02,"wavelet_2pct")]:
        c = wavelet_compress(T_cal, keep_fraction=kf)
        r = wavelet_decompress(c)
        rows.append({
            "Analysis"          : "5b_compression_tradeoff",
            "Method"            : label,
            "Type"              : "Wavelet",
            "Ratio_x"           : round(compression_ratio(T_cal,c), 1),
            "YOUR_ATP3_Recon_C" : round(reconstruction_rmse(T_cal,r), 4),
        })
    return pd.DataFrame(rows)


def analysis_5c_calibration_compression_interaction(T_den, T_tc):
    """
    5c: How YOUR ATP-2 calibration choice affects YOUR ATP-3 accuracy.
    Shows that bad calibration is NOT amplified much by compression.
    """
    rows = []
    scenarios = [
        ("Best: linear 20%",   linear_calibration,      {"cal_fraction":0.20}),
        ("Good: linear 10%",   linear_calibration,       {"cal_fraction":0.10}),
        ("Weak: linear 5%",    linear_calibration,        {"cal_fraction":0.05}),
        ("Worst: mean offset", mean_offset_calibration,  {"cal_fraction":0.20}),
    ]
    for label, fn, kw in scenarios:
        T_c, _ = fn(T_den, T_tc, **kw)
        T_c    = remove_drift(T_c, T_tc)
        cal_e  = round(_rmse(T_c, T_tc), 3)
        c      = delta_compress(T_c, quantize_step=0.1)
        r      = delta_decompress(c)
        e2e    = round(_rmse(r, T_tc), 3)
        rows.append({
            "Analysis"         : "5c_interaction",
            "Scenario"         : label,
            "YOUR_ATP2_RMSE"   : cal_e,
            "E2E_after_ATP3"   : e2e,
            "Compression_adds" : round(e2e - cal_e, 4),
        })
    return pd.DataFrame(rows)


def analysis_5d_emissivity_robustness(T_den_base, T_tc_base):
    """
    5d: How robust is YOUR ATP-2 calibration to different emissivity errors?
    """
    rows = []
    rng = np.random.default_rng(42)
    n   = len(T_den_base)
    T_true_approx = T_tc_base

    for gain_err, label in [(0.98,"2% error"),(0.95,"5% error"),
                              (0.90,"10% error"),(0.85,"15% error"),
                              (0.80,"20% error")]:
        T_pyr_sim = gain_err * T_true_approx + rng.normal(0, 5, n)
        T_den_sim = atp1_denoise(T_pyr_sim)
        for name, fn, kw in [
            ("linear",    linear_calibration,    {}),
            ("polynomial",polynomial_calibration, {"degree":2}),
        ]:
            T_c, _ = fn(T_den_sim, T_tc_base, cal_fraction=0.20, **kw)
            T_c    = remove_drift(T_c, T_tc_base)
            rows.append({
                "Analysis"    : "5d_robustness",
                "GainError"   : gain_err,
                "ErrorLabel"  : label,
                "Method"      : name,
                "RMSE_C"      : round(_rmse(T_c, T_tc_base), 3),
            })
    return pd.DataFrame(rows)


def run_d5(nist_dir, output_dir, figs_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Use Layer01
    path = os.path.join(nist_dir, "Layer01.mat")
    if not os.path.exists(path):
        print(f"  Layer01.mat not found in {nist_dir}"); return

    T_pyr, T_tc, time_s, meta = load_mat_layer(path)
    T_den = atp1_denoise(T_pyr)

    # Best calibration for compression analysis
    T_cal, _ = linear_calibration(T_den, T_tc, cal_fraction=0.20)
    T_cal     = remove_drift(T_cal, T_tc)

    print("  Running analysis 5a: window sensitivity...")
    df5a = analysis_5a_window_sensitivity(T_den, T_tc)
    print(df5a.to_string(index=False))

    print("\n  Running analysis 5b: compression trade-off...")
    df5b = analysis_5b_compression_tradeoff(T_cal)
    print(df5b.to_string(index=False))

    print("\n  Running analysis 5c: calibration × compression interaction...")
    df5c = analysis_5c_calibration_compression_interaction(T_den, T_tc)
    print(df5c.to_string(index=False))

    print("\n  Running analysis 5d: emissivity robustness...")
    df5d = analysis_5d_emissivity_robustness(T_den, T_tc)
    print(df5d.to_string(index=False))

    # Save combined CSV
    all_cols = list(set(df5a.columns) | set(df5b.columns) |
                    set(df5c.columns) | set(df5d.columns))
    df_all = pd.concat([df5a, df5b, df5c, df5d], ignore_index=True)
    df_all.to_csv(os.path.join(output_dir, "d5_summary.csv"), index=False)
    print(f"\n  Saved: d5_summary.csv")

    # ── D5 Figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='white')
    fig.suptitle(
        "D5 — Trade-Off Analysis: YOUR ATP-2 + ATP-3\n"
        "NIST AMBench Layer01  —  YOUR WORK",
        fontsize=12, fontweight='bold')

    # 5a: Window sensitivity
    ax = axes[0, 0]
    for method in df5a['Method'].unique():
        d   = df5a[df5a['Method'] == method]
        col = {'linear':'#3498DB','polynomial':'#E07B39','piecewise':'#27AE60'}[method]
        ax.plot(d['CalFraction'], d['YOUR_ATP2_RMSE_C'],
                'o-', color=col, lw=1.5, ms=6, label=method)
    ax.axvline(0.20, color='gray', lw=0.8, ls='--', alpha=0.6,
               label="Default (20%)")
    ax.set_xlabel("Calibration window fraction")
    ax.set_ylabel("YOUR ATP-2 RMSE (°C)")
    ax.set_title("5a: Calibration window sensitivity", fontsize=10,
                  fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 5b: Compression trade-off
    ax = axes[0, 1]
    for _, row in df5b.iterrows():
        col = '#27AE60' if row['Type']=='Delta' else '#E07B39'
        ax.scatter(row['Ratio_x'], row['YOUR_ATP3_Recon_C'],
                   color=col, s=80, zorder=3,
                   edgecolors='white', lw=0.5)
        ax.annotate(row['Method'].replace('_','\n'),
                    (row['Ratio_x'], row['YOUR_ATP3_Recon_C']),
                    textcoords='offset points', xytext=(4,3), fontsize=6.5)
    ax.set_xlabel("Compression ratio (×)")
    ax.set_ylabel("YOUR ATP-3 Reconstruction RMSE (°C)")
    ax.set_title("5b: Compression ratio vs accuracy", fontsize=10,
                  fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color='#27AE60', label='Delta'),
                        mpatches.Patch(color='#E07B39', label='Wavelet')],
              fontsize=9)
    ax.grid(alpha=0.3)

    # 5c: Calibration × compression interaction
    ax = axes[1, 0]
    bar_colors = ['#27AE60','#3498DB','#E07B39','#C0392B']
    bars = ax.bar(df5c['Scenario'], df5c['E2E_after_ATP3'],
                  color=bar_colors, edgecolor='white', width=0.55)
    for bar, row in zip(bars, df5c.itertuples()):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.3,
                f"ATP2={row.YOUR_ATP2_RMSE:.1f}°C\n"
                f"+{row.Compression_adds:.2f}°C",
                ha='center', fontsize=8)
    ax.set_ylabel("End-to-end RMSE after YOUR ATP-3 (°C)")
    ax.set_title("5c: YOUR ATP-2 choice → ATP-3 accuracy", fontsize=10,
                  fontweight='bold')
    ax.set_xticklabels(df5c['Scenario'], rotation=15, ha='right', fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    ax.text(0.02, 0.98,
            "'Compression_adds' shows compression barely\n"
            "amplifies calibration error",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    # 5d: Emissivity robustness
    ax = axes[1, 1]
    pivot = df5d.pivot(index='ErrorLabel', columns='Method',
                        values='RMSE_C').reset_index()
    x = np.arange(len(pivot))
    w = 0.3
    if 'linear' in pivot.columns:
        ax.bar(x-w/2, pivot['linear'], w, color='#3498DB',
               label='linear', edgecolor='white')
    if 'polynomial' in pivot.columns:
        ax.bar(x+w/2, pivot['polynomial'], w, color='#E07B39',
               label='polynomial', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot['ErrorLabel'], rotation=15, ha='right', fontsize=8)
    ax.set_ylabel("YOUR ATP-2 RMSE (°C)")
    ax.set_title("5d: Emissivity error robustness", fontsize=10,
                  fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "d5_analysis.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: d5_analysis.png")
    print(f"\n  D5 complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D5 — Trade-off analysis")
    parser.add_argument('--nist_dir', type=str, default=NIST_DIR)
    parser.add_argument('--output',   type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    NIST_DIR   = args.nist_dir
    OUTPUT_DIR = args.output
    FIGS_DIR   = os.path.join(OUTPUT_DIR, "figures")
    run_d5(NIST_DIR, OUTPUT_DIR, FIGS_DIR)
