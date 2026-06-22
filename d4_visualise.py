"""
=============================================================================
d4_visualise.py  —  D4 Deliverable: Visualisation Tool
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

DELIVERABLE D4:
    "Simple visualisation tool(s) for raw vs processed temperature
    and basic event markers."

OUTPUTS:
    outputs/figures/d4_dashboard.png          — main dashboard
    outputs/figures/d4_comparison.png         — method comparison
    outputs/figures/results.png               — clean results summary

HOW TO RUN:
    python d4_visualise.py
    python d4_visualise.py --nist_dir data/ --output outputs/
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

from load_nist       import load_mat_layer
from calibrate       import (linear_calibration, remove_drift, rmse, mae)
from compress        import (delta_compress, delta_decompress,
                              compression_ratio, reconstruction_rmse)
from classical_methods import (run_all_classical_calibration,
                                run_all_classical_compression)

NIST_DIR   = "data/"
OUTPUT_DIR = "outputs/"
FIGS_DIR   = "outputs/figures/"


def atp1_denoise(s):
    return gaussian_filter1d(
        median_filter(s.astype(np.float64), size=7), sigma=2.0)


def detect_events(T_cal, time_s):
    """Detect peak temp and max cooling rate in calibrated signal."""
    peak_idx = int(T_cal.argmax())
    dT       = np.gradient(T_cal, time_s)
    cool_idx = int(np.argmin(dT))
    return [(time_s[peak_idx], f"Peak {T_cal[peak_idx]:.0f}°C"),
            (time_s[cool_idx], "Max cooling rate")]


def plot_dashboard(T_pyr, T_den, T_cal_best, T_rec,
                   T_tc, time_s, cal_results, comp_results,
                   best_cal, best_comp, layer_name, figs_dir):
    """
    D4 main dashboard — 6 panels showing all YOUR results.
    """
    events = detect_events(T_cal_best, time_s)

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    fig.suptitle(
        f"D4 — Visualisation Dashboard  |  NIST {layer_name}\n"
        f"YOUR ATP-2 [{best_cal}] + YOUR ATP-3 [{best_comp}]  —  YOUR WORK",
        fontsize=12, fontweight='bold')
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30,
                          left=0.07, right=0.97, top=0.92, bottom=0.06)

    # Panel 1: Main result — YOUR calibrated + reconstructed
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_s, T_den, color='#A0A0A0', lw=0.7, alpha=0.5,
             label="Denoised (ATP-1 prerequisite — friend)")
    ax1.plot(time_s, T_cal_best, color='#E07B39', lw=1.8,
             label=f"YOUR ATP-2: Calibrated [{best_cal}]  "
                   f"RMSE={rmse(T_cal_best,T_tc):.1f}°C")
    ax1.plot(time_s, T_rec, color='#27AE60', lw=1.2, ls='--',
             label=f"YOUR ATP-3: Reconstructed [{best_comp}]  "
                   f"{comp_results[best_comp]['ratio']:.1f}×  "
                   f"recon={comp_results[best_comp]['recon_rmse']:.4f}°C")
    ax1.plot(time_s, T_tc, color='#C0392B', lw=0.8, ls=':',
             label="TC reference (simulated)")
    for t_ev, lbl in events:
        ax1.axvline(t_ev, color='#9B59B6', lw=1.0, ls='-.', alpha=0.8)
        ax1.text(t_ev+time_s[-1]*0.01, T_cal_best.max()*0.97,
                 lbl, color='#9B59B6', fontsize=8, va='top')
    ax1.set_ylabel("Temperature (°C)"); ax1.set_xlabel("Time (s)")
    ax1.set_title("YOUR results: Calibrated + Reconstructed with event markers",
                  fontsize=10)
    ax1.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.9)
    ax1.grid(alpha=0.3)

    # Panel 2: All YOUR ATP-2 methods
    ax2 = fig.add_subplot(gs[1, 0])
    c_idx = {'Classical': -1, 'ML': -1}
    ci = {'Classical': ['#3498DB','#1A6FA8','#0D4F7A','#062F4A'],
          'ML':        ['#E07B39','#A85520','#6E3210','#3A1A08']}
    for name, r in cal_results.items():
        cat = r["category"]; c_idx[cat] += 1
        ax2.plot(time_s, r["T_cal"],
                 color=ci[cat][c_idx[cat] % 4],
                 lw=0.9, ls='-' if cat == 'Classical' else '--',
                 label=f"{name} ({r['rmse']:.1f}°C)")
    ax2.plot(time_s, T_tc, color='#C0392B', lw=0.8, ls=':', label="TC ref")
    ax2.set_ylabel("Temp (°C)")
    ax2.set_title("YOUR ATP-2: All calibration methods", fontsize=10)
    ax2.legend(fontsize=6, ncol=2); ax2.grid(alpha=0.3)

    # Panel 3: All YOUR ATP-3 methods
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_s, T_cal_best, color='#333333', lw=1.2,
             label=f"Calibrated input [{best_cal}]")
    colors_comp = ['#27AE60','#1E8449','#145A32','#2ECC71',
                   '#9B59B6','#7D3C98','#A569BD','#6C3483','#5B2C6F']
    for i, (name, r) in enumerate(comp_results.items()):
        ax3.plot(time_s, r["T_rec"],
                 color=colors_comp[i % len(colors_comp)],
                 lw=0.8, ls='-' if r["category"] == 'Classical' else '--',
                 label=f"{name} ({r['ratio']:.1f}×)")
    ax3.set_ylabel("Temp (°C)")
    ax3.set_title("YOUR ATP-3: All compression methods", fontsize=10)
    ax3.legend(fontsize=6, ncol=2); ax3.grid(alpha=0.3)

    # Panel 4: Calibration residuals
    ax4 = fig.add_subplot(gs[2, 0])
    c_idx2 = {'Classical': -1, 'ML': -1}
    for name, r in cal_results.items():
        cat = r["category"]; c_idx2[cat] += 1
        res = r["T_cal"] - T_tc
        ax4.plot(time_s, res,
                 color=ci[cat][c_idx2[cat] % 4],
                 lw=0.6, alpha=0.7,
                 ls='-' if cat == 'Classical' else '--',
                 label=f"{name} ({r['rmse']:.1f}°C)")
    ax4.axhline(0, color='black', lw=0.8, ls='--')
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Residual (°C)")
    ax4.set_title("YOUR ATP-2: Calibration residuals", fontsize=10)
    ax4.legend(fontsize=6, ncol=2); ax4.grid(alpha=0.3)

    # Panel 5: Compression scatter
    ax5 = fig.add_subplot(gs[2, 1])
    for name, r in comp_results.items():
        col = '#27AE60' if r["category"] == 'Classical' else '#9B59B6'
        mk  = 'o' if r["category"] == 'Classical' else 's'
        ax5.scatter(r["ratio"], r["recon_rmse"],
                    color=col, marker=mk, s=100, zorder=3,
                    edgecolors='white', lw=0.5)
        ax5.annotate(name.replace('_','\n'),
                     (r["ratio"], r["recon_rmse"]),
                     textcoords='offset points', xytext=(4,3), fontsize=6.5)
    ax5.set_xlabel("Compression ratio (×)")
    ax5.set_ylabel("Reconstruction RMSE (°C)")
    ax5.set_title("YOUR ATP-3: Compression ratio vs accuracy", fontsize=10)
    ax5.legend(handles=[
        mpatches.Patch(color='#27AE60', label='Classical'),
        mpatches.Patch(color='#9B59B6', label='ML')], fontsize=9)
    ax5.grid(alpha=0.3)

    plt.savefig(os.path.join(figs_dir, "d4_dashboard.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: d4_dashboard.png")


def plot_results_summary(layer_data_list, figs_dir):
    """
    results.png — summary of YOUR results across all layers.
    """
    if not layer_data_list:
        return

    layers  = [d["layer"] for d in layer_data_list]
    cal_rmse = [d["cal_rmse"] for d in layer_data_list]
    comp_r   = [d["comp_ratio"] for d in layer_data_list]
    recon_e  = [d["recon_rmse"] for d in layer_data_list]
    x = np.arange(len(layers))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
    fig.suptitle(
        "Results Summary — YOUR ATP-2 + ATP-3  |  NIST AMBench All Layers",
        fontsize=12, fontweight='bold')

    bars = axes[0].bar(x, cal_rmse, color='#E07B39', edgecolor='white', width=0.6)
    for bar, val in zip(bars, cal_rmse):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.5, f"{val:.1f}",
                     ha='center', fontsize=7.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(layers, rotation=45, ha='right')
    axes[0].set_ylabel("RMSE vs TC (°C)")
    axes[0].set_title("YOUR ATP-2: Best calibration RMSE per layer")
    axes[0].grid(alpha=0.3, axis='y')

    bars2 = axes[1].bar(x, comp_r, color='#27AE60', edgecolor='white', width=0.6)
    for bar, val in zip(bars2, comp_r):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.05, f"{val:.1f}×",
                     ha='center', fontsize=7.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(layers, rotation=45, ha='right')
    axes[1].set_ylabel("Compression ratio (×)")
    axes[1].set_title("YOUR ATP-3: Compression ratio per layer")
    axes[1].grid(alpha=0.3, axis='y')

    bars3 = axes[2].bar(x, recon_e, color='#3498DB', edgecolor='white', width=0.6)
    for bar, val in zip(bars3, recon_e):
        axes[2].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.01, f"{val:.3f}",
                     ha='center', fontsize=7.5)
    axes[2].set_xticks(x); axes[2].set_xticklabels(layers, rotation=45, ha='right')
    axes[2].set_ylabel("Reconstruction RMSE (°C)")
    axes[2].set_title("YOUR ATP-3: Reconstruction error per layer")
    axes[2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "results.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: results.png")


def run_d4(nist_dir, output_dir, figs_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    layer_summary = []

    for n in range(1, 11):
        path = os.path.join(nist_dir, f"Layer{n:02d}.mat")
        if not os.path.exists(path):
            print(f"  [skip] Layer{n:02d}"); continue

        T_pyr, T_tc, time_s, meta = load_mat_layer(path)
        T_den = atp1_denoise(T_pyr)
        layer = f"Layer{n:02d}"
        print(f"  Visualising {layer}...")

        # Run YOUR calibration and compression
        from ml_calibrate import (rf_calibration, mlp_calibration,
                                   gb_calibration, svr_calibration)
        cal_results = run_all_classical_calibration(T_den, T_tc)
        for name, fn in [("random_forest", rf_calibration),
                          ("mlp", mlp_calibration),
                          ("gradient_boosting", gb_calibration),
                          ("svr", svr_calibration)]:
            T_cal, info = fn(T_den, T_tc, cal_fraction=0.20)
            cal_results[name] = {
                "T_cal": T_cal,
                "rmse": round(float(rmse(T_cal, T_tc)), 3),
                "mae": round(float(mae(T_cal, T_tc)), 3),
                "fit_time": info.get("fit_time_s", 0),
                "category": "ML",
            }

        best_cal   = min(cal_results, key=lambda k: cal_results[k]["rmse"])
        T_cal_best = cal_results[best_cal]["T_cal"]

        comp_results = run_all_classical_compression(T_cal_best)
        best_comp    = min(comp_results, key=lambda k: comp_results[k]["recon_rmse"])
        T_rec        = comp_results[best_comp]["T_rec"]

        plot_dashboard(
            T_pyr, T_den, T_cal_best, T_rec,
            T_tc, time_s, cal_results, comp_results,
            best_cal, best_comp, layer, figs_dir
        )

        layer_summary.append({
            "layer"     : layer,
            "cal_rmse"  : cal_results[best_cal]["rmse"],
            "comp_ratio": comp_results[best_comp]["ratio"],
            "recon_rmse": comp_results[best_comp]["recon_rmse"],
        })

    plot_results_summary(layer_summary, figs_dir)
    print(f"\n  D4 complete. Figures saved to {figs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D4 — Visualisation tool")
    parser.add_argument('--nist_dir', type=str, default=NIST_DIR)
    parser.add_argument('--output',   type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    NIST_DIR   = args.nist_dir
    OUTPUT_DIR = args.output
    FIGS_DIR   = os.path.join(OUTPUT_DIR, "figures")
    run_d4(NIST_DIR, OUTPUT_DIR, FIGS_DIR)
