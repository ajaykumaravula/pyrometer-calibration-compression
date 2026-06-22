"""
=============================================================================
d3_comparison.py  —  D3 Deliverable: ML/AI Architecture Investigation
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

DELIVERABLE D3:
    "Investigation of ML/AI architectures for denoising,
    calibration and compression."

YOUR PART (ATP-2 + ATP-3):
    Calibration  — 4 classical + 4 ML methods compared
    Compression  — 4 classical delta + 3 classical wavelet + 2 ML compared

OUTPUTS:
    outputs/figures/d3_comparison.png          — main comparison figure
    outputs/figures/d3_calibration_all.png     — calibration detail per layer
    outputs/figures/d3_compression_all.png     — compression detail
    outputs/d3_calibration_summary.csv         — calibration results table
    outputs/d3_compress_summary.csv            — compression results table

HOW TO RUN:
    python d3_comparison.py
    python d3_comparison.py --nist_dir data/ --output outputs/
=============================================================================
"""

import os, sys, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import median_filter, gaussian_filter1d

sys.path.insert(0, os.path.dirname(__file__))

from load_nist      import load_mat_layer
from calibrate      import (mean_offset_calibration, linear_calibration,
                             polynomial_calibration, piecewise_calibration,
                             remove_drift, rmse, mae)
from ml_calibrate   import (rf_calibration, mlp_calibration,
                             gb_calibration, svr_calibration)
from compress       import (delta_compress, delta_decompress,
                             wavelet_compress, wavelet_decompress,
                             compression_ratio, reconstruction_rmse)
from classical_methods import (run_all_classical_calibration,
                                run_all_classical_compression)

NIST_DIR   = "data/"
OUTPUT_DIR = "outputs/"
FIGS_DIR   = "outputs/figures/"


def atp1_denoise(s):
    """ATP-1 prerequisite — friend's module."""
    return gaussian_filter1d(
        median_filter(s.astype(np.float64), size=7), sigma=2.0)


def run_all_ml_calibration(T_den, T_tc):
    """Run all 4 ML calibration methods."""
    results = {}
    for name, fn in [("random_forest", rf_calibration),
                      ("mlp",          mlp_calibration),
                      ("gradient_boosting", gb_calibration),
                      ("svr",          svr_calibration)]:
        T_cal, info = fn(T_den, T_tc, cal_fraction=0.20)
        results[name] = {
            "T_cal"    : T_cal,
            "rmse"     : round(float(rmse(T_cal, T_tc)), 3),
            "mae"      : round(float(mae(T_cal,  T_tc)), 3),
            "fit_time" : info.get("fit_time_s", 0),
            "category" : "ML",
        }
    return results


def run_all_ml_compression(T_cal):
    """Run VAE + Deep Autoencoder compression."""
    results = {}
    try:
        from ml_compress import train_autoencoder, compress_and_reconstruct
        for mt, lat in [("vae", 8), ("deep", 8)]:
            model, scaler, info = train_autoencoder(
                T_cal, model_type=mt, latent_dim=lat,
                window_size=64, stride=4, epochs=60)
            T_rec = compress_and_reconstruct(
                model, scaler, T_cal, window_size=64, stride=1)
            name = f"{mt}_latent{lat}"
            results[name] = {
                "T_rec"     : T_rec,
                "ratio"     : round(64 / lat, 1),
                "recon_rmse": round(reconstruction_rmse(T_cal, T_rec), 4),
                "category"  : "ML",
                "type"      : mt,
            }
            # Save model weights
            import torch
            os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, "models",
                                    f"{name}_weights.pth"))
            print(f"    Saved model: models/{name}_weights.pth")
    except Exception as e:
        print(f"    [ML compress skipped: {e}]")
    return results


def run_d3(nist_dir, output_dir, figs_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    all_cal_rows  = []
    all_comp_rows = []
    layer_data    = {}

    # ── Run all 10 NIST layers ─────────────────────────────────────────
    for n in range(1, 11):
        path = os.path.join(nist_dir, f"Layer{n:02d}.mat")
        if not os.path.exists(path):
            print(f"  [skip] Layer{n:02d}"); continue

        print(f"\n  Layer{n:02d} ------------------------------------------")
        T_pyr, T_tc, time_s, meta = load_mat_layer(path)
        T_den = atp1_denoise(T_pyr)
        layer = f"Layer{n:02d}"

        # YOUR ATP-2: all calibration methods
        print(f"  YOUR ATP-2 calibration...")
        cal_c = run_all_classical_calibration(T_den, T_tc)
        cal_m = run_all_ml_calibration(T_den, T_tc)
        all_cal = {**cal_c, **cal_m}

        for name, r in all_cal.items():
            print(f"    {name:<22} RMSE={r['rmse']:.2f}C  "
                  f"cat={r['category']}")
            all_cal_rows.append({"Layer": layer, "Method": name,
                                   "Category": r["category"],
                                   "RMSE_C": r["rmse"],
                                   "MAE_C": r["mae"],
                                   "Time_s": r.get("fit_time", 0)})

        best_cal   = min(all_cal, key=lambda k: all_cal[k]["rmse"])
        T_cal_best = all_cal[best_cal]["T_cal"]

        # YOUR ATP-3: all compression methods
        print(f"  YOUR ATP-3 compression...")
        comp_c = run_all_classical_compression(T_cal_best)
        comp_m = run_all_ml_compression(T_cal_best)
        all_comp = {**comp_c, **comp_m}

        for name, r in all_comp.items():
            print(f"    {name:<22} ratio={r['ratio']:.1f}x  "
                  f"recon={r['recon_rmse']:.4f}C")
            all_comp_rows.append({"Layer": layer, "Method": name,
                                    "Category": r["category"],
                                    "Ratio_x": r["ratio"],
                                    "ReconRMSE_C": r["recon_rmse"]})

        layer_data[layer] = {
            "T_pyr": T_pyr, "T_den": T_den, "T_tc": T_tc,
            "time_s": time_s, "meta": meta,
            "cal": all_cal, "comp": all_comp,
            "T_cal_best": T_cal_best, "best_cal": best_cal,
        }

    # ── Build summary DataFrames ───────────────────────────────────────
    df_cal = (pd.DataFrame(all_cal_rows)
              .groupby(["Method","Category"])
              .agg(RMSE_C=("RMSE_C","mean"),
                   MAE_C=("MAE_C","mean"),
                   Time_s=("Time_s","mean"))
              .reset_index().sort_values("RMSE_C"))
    df_cal.to_csv(os.path.join(output_dir, "d3_calibration_summary.csv"),
                  index=False)

    df_comp = (pd.DataFrame(all_comp_rows)
               .groupby(["Method","Category"])
               .agg(Ratio_x=("Ratio_x","mean"),
                    ReconRMSE_C=("ReconRMSE_C","mean"))
               .reset_index().sort_values("ReconRMSE_C"))
    df_comp.to_csv(os.path.join(output_dir, "d3_compress_summary.csv"),
                   index=False)

    print("\n  D3 Calibration Summary (avg 10 layers):")
    print(df_cal.to_string(index=False))
    print("\n  D3 Compression Summary (avg 10 layers):")
    print(df_comp.to_string(index=False))

    # ── FIGURE: d3_comparison.png ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    fig.suptitle(
        "D3 — Classical vs ML: YOUR ATP-2 Calibration + ATP-3 Compression\n"
        "NIST AMBench (10 layers averaged)  —  YOUR WORK",
        fontsize=12, fontweight='bold')

    cols_cal = ['#3498DB' if c == 'Classical' else '#E07B39'
                for c in df_cal['Category']]
    bars = axes[0].barh(df_cal['Method'], df_cal['RMSE_C'],
                        color=cols_cal, edgecolor='white', height=0.6)
    for bar, val in zip(bars, df_cal['RMSE_C']):
        axes[0].text(bar.get_width()+1,
                     bar.get_y()+bar.get_height()/2,
                     f"{val:.1f}°C", va='center', fontsize=8)
    axes[0].set_xlabel("RMSE vs TC (°C)")
    axes[0].set_title("RQ1: YOUR ATP-2 Calibration",
                       fontsize=11, fontweight='bold')
    axes[0].legend(handles=[
        mpatches.Patch(color='#3498DB', label='Classical'),
        mpatches.Patch(color='#E07B39', label='ML')], fontsize=9)
    axes[0].grid(alpha=0.3, axis='x')

    for _, row in df_cal.iterrows():
        col = '#3498DB' if row['Category'] == 'Classical' else '#E07B39'
        mk  = 'o' if row['Category'] == 'Classical' else 's'
        axes[1].scatter(row['Time_s'], row['RMSE_C'],
                        color=col, marker=mk, s=120, zorder=3,
                        edgecolors='white', lw=0.8)
        axes[1].annotate(row['Method'],
                         (row['Time_s'], row['RMSE_C']),
                         textcoords='offset points', xytext=(5,4), fontsize=8)
    axes[1].set_xlabel("Fit time (s)")
    axes[1].set_ylabel("RMSE (°C)")
    axes[1].set_title("RQ1: Accuracy vs complexity trade-off",
                       fontsize=11, fontweight='bold')
    axes[1].grid(alpha=0.3)

    for _, row in df_comp.iterrows():
        col = '#27AE60' if row['Category'] == 'Classical' else '#9B59B6'
        mk  = 'o' if row['Category'] == 'Classical' else 's'
        axes[2].scatter(row['Ratio_x'], row['ReconRMSE_C'],
                        color=col, marker=mk, s=120, zorder=3,
                        edgecolors='white', lw=0.8)
        axes[2].annotate(row['Method'].replace('_','\n'),
                         (row['Ratio_x'], row['ReconRMSE_C']),
                         textcoords='offset points', xytext=(5,3), fontsize=7)
    axes[2].set_xlabel("Compression ratio (×)")
    axes[2].set_ylabel("Recon RMSE (°C)")
    axes[2].set_title("RQ2: YOUR ATP-3 Compression trade-off",
                       fontsize=11, fontweight='bold')
    axes[2].legend(handles=[
        mpatches.Patch(color='#27AE60', label='Classical'),
        mpatches.Patch(color='#9B59B6', label='ML')], fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "d3_comparison.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ── FIGURE: per-layer calibration noise level ──────────────────────
    _plot_noise_level(layer_data, figs_dir)

    # ── FIGURE: 10 per-layer calibration pngs ─────────────────────────
    for layer, L in layer_data.items():
        _plot_layer_calibration(layer, L, figs_dir)

    # ── FIGURE: compression methods overview ───────────────────────────
    if layer_data:
        first_layer = list(layer_data.values())[0]
        _plot_compression_methods(first_layer, figs_dir)

    print(f"\n  D3 complete. Outputs saved to {output_dir}")
    return df_cal, df_comp, layer_data


def _plot_noise_level(layer_data, figs_dir):
    """Noise level (std) before and after calibration — lower is better."""
    layers = list(layer_data.keys())
    raw_std = []
    cal_std = {}

    for layer, L in layer_data.items():
        raw_std.append(float(np.std(L["T_pyr"])))
        for name, r in L["cal"].items():
            if name not in cal_std:
                cal_std[name] = []
            cal_std[name].append(float(np.std(r["T_cal"])))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle("Noise Level (Signal Std Dev) — Lower is Better\n"
                 "YOUR ATP-2 Calibration Effect on Signal Noise",
                 fontsize=12, fontweight='bold')

    x = np.arange(len(layers))
    axes[0].bar(x, raw_std, color='#D0D0D0', edgecolor='white', width=0.6,
                label='Raw noise level')
    axes[0].set_xticks(x); axes[0].set_xticklabels(layers, rotation=45, ha='right')
    axes[0].set_ylabel("Signal std dev (°C)")
    axes[0].set_title("Raw signal noise level per layer")
    axes[0].grid(alpha=0.3, axis='y'); axes[0].legend()

    colors_n = ['#3498DB','#1A6FA8','#0D4F7A','#062F4A',
                '#E07B39','#A85520','#6E3210','#3A1A08']
    for i, (name, stds) in enumerate(cal_std.items()):
        col = colors_n[i % len(colors_n)]
        ls  = '-' if i < 4 else '--'
        axes[1].plot(x, stds, 'o'+ls, color=col, lw=1.2, ms=5,
                     label=f"{name}")
    axes[1].set_xticks(x); axes[1].set_xticklabels(layers, rotation=45, ha='right')
    axes[1].set_ylabel("Calibrated signal std dev (°C)")
    axes[1].set_title("YOUR ATP-2: Noise level after calibration\n(lower = better)")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "noise_level_lower_better.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: noise_level_lower_better.png")


def _plot_layer_calibration(layer_name, L, figs_dir):
    """One calibration figure per NIST layer."""
    time_s = L["time_s"]; T_tc = L["T_tc"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              facecolor='white')
    fig.suptitle(
        f"YOUR ATP-2 Calibration — NIST {layer_name}\n"
        f"All methods vs TC reference  |  "
        f"Best: {L['best_cal']}  "
        f"RMSE={L['cal'][L['best_cal']]['rmse']:.1f}°C",
        fontsize=11, fontweight='bold')

    axes[0].plot(time_s, L["T_den"], color='#A0A0A0', lw=0.6, alpha=0.5,
                 label="Denoised input (ATP-1 prerequisite)")
    axes[0].plot(time_s, T_tc, color='#C0392B', lw=0.8, ls=':',
                 label="TC ref (simulated)")
    c_idx = {'Classical': -1, 'ML': -1}
    ci = {'Classical': ['#3498DB','#1A6FA8','#0D4F7A','#062F4A'],
          'ML':        ['#E07B39','#A85520','#6E3210','#3A1A08']}
    for name, r in L["cal"].items():
        cat = r["category"]; c_idx[cat] += 1
        axes[0].plot(time_s, r["T_cal"],
                     color=ci[cat][c_idx[cat] % 4],
                     lw=0.8, ls='-' if cat=='Classical' else '--',
                     label=f"{name} ({r['rmse']:.1f}°C)")
    axes[0].set_ylabel("Temp (°C)")
    axes[0].set_title("All calibration methods", fontsize=10)
    axes[0].legend(fontsize=6, ncol=3); axes[0].grid(alpha=0.3)

    c_idx2 = {'Classical': -1, 'ML': -1}
    for name, r in L["cal"].items():
        cat = r["category"]; c_idx2[cat] += 1
        res = r["T_cal"] - T_tc
        axes[1].plot(time_s, res,
                     color=ci[cat][c_idx2[cat] % 4],
                     lw=0.6, alpha=0.7, ls='-' if cat=='Classical' else '--',
                     label=f"{name} ({r['rmse']:.1f}°C)")
    axes[1].axhline(0, color='black', lw=0.8, ls='--')
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Residual (°C)")
    axes[1].set_title("Calibration residuals", fontsize=10)
    axes[1].legend(fontsize=6, ncol=3); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"{layer_name}_calibration.png"),
                dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {layer_name}_calibration.png")


def _plot_compression_methods(L, figs_dir):
    """Compression methods overview figure."""
    time_s = L["time_s"]; T_cal = L["T_cal_best"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
    fig.suptitle("YOUR ATP-3: All Compression Methods\n"
                 "NIST Layer01 — Calibrated signal",
                 fontsize=12, fontweight='bold')

    axes[0].plot(time_s, T_cal, color='#333333', lw=1.2,
                 label="Calibrated input (YOUR ATP-2)")
    colors_comp = ['#27AE60','#1E8449','#145A32','#2ECC71',
                   '#9B59B6','#7D3C98','#A569BD','#6C3483','#5B2C6F']
    for i, (name, r) in enumerate(L["comp"].items()):
        axes[0].plot(time_s, r["T_rec"],
                     color=colors_comp[i % len(colors_comp)],
                     lw=0.7, ls='--',
                     label=f"{name} ({r['ratio']:.1f}×)")
    axes[0].set_ylabel("Temp (°C)"); axes[0].set_xlabel("Time (s)")
    axes[0].set_title("Reconstructed signals — all methods")
    axes[0].legend(fontsize=6, ncol=2); axes[0].grid(alpha=0.3)

    for name, r in L["comp"].items():
        col = '#27AE60' if r["category"]=='Classical' else '#9B59B6'
        mk  = 'o' if r["category"]=='Classical' else 's'
        axes[1].scatter(r["ratio"], r["recon_rmse"],
                        color=col, marker=mk, s=100, zorder=3,
                        edgecolors='white', lw=0.5)
        axes[1].annotate(name.replace('_','\n'),
                         (r["ratio"], r["recon_rmse"]),
                         textcoords='offset points', xytext=(4,3), fontsize=7)
    axes[1].set_xlabel("Compression ratio (×)")
    axes[1].set_ylabel("Reconstruction RMSE (°C)")
    axes[1].set_title("Ratio vs accuracy trade-off")
    axes[1].legend(handles=[
        mpatches.Patch(color='#27AE60', label='Classical'),
        mpatches.Patch(color='#9B59B6', label='ML')], fontsize=9)
    axes[1].grid(alpha=0.3)

    methods = list(L["comp"].keys())
    ratios  = [L["comp"][m]["ratio"] for m in methods]
    recons  = [L["comp"][m]["recon_rmse"] for m in methods]
    cols    = ['#27AE60' if L["comp"][m]["category"]=='Classical'
               else '#9B59B6' for m in methods]
    bars = axes[2].bar(methods, recons, color=cols,
                       edgecolor='white', width=0.6)
    for bar, val in zip(bars, recons):
        axes[2].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.1,
                     f"{val:.3f}", ha='center', fontsize=7)
    axes[2].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel("Reconstruction RMSE (°C)")
    axes[2].set_title("Reconstruction error — all methods")
    axes[2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "d3_compression_all.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: d3_compression_all.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="D3 — Classical vs ML comparison")
    parser.add_argument('--nist_dir', type=str, default=NIST_DIR)
    parser.add_argument('--output',   type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    NIST_DIR   = args.nist_dir
    OUTPUT_DIR = args.output
    FIGS_DIR   = os.path.join(OUTPUT_DIR, "figures")
    run_d3(NIST_DIR, OUTPUT_DIR, FIGS_DIR)
