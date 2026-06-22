"""
run_all.py — Master Pipeline: YOUR ATP-2 + ATP-3 Results Only
Your work: ATP-2 (calibration) + ATP-3 (compression)
NOT yours: ATP-1 (denoising) — friend's module, used as prerequisite only

RUN ORDER:
  Phase 1 — NIST: load data + run YOUR ATP-2 + ATP-3 (all methods)
  Phase 2 — D3: classical vs ML comparison (YOUR results only)
  Phase 3 — D4: visualisation dashboard (YOUR calibrated + reconstructed)
  Phase 4 — D5: trade-off analysis (YOUR calibration + compression choices)
  Phase 5 — PODFAM: industrial dataset

HOW TO RUN:
  python run_all.py                         # full run all phases
  python run_all.py --phase 1 2 3 4 5       # explicit
  python run_all.py --phase 5               # PODFAM only
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import median_filter, gaussian_filter1d

sys.path.insert(0, os.path.dirname(__file__))

NIST_DIR   = "/home/ajay/linux-home/Videos/THesis_folder_26/Pyrometer data/"
PODFAM_DIR = "/home/ajay/Downloads/"
OUTPUT_DIR = "/home/ajay/linux-home/Videos/THesis_folder_26/outputs/thesis_run_v2/"

from load_nist    import load_mat_layer
from load_podfam  import load_pcd_file, PODFAM_FILES
from calibrate    import (mean_offset_calibration, linear_calibration,
                           polynomial_calibration,  piecewise_calibration,
                           remove_drift, rmse, mae)
from ml_calibrate import (rf_calibration, mlp_calibration,
                           gb_calibration, svr_calibration)
from compress     import (delta_compress, delta_decompress,
                           wavelet_compress, wavelet_decompress,
                           compression_ratio, reconstruction_rmse)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def sub(t):    print(f"\n  -- {t}")
def _rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def _mae(a,b):  return float(np.mean(np.abs(a-b)))


# =============================================================================
# ATP-1 PREREQUISITE — NOT YOUR RESULT
# =============================================================================

def atp1_denoise(signal):
    """
    ATP-1: friend's denoising module.
    Used internally as a PREREQUISITE before your calibration.
    Its RMSE improvement is NOT printed in your results.
    Replace with: from denoise import denoise_signal
    """
    return gaussian_filter1d(
        median_filter(signal.astype(np.float64), size=7), sigma=2.0
    )


# =============================================================================
# YOUR ATP-2 — ALL CALIBRATION METHODS
# =============================================================================

def run_atp2_calibration(T_den, T_tc):
    """YOUR ATP-2: run all 8 calibration methods on denoised input."""
    results = {}

    # Classical methods
    for name, fn, kw in [
        ("mean_offset",  mean_offset_calibration, {}),
        ("linear",       linear_calibration,       {}),
        ("polynomial",   polynomial_calibration,   {"degree": 2}),
        ("piecewise",    piecewise_calibration,    {}),
    ]:
        t0 = time.perf_counter()
        T_cal, _ = fn(T_den, T_tc, cal_fraction=0.20, **kw)
        T_cal = remove_drift(T_cal, T_tc)
        results[name] = {
            "T_cal": T_cal,
            "rmse": round(_rmse(T_cal, T_tc), 3),
            "mae":  round(_mae(T_cal,  T_tc), 3),
            "fit_time": round(time.perf_counter() - t0, 4),
            "category": "Classical",
        }

    # ML methods
    for name, fn in [
        ("random_forest",     rf_calibration),
        ("mlp",               mlp_calibration),
        ("gradient_boosting", gb_calibration),
        ("svr",               svr_calibration),
    ]:
        T_cal, info = fn(T_den, T_tc, cal_fraction=0.20)
        results[name] = {
            "T_cal": T_cal,
            "rmse": round(_rmse(T_cal, T_tc), 3),
            "mae":  round(_mae(T_cal,  T_tc), 3),
            "fit_time": info.get("fit_time_s", 0),
            "category": "ML",
        }

    return results


# =============================================================================
# YOUR ATP-3 — ALL COMPRESSION METHODS
# =============================================================================

def run_atp3_compression(T_cal, T_tc):
    """YOUR ATP-3: run all compression methods on calibrated input."""
    results = {}

    # Delta encoding variants
    for step, name in [
        (None, "delta_lossless"),
        (0.1,  "delta_0.1C"),
        (0.5,  "delta_0.5C"),
        (1.0,  "delta_1.0C"),
    ]:
        c = delta_compress(T_cal, quantize_step=step)
        r = delta_decompress(c)
        results[name] = {
            "T_rec": r,
            "ratio": round(compression_ratio(T_cal, c), 1),
            "recon": round(reconstruction_rmse(T_cal, r), 4),
            "category": "Classical",
        }

    # Wavelet variants
    for kf, name in [
        (0.20, "wavelet_20pct"),
        (0.10, "wavelet_10pct"),
        (0.05, "wavelet_5pct"),
    ]:
        c = wavelet_compress(T_cal, keep_fraction=kf)
        r = wavelet_decompress(c)
        results[name] = {
            "T_rec": r,
            "ratio": round(compression_ratio(T_cal, c), 1),
            "recon": round(reconstruction_rmse(T_cal, r), 4),
            "category": "Classical",
        }

    # ML: VAE + Deep Autoencoder
    try:
        from ml_compress import train_autoencoder, compress_and_reconstruct
        for mt, lat in [("vae", 8), ("deep", 8)]:
            model, scaler, info = train_autoencoder(
                T_cal, model_type=mt, latent_dim=lat,
                window_size=64, stride=4, epochs=60)
            T_rec = compress_and_reconstruct(
                model, scaler, T_cal, window_size=64, stride=1)
            results[f"{mt}_latent{lat}"] = {
                "T_rec": T_rec,
                "ratio": round(64 / lat, 1),
                "recon": round(reconstruction_rmse(T_cal, T_rec), 4),
                "category": "ML",
            }
    except Exception as e:
        print(f"    [ML compress skipped: {e}]")

    return results


# =============================================================================
# PHASE 1 — NIST PIPELINE (YOUR ATP-2 + ATP-3)
# =============================================================================

def phase1_nist_pipeline():
    banner("PHASE 1 -- NIST Data: YOUR ATP-2 Calibration + ATP-3 Compression")
    all_layers = {}

    for n in range(1, 11):
        path = os.path.join(NIST_DIR, f"Layer{n:02d}.mat")
        if not os.path.exists(path):
            print(f"  [skip] Layer{n:02d}.mat"); continue

        # Load data
        sub(f"Load NIST Layer{n:02d}")
        T_pyr, T_tc, time_s, meta = load_mat_layer(path)
        print(f"    {meta['n_frames_active']} frames | "
              f"T_max={meta['T_max_C']:.0f}C | "
              f"~{1/float(np.median(np.diff(time_s))):.0f}Hz")

        # ATP-1: PREREQUISITE ONLY — not your result, not printed
        T_den = atp1_denoise(T_pyr)

        # YOUR ATP-2 calibration
        sub(f"YOUR ATP-2 Calibration  (Layer{n:02d})")
        print(f"    Input: denoised signal (ATP-1 prerequisite from friend)")
        print(f"    {'Method':<22} {'Category':<12} "
              f"{'RMSE(C)':>9}  {'MAE(C)':>9}  {'Time(s)':>8}")
        print(f"    {'─'*64}")
        cal = run_atp2_calibration(T_den, T_tc)
        for name, r in cal.items():
            print(f"    {name:<22} {r['category']:<12} "
                  f"{r['rmse']:>9.2f}  {r['mae']:>9.2f}  {r['fit_time']:>8.4f}")

        best_cal   = min(cal, key=lambda k: cal[k]['rmse'])
        T_cal_best = cal[best_cal]['T_cal']
        print(f"\n    Best method: {best_cal}  "
              f"RMSE={cal[best_cal]['rmse']:.2f}C")
        print(f"    Using as input to YOUR ATP-3 compression")

        # YOUR ATP-3 compression
        sub(f"YOUR ATP-3 Compression  (Layer{n:02d})")
        print(f"    Input: YOUR calibrated signal [{best_cal}]")
        print(f"    {'Method':<22} {'Category':<12} "
              f"{'Ratio':>8}  {'ReconRMSE(C)':>14}")
        print(f"    {'─'*60}")
        comp = run_atp3_compression(T_cal_best, T_tc)
        for name, r in comp.items():
            print(f"    {name:<22} {r['category']:<12} "
                  f"{r['ratio']:>8.1f}x  {r['recon']:>14.4f}")

        all_layers[f"Layer{n:02d}"] = {
            "T_pyr": T_pyr, "T_den": T_den,
            "T_tc": T_tc, "time_s": time_s, "meta": meta,
            "cal": cal, "comp": comp,
            "T_cal_best": T_cal_best, "best_cal": best_cal,
        }

    # Save all results to CSV
    rows = []
    for layer, L in all_layers.items():
        for name, r in L["cal"].items():
            rows.append({"Layer": layer, "Stage": "ATP-2_calibration",
                          "Method": name, "Category": r["category"],
                          "RMSE_C": r["rmse"], "MAE_C": r["mae"],
                          "Time_s": r["fit_time"]})
        for name, r in L["comp"].items():
            rows.append({"Layer": layer, "Stage": "ATP-3_compression",
                          "Method": name, "Category": r["category"],
                          "Ratio": r["ratio"], "ReconRMSE_C": r["recon"]})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "Phase1_YOUR_results.csv"), index=False)
    print(f"\n  All YOUR results saved -> Phase1_YOUR_results.csv")
    return all_layers


# =============================================================================
# PHASE 2 — D3: CLASSICAL vs ML (YOUR RESULTS ONLY)
# =============================================================================

def phase2_d3(all_layers):
    banner("PHASE 2 -- D3: Classical vs ML (YOUR ATP-2 + ATP-3 Only)")

    # Average calibration results across all layers
    cal_rows = []
    for L in all_layers.values():
        for name, r in L["cal"].items():
            cal_rows.append({"Method": name, "Category": r["category"],
                              "RMSE_C": r["rmse"], "MAE_C": r["mae"],
                              "Time_s": r["fit_time"]})
    df_cal = (pd.DataFrame(cal_rows)
              .groupby(["Method", "Category"])
              .agg(RMSE_C=("RMSE_C","mean"),
                   MAE_C=("MAE_C","mean"),
                   Time_s=("Time_s","mean"))
              .reset_index().sort_values("RMSE_C"))
    df_cal.to_csv(os.path.join(OUTPUT_DIR, "D3_YOUR_ATP2_calibration.csv"),
                  index=False)

    # Average compression results across all layers
    comp_rows = []
    for L in all_layers.values():
        for name, r in L["comp"].items():
            comp_rows.append({"Method": name, "Category": r["category"],
                               "Ratio": r["ratio"], "ReconRMSE": r["recon"]})
    df_comp = (pd.DataFrame(comp_rows)
               .groupby(["Method", "Category"])
               .agg(Ratio=("Ratio","mean"), ReconRMSE=("ReconRMSE","mean"))
               .reset_index().sort_values("ReconRMSE"))
    df_comp.to_csv(os.path.join(OUTPUT_DIR, "D3_YOUR_ATP3_compression.csv"),
                   index=False)

    print("\n  YOUR ATP-2 Calibration (averaged, 10 NIST layers):")
    print(df_cal.to_string(index=False))
    print("\n  YOUR ATP-3 Compression (averaged, 10 NIST layers):")
    print(df_comp.to_string(index=False))

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    fig.suptitle(
        "D3 -- Classical vs ML Comparison\n"
        "YOUR ATP-2 Calibration + YOUR ATP-3 Compression "
        "(NIST AMBench, 10 layers avg)",
        fontsize=12, fontweight='bold')

    # Panel 1: YOUR ATP-2 calibration RMSE bar
    cols = ['#3498DB' if c == 'Classical' else '#E07B39'
            for c in df_cal['Category']]
    bars = axes[0].barh(df_cal['Method'], df_cal['RMSE_C'],
                        color=cols, edgecolor='white', height=0.6)
    for bar, val in zip(bars, df_cal['RMSE_C']):
        axes[0].text(bar.get_width()+1,
                     bar.get_y()+bar.get_height()/2,
                     f"{val:.1f}C", va='center', fontsize=8)
    axes[0].set_xlabel("RMSE vs TC (C) -- lower is better")
    axes[0].set_title("RQ1: YOUR ATP-2 Calibration accuracy",
                       fontsize=11, fontweight='bold')
    axes[0].legend(handles=[
        mpatches.Patch(color='#3498DB', label='Classical'),
        mpatches.Patch(color='#E07B39', label='ML')], fontsize=9)
    axes[0].grid(alpha=0.3, axis='x')

    # Panel 2: YOUR ATP-2 accuracy vs complexity
    for _, row in df_cal.iterrows():
        col = '#3498DB' if row['Category'] == 'Classical' else '#E07B39'
        mk  = 'o' if row['Category'] == 'Classical' else 's'
        axes[1].scatter(row['Time_s'], row['RMSE_C'],
                        color=col, marker=mk, s=120, zorder=3,
                        edgecolors='white', lw=0.8)
        axes[1].annotate(row['Method'],
                         (row['Time_s'], row['RMSE_C']),
                         textcoords='offset points',
                         xytext=(5, 4), fontsize=8)
    axes[1].set_xlabel("Fit time (s) -- lower is better")
    axes[1].set_ylabel("RMSE (C) -- lower is better")
    axes[1].set_title("RQ1: YOUR ATP-2 Accuracy vs complexity",
                       fontsize=11, fontweight='bold')
    axes[1].grid(alpha=0.3)

    # Panel 3: YOUR ATP-3 compression trade-off
    for _, row in df_comp.iterrows():
        col = '#27AE60' if row['Category'] == 'Classical' else '#9B59B6'
        mk  = 'o' if row['Category'] == 'Classical' else 's'
        axes[2].scatter(row['Ratio'], row['ReconRMSE'],
                        color=col, marker=mk, s=120, zorder=3,
                        edgecolors='white', lw=0.8)
        axes[2].annotate(row['Method'].replace('_','\n'),
                         (row['Ratio'], row['ReconRMSE']),
                         textcoords='offset points',
                         xytext=(5, 3), fontsize=7)
    axes[2].set_xlabel("Compression ratio (x) -- higher better")
    axes[2].set_ylabel("Recon RMSE (C) -- lower better")
    axes[2].set_title("RQ2: YOUR ATP-3 Compression trade-off",
                       fontsize=11, fontweight='bold')
    axes[2].legend(handles=[
        mpatches.Patch(color='#27AE60', label='Classical'),
        mpatches.Patch(color='#9B59B6', label='ML')], fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "D3_classical_vs_ml.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  D3 figure saved -> D3_classical_vs_ml.png")


# =============================================================================
# PHASE 3 — D4: VISUALISATION DASHBOARD
# Shows: denoised input (context) -> YOUR calibrated -> YOUR reconstructed
# =============================================================================

def phase3_d4(all_layers):
    banner("PHASE 3 -- D4: Visualisation Dashboard (YOUR ATP-2 + ATP-3)")

    key = "Layer01" if "Layer01" in all_layers else list(all_layers.keys())[0]
    L   = all_layers[key]
    time_s     = L["time_s"]
    T_den      = L["T_den"]
    T_tc       = L["T_tc"]
    T_cal_best = L["T_cal_best"]
    best_cal   = L["best_cal"]

    # Best classical compression for display
    classical = {k: v for k, v in L["comp"].items()
                 if v["category"] == "Classical"}
    best_comp = min(classical, key=lambda k: classical[k]["recon"])
    T_rec     = classical[best_comp]["T_rec"]

    # Event markers on YOUR calibrated signal
    peak_idx = int(T_cal_best.argmax())
    dT       = np.gradient(T_cal_best, time_s)
    events   = [(time_s[peak_idx], f"Peak {T_cal_best[peak_idx]:.0f}C"),
                (time_s[int(np.argmin(dT))], "Max cooling rate")]

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    fig.suptitle(
        f"D4 -- Visualisation Dashboard | NIST {key} | "
        f"YOUR ATP-2 [{best_cal}] + YOUR ATP-3 [{best_comp}]",
        fontsize=12, fontweight='bold')
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30,
                          left=0.07, right=0.97, top=0.92, bottom=0.06)

    # Panel 1: YOUR main results
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_s, T_den, color='#A0A0A0', lw=0.8, alpha=0.6,
             label="Denoised (ATP-1 prerequisite -- friend's output, your input)")
    ax1.plot(time_s, T_cal_best, color='#E07B39', lw=1.8,
             label=f"YOUR ATP-2: Calibrated [{best_cal}]  "
                   f"RMSE={_rmse(T_cal_best,T_tc):.1f}C")
    ax1.plot(time_s, T_rec, color='#27AE60', lw=1.2, ls='--',
             label=f"YOUR ATP-3: Reconstructed [{best_comp}]  "
                   f"{classical[best_comp]['ratio']:.1f}x  "
                   f"recon={classical[best_comp]['recon']:.4f}C")
    ax1.plot(time_s, T_tc, color='#C0392B', lw=0.8, ls=':',
             label="TC reference (simulated)")
    for t_ev, lbl in events:
        ax1.axvline(t_ev, color='#9B59B6', lw=1.0, ls='-.', alpha=0.8)
        ax1.text(t_ev + time_s[-1]*0.01, T_cal_best.max()*0.97,
                 lbl, color='#9B59B6', fontsize=8, va='top')
    ax1.set_ylabel("Temp (C)"); ax1.set_xlabel("Time (s)")
    ax1.set_title(
        "YOUR ATP-2 calibrated + ATP-3 reconstructed "
        "(denoised = friend's prerequisite, shown as context only)",
        fontsize=10)
    ax1.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.9)
    ax1.grid(alpha=0.3)

    # Panel 2: All YOUR ATP-2 methods overlaid
    ax2 = fig.add_subplot(gs[1, 0])
    c_idx = {'Classical': -1, 'ML': -1}
    ci = {'Classical': ['#3498DB','#1A6FA8','#0D4F7A','#062F4A'],
          'ML':        ['#E07B39','#A85520','#6E3210','#3A1A08']}
    for name, r in L["cal"].items():
        cat = r["category"]; c_idx[cat] += 1
        ax2.plot(time_s, r["T_cal"],
                 color=ci[cat][c_idx[cat] % 4],
                 lw=0.9, ls='-' if cat=='Classical' else '--',
                 label=f"{name} ({r['rmse']}C)")
    ax2.plot(time_s, T_tc, color='#C0392B', lw=0.8, ls=':', label="TC ref")
    ax2.set_title("YOUR ATP-2: All calibration methods", fontsize=10)
    ax2.legend(fontsize=6, ncol=2); ax2.grid(alpha=0.3); ax2.set_ylabel("Temp (C)")

    # Panel 3: All YOUR ATP-3 methods overlaid
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_s, T_cal_best, color='#555555', lw=1.2,
             label=f"YOUR calibrated input [{best_cal}]")
    colors_c = ['#27AE60','#1E8449','#145A32','#2ECC71',
                '#9B59B6','#7D3C98','#A569BD','#6C3483','#5B2C6F']
    for i, (name, r) in enumerate(L["comp"].items()):
        ax3.plot(time_s, r["T_rec"],
                 color=colors_c[i % len(colors_c)],
                 lw=0.8, ls='-' if r["category"]=='Classical' else '--',
                 label=f"{name} ({r['ratio']}x)")
    ax3.set_title("YOUR ATP-3: All compression methods", fontsize=10)
    ax3.legend(fontsize=6, ncol=2); ax3.grid(alpha=0.3); ax3.set_ylabel("Temp (C)")

    # Panel 4: YOUR ATP-2 calibration residuals
    ax4 = fig.add_subplot(gs[2, 0])
    for name, r in L["cal"].items():
        res = r["T_cal"] - T_tc
        col = '#3498DB' if r["category"]=='Classical' else '#E07B39'
        ax4.plot(time_s, res, color=col, lw=0.6, alpha=0.7,
                 label=f"{name} ({r['rmse']}C)")
    ax4.axhline(0, color='black', lw=0.8, ls='--')
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Residual (C)")
    ax4.set_title("YOUR ATP-2: Calibration residuals", fontsize=10)
    ax4.legend(fontsize=6, ncol=2); ax4.grid(alpha=0.3)

    # Panel 5: YOUR ATP-3 compression scatter
    ax5 = fig.add_subplot(gs[2, 1])
    for name, r in L["comp"].items():
        col = '#27AE60' if r["category"]=='Classical' else '#9B59B6'
        mk  = 'o' if r["category"]=='Classical' else 's'
        ax5.scatter(r["ratio"], r["recon"],
                    color=col, marker=mk, s=100, zorder=3,
                    edgecolors='white', lw=0.5)
        ax5.annotate(name.replace('_','\n'), (r["ratio"], r["recon"]),
                     textcoords='offset points', xytext=(4,3), fontsize=6.5)
    ax5.set_xlabel("Compression ratio (x)")
    ax5.set_ylabel("Recon RMSE (C)")
    ax5.set_title("YOUR ATP-3: Compression ratio vs accuracy", fontsize=10)
    ax5.legend(handles=[
        mpatches.Patch(color='#27AE60', label='Classical'),
        mpatches.Patch(color='#9B59B6', label='ML')], fontsize=9)
    ax5.grid(alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, "D4_dashboard.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  D4 dashboard saved -> D4_dashboard.png")


# =============================================================================
# PHASE 4 — D5: TRADE-OFF ANALYSIS
# =============================================================================

def phase4_d5(all_layers):
    banner("PHASE 4 -- D5: Trade-Off Analysis (YOUR ATP-2 + ATP-3 choices)")

    key   = "Layer01" if "Layer01" in all_layers else list(all_layers.keys())[0]
    T_den = all_layers[key]["T_den"]
    T_tc  = all_layers[key]["T_tc"]
    T_cal = all_layers[key]["T_cal_best"]

    # 4a: YOUR ATP-2 calibration window sensitivity
    sub("4a -- YOUR ATP-2: How window size affects calibration accuracy")
    rows = []
    for frac in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        for name, fn, kw in [
            ("linear",     linear_calibration,    {}),
            ("polynomial", polynomial_calibration, {"degree": 2}),
        ]:
            T_c, _ = fn(T_den, T_tc, cal_fraction=frac, **kw)
            T_c    = remove_drift(T_c, T_tc)
            rows.append({"Method": name, "CalFraction": frac,
                          "YOUR_ATP2_RMSE_C": round(_rmse(T_c, T_tc), 3)})
    df4a = pd.DataFrame(rows)
    df4a.to_csv(os.path.join(OUTPUT_DIR, "D5_YOUR_ATP2_window_sensitivity.csv"),
                index=False)
    print(df4a.to_string(index=False))

    # 4b: YOUR ATP-3 compression ratio vs accuracy
    sub("4b -- YOUR ATP-3: Compression ratio vs reconstruction accuracy")
    rows2 = []
    for step, label in [(None,"delta_lossless"),(0.1,"delta_0.1C"),
                         (0.5,"delta_0.5C"),(1.0,"delta_1.0C"),(2.0,"delta_2.0C")]:
        c = delta_compress(T_cal, quantize_step=step)
        r = delta_decompress(c)
        rows2.append({"Method": label, "Type": "Delta",
                       "Ratio": round(compression_ratio(T_cal,c),1),
                       "YOUR_ATP3_ReconRMSE": round(reconstruction_rmse(T_cal,r),4)})
    for kf, label in [(0.30,"wavelet_30pct"),(0.20,"wavelet_20pct"),
                       (0.10,"wavelet_10pct"),(0.05,"wavelet_5pct"),
                       (0.02,"wavelet_2pct")]:
        c = wavelet_compress(T_cal, keep_fraction=kf)
        r = wavelet_decompress(c)
        rows2.append({"Method": label, "Type": "Wavelet",
                       "Ratio": round(compression_ratio(T_cal,c),1),
                       "YOUR_ATP3_ReconRMSE": round(reconstruction_rmse(T_cal,r),4)})
    df4b = pd.DataFrame(rows2)
    df4b.to_csv(os.path.join(OUTPUT_DIR, "D5_YOUR_ATP3_compression_tradeoff.csv"),
                index=False)
    print(df4b.to_string(index=False))

    # 4c: How YOUR ATP-2 calibration choice affects YOUR ATP-3 compression
    sub("4c -- YOUR ATP-2 choice vs YOUR ATP-3 compression end-to-end error")
    rows3 = []
    for label, fn, kw in [
        ("Good: linear 20%",   linear_calibration,       {"cal_fraction": 0.20}),
        ("Weak: linear 5%",    linear_calibration,        {"cal_fraction": 0.05}),
        ("Worst: mean offset", mean_offset_calibration,  {"cal_fraction": 0.20}),
    ]:
        T_c, _ = fn(T_den, T_tc, **kw)
        T_c    = remove_drift(T_c, T_tc)
        c      = delta_compress(T_c, quantize_step=0.1)
        r      = delta_decompress(c)
        rows3.append({"Scenario": label,
                       "YOUR_ATP2_RMSE": round(_rmse(T_c, T_tc), 3),
                       "E2E_RMSE_after_ATP3": round(_rmse(r, T_tc), 3),
                       "Compression_adds": round(_rmse(r,T_tc)-_rmse(T_c,T_tc), 4)})
    df4c = pd.DataFrame(rows3)
    df4c.to_csv(os.path.join(OUTPUT_DIR, "D5_YOUR_ATP2xATP3_interaction.csv"),
                index=False)
    print(df4c.to_string(index=False))
    print("  'Compression_adds' = RMSE your ATP-3 adds on top of your ATP-2 error")

    # D5 Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
    fig.suptitle(
        "D5 -- Trade-Off Analysis: YOUR ATP-2 + ATP-3\n"
        "NIST AMBench Layer01",
        fontsize=12, fontweight='bold')

    for method in df4a['Method'].unique():
        d   = df4a[df4a['Method'] == method]
        col = '#3498DB' if method == 'linear' else '#E07B39'
        axes[0].plot(d['CalFraction'], d['YOUR_ATP2_RMSE_C'],
                     'o-', color=col, lw=1.5, ms=6, label=method)
    axes[0].set_xlabel("Calibration window fraction")
    axes[0].set_ylabel("YOUR ATP-2 RMSE (C)")
    axes[0].set_title("4a: YOUR ATP-2 window sensitivity", fontsize=10)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    for _, row in df4b.iterrows():
        col = '#27AE60' if row['Type']=='Delta' else '#E07B39'
        axes[1].scatter(row['Ratio'], row['YOUR_ATP3_ReconRMSE'],
                        color=col, s=80, zorder=3,
                        edgecolors='white', lw=0.5)
        axes[1].annotate(row['Method'].replace('_','\n'),
                         (row['Ratio'], row['YOUR_ATP3_ReconRMSE']),
                         textcoords='offset points', xytext=(4,3), fontsize=6.5)
    axes[1].set_xlabel("Compression ratio (x)")
    axes[1].set_ylabel("YOUR ATP-3 Recon RMSE (C)")
    axes[1].set_title("4b: YOUR ATP-3 compression trade-off", fontsize=10)
    axes[1].legend(handles=[
        mpatches.Patch(color='#27AE60', label='Delta'),
        mpatches.Patch(color='#E07B39', label='Wavelet')], fontsize=9)
    axes[1].grid(alpha=0.3)

    bars = axes[2].bar(df4c['Scenario'], df4c['E2E_RMSE_after_ATP3'],
                       color=['#27AE60','#E07B39','#C0392B'],
                       edgecolor='white', width=0.5)
    for bar, row in zip(bars, df4c.itertuples()):
        axes[2].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.5,
                     f"ATP2={row.YOUR_ATP2_RMSE:.1f}C\n"
                     f"+{row.Compression_adds:.2f}C",
                     ha='center', fontsize=8)
    axes[2].set_ylabel("End-to-end RMSE after YOUR ATP-3 (C)")
    axes[2].set_title("4c: YOUR ATP-2 choice -> YOUR ATP-3 accuracy", fontsize=10)
    axes[2].set_xticklabels(df4c['Scenario'], rotation=15, ha='right', fontsize=8)
    axes[2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "D5_tradeoff.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  D5 figure saved -> D5_tradeoff.png")


# =============================================================================
# PHASE 5 — PODFAM INDUSTRIAL DATASET
# =============================================================================

def phase5_podfam():
    banner("PHASE 5 -- PODFAM Industrial Dataset (Primary Dataset)")

    def read_pcd(p, max_rows=100000):
        rows = []; done = False
        with open(p, 'r', errors='replace') as f:
            for line in f:
                line = line.strip()
                if line == 'DATA ascii': done = True; continue
                if done and line:
                    try:
                        v = list(map(int, line.split()))
                        if len(v) == 10: rows.append(v)
                        if len(rows) >= max_rows: break
                    except: continue
        return np.array(rows, dtype=np.float64)

    sub("Load PODFAM files (2-pyrometer + TC experiment)")
    pts = []; avail = []
    for fname, ref in PODFAM_FILES:
        path = os.path.join(PODFAM_DIR, fname)
        if not os.path.exists(path):
            dot_fname = fname.replace('_', '.')
            path = os.path.join(PODFAM_DIR, dot_fname)
        if not os.path.exists(path):
            print(f"    [skip] {fname}"); continue
        data = read_pcd(path)
        s0 = data[:, 4]; s1 = data[:, 5]; t = data[:, 0] / 1e6
        pts.append((ref, s0.mean(), s0.std(), s1.mean(), s1.std()))
        avail.append((fname, ref, s0, s1, t))
        print(f"    {fname}: TC={ref:.0f}C  "
              f"Pyr1={s0.mean():.1f}+/-{s0.std():.1f}C  "
              f"Pyr2={s1.mean():.1f}+/-{s1.std():.1f}C  "
              f"err={s0.mean()-ref:+.1f}C")

    # YOUR ATP-2: cross-file calibration (correct evaluation for PODFAM)
    sub("YOUR ATP-2: Cross-file calibration curve")
    refs   = np.array([p[0] for p in pts])
    means1 = np.array([p[1] for p in pts])
    means2 = np.array([p[3] for p in pts])

    def cross_fit(m, r):
        A = np.vstack([m, np.ones(len(m))]).T
        a, b = np.linalg.lstsq(A, r, rcond=None)[0]
        p = a * m + b
        return a, b, p, float(np.sqrt(np.mean((p-r)**2)))

    a1, b1, p1, e1 = cross_fit(means1, refs)
    a2, b2, p2, e2 = cross_fit(means2, refs)

    print(f"\n  YOUR ATP-2 cross-file result:")
    print(f"    Pyr1: TC = {a1:.4f} x Pyr1_mean + {b1:.2f}  "
          f"cross-file RMSE = {e1:.2f}C")
    print(f"    Pyr2: TC = {a2:.4f} x Pyr2_mean + {b2:.2f}  "
          f"cross-file RMSE = {e2:.2f}C")
    print(f"\n  This is the REAL calibration accuracy on PODFAM data.")
    print(f"  (Within-file RMSE=0 was overfitting to a constant TC)")

    pd.DataFrame({
        "TC_ref_C": refs,
        "Pyr1_mean": means1, "Pyr1_pred": np.round(p1,1),
        "YOUR_ATP2_Pyr1_err": np.round(p1-refs, 2),
        "Pyr2_mean": means2, "Pyr2_pred": np.round(p2,1),
        "YOUR_ATP2_Pyr2_err": np.round(p2-refs, 2),
    }).to_csv(os.path.join(OUTPUT_DIR, "D1_PODFAM_YOUR_ATP2_calibration.csv"),
              index=False)

    # YOUR ATP-3: compression on raw denoised signal
    sub("YOUR ATP-3: Compression on raw denoised Pyr1 signal")
    fname, ref, s0, s1, t = avail[0]
    T_den1 = atp1_denoise(s0)
    print(f"  File: {fname}  TC ref={ref:.0f}C")
    print(f"  Signal std={T_den1.std():.1f}C  (real variation, not constant)")
    print(f"\n  {'Method':<18} {'Type':<10} {'Ratio':>8}  {'ReconRMSE(C)':>14}")
    print(f"  {'─'*55}")
    comp_rows = []
    for step, label in [(None,"delta_lossless"),(0.1,"delta_0.1C"),
                         (0.5,"delta_0.5C"),(1.0,"delta_1.0C")]:
        c   = delta_compress(T_den1, quantize_step=step)
        r   = delta_decompress(c)
        rat = compression_ratio(T_den1, c)
        err = reconstruction_rmse(T_den1, r)
        print(f"  {label:<18} {'Delta':<10} {rat:>8.1f}x  {err:>14.4f}")
        comp_rows.append({"Method": label, "Type": "Delta",
                           "Ratio": round(rat,1),
                           "YOUR_ATP3_ReconRMSE": round(err,4)})
    for kf, label in [(0.20,"wavelet_20pct"),(0.10,"wavelet_10pct"),
                       (0.05,"wavelet_5pct")]:
        c   = wavelet_compress(T_den1, keep_fraction=kf)
        r   = wavelet_decompress(c)
        rat = compression_ratio(T_den1, c)
        err = reconstruction_rmse(T_den1, r)
        print(f"  {label:<18} {'Wavelet':<10} {rat:>8.1f}x  {err:>14.4f}")
        comp_rows.append({"Method": label, "Type": "Wavelet",
                           "Ratio": round(rat,1),
                           "YOUR_ATP3_ReconRMSE": round(err,4)})
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(OUTPUT_DIR, "D1_PODFAM_YOUR_ATP3_compression.csv"),
        index=False)

    # PODFAM Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='white')
    fig.suptitle(
        "D1 -- PODFAM Industrial Dataset | Primary Dataset\n"
        "YOUR ATP-2 Calibration + YOUR ATP-3 Compression | "
        "2-Pyrometer + Thermocouple Experiment",
        fontsize=12, fontweight='bold')

    fname_d, ref_d, s0_d, s1_d, t_d = avail[0]
    T_d1 = atp1_denoise(s0_d); T_d2 = atp1_denoise(s1_d)

    axes[0,0].plot(t_d, s0_d, color='#3498DB', lw=0.4, alpha=0.7,
                   label=f"Pyr1 raw  mean={s0_d.mean():.1f}C  err={s0_d.mean()-ref_d:+.1f}C")
    axes[0,0].plot(t_d, s1_d, color='#E07B39', lw=0.4, alpha=0.7,
                   label=f"Pyr2 raw  mean={s1_d.mean():.1f}C  err={s1_d.mean()-ref_d:+.1f}C")
    axes[0,0].axhline(ref_d, color='#C0392B', lw=1.2, ls='--',
                      label=f"TC ref={ref_d:.0f}C")
    axes[0,0].set_title(f"Raw signals ({fname_d})", fontsize=10)
    axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)
    axes[0,0].set_ylabel("Temp (C)")

    pr = np.linspace(means1.min()-20, means1.max()+20, 100)
    axes[0,1].scatter(means1, refs, color='#3498DB', s=100, zorder=4,
                      label=f"Pyr1  RMSE={e1:.1f}C")
    axes[0,1].scatter(means2, refs, color='#E07B39', s=100, marker='s',
                      zorder=4, label=f"Pyr2  RMSE={e2:.1f}C")
    axes[0,1].plot(pr, a1*pr+b1, color='#3498DB', lw=1.5, ls='--')
    axes[0,1].plot(pr, a2*pr+b2, color='#E07B39', lw=1.5, ls='--')
    axes[0,1].plot([refs.min()-20, refs.max()+20],
                   [refs.min()-20, refs.max()+20],
                   color='#C0392B', lw=0.8, ls=':', label="Perfect fit")
    axes[0,1].set_xlabel("Pyr mean reading (C)")
    axes[0,1].set_ylabel("TC reference (C)")
    axes[0,1].set_title("YOUR ATP-2: Cross-file calibration curve\n"
                         f"(correct evaluation, {len(pts)} calibration points)",
                         fontsize=10)
    axes[0,1].legend(fontsize=8); axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(t_d, T_d1, color='#555555', lw=1.0, label="Denoised Pyr1 (input)")
    for row in comp_rows:
        if row['Type'] == 'Delta':
            s = None if 'lossless' in row['Method'] else float(
                row['Method'].replace('delta_','').replace('C',''))
            c = delta_compress(T_d1, quantize_step=s)
            r = delta_decompress(c)
        else:
            kf = 0.20 if '20' in row['Method'] else 0.10 if '10' in row['Method'] else 0.05
            c  = wavelet_compress(T_d1, keep_fraction=kf)
            r  = wavelet_decompress(c)
        col = '#27AE60' if row['Type']=='Delta' else '#E07B39'
        axes[1,0].plot(t_d, r, color=col, lw=0.8, ls='--',
                       label=f"{row['Method']} {row['Ratio']}x")
    axes[1,0].set_title("YOUR ATP-3: Compression on Pyr1 signal", fontsize=10)
    axes[1,0].legend(fontsize=7); axes[1,0].grid(alpha=0.3)
    axes[1,0].set_xlabel("Time (s)"); axes[1,0].set_ylabel("Temp (C)")

    for row in comp_rows:
        col = '#27AE60' if row['Type']=='Delta' else '#E07B39'
        mk  = 'o' if row['Type']=='Delta' else 's'
        axes[1,1].scatter(row['Ratio'], row['YOUR_ATP3_ReconRMSE'],
                          color=col, marker=mk, s=100, zorder=3,
                          edgecolors='white', lw=0.5)
        axes[1,1].annotate(row['Method'].replace('_','\n'),
                           (row['Ratio'], row['YOUR_ATP3_ReconRMSE']),
                           textcoords='offset points', xytext=(4,3), fontsize=8)
    axes[1,1].set_xlabel("Compression ratio (x)")
    axes[1,1].set_ylabel("YOUR ATP-3 Recon RMSE (C)")
    axes[1,1].set_title("YOUR ATP-3: Compression trade-off (PODFAM signal)",
                         fontsize=10)
    axes[1,1].legend(handles=[
        mpatches.Patch(color='#27AE60', label='Delta'),
        mpatches.Patch(color='#E07B39', label='Wavelet')], fontsize=9)
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "D1_PODFAM.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  PODFAM figure saved -> D1_PODFAM.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master pipeline: YOUR ATP-2 + ATP-3 results only")
    parser.add_argument('--phase', type=int, nargs='+', default=[1,2,3,4,5])
    parser.add_argument('--nist_dir',   type=str, default=NIST_DIR)
    parser.add_argument('--podfam_dir', type=str, default=PODFAM_DIR)
    parser.add_argument('--output',     type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    NIST_DIR   = args.nist_dir
    PODFAM_DIR = args.podfam_dir
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  THESIS PIPELINE")
    print("  YOUR WORK : ATP-2 (calibration) + ATP-3 (compression)")
    print("  NOT YOURS : ATP-1 (denoising) = prerequisite only")
    print(f"  Phases    : {args.phase}")
    print(f"  NIST      : {NIST_DIR}")
    print(f"  PODFAM    : {PODFAM_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print("=" * 65)

    t0 = time.perf_counter()
    all_layers = {}

    if 1 in args.phase:
        all_layers = phase1_nist_pipeline()
    if 2 in args.phase:
        if not all_layers: all_layers = phase1_nist_pipeline()
        phase2_d3(all_layers)
    if 3 in args.phase:
        if not all_layers: all_layers = phase1_nist_pipeline()
        phase3_d4(all_layers)
    if 4 in args.phase:
        if not all_layers: all_layers = phase1_nist_pipeline()
        phase4_d5(all_layers)
    if 5 in args.phase:
        phase5_podfam()

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*65}")
    print(f"  COMPLETE  ({elapsed:.1f}s)  ->  {OUTPUT_DIR}")
    print("=" * 65)
    print("\n  Output files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f:<50} {sz/1e3:>8.1f} KB")
