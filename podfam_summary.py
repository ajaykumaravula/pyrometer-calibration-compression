"""
=============================================================================
podfam_summary.py  —  PODFAM Industrial Dataset: Full Analysis + Summary
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

WHAT THIS SCRIPT DOES:
    Runs the complete analysis on the PODFAM primary dataset:

    1. Load all available PODFAM .pcd files
    2. Print dataset summary table
    3. YOUR ATP-2: Cross-file calibration curve (correct evaluation)
       — Within-file RMSE=0 is overfitting to constant TC
       — Cross-file RMSE is the real accuracy
    4. YOUR ATP-3: Compression on raw denoised signal
    5. Generate all PODFAM figures:
       - podfam_summary.png   — overview of all files
       - podfam_calibration.png — YOUR ATP-2 cross-file curve
       - podfam_compression.png — YOUR ATP-3 trade-off
       - podfam_both_pyr.png  — both pyrometers per file
    6. Save podfam_summary.csv

OUTPUTS:
    outputs/podfam/podfam_summary.png
    outputs/podfam/podfam_calibration.png
    outputs/podfam/podfam_compression.png
    outputs/podfam/podfam_both_pyr.png
    outputs/podfam_summary.csv

HOW TO RUN:
    python podfam_summary.py
    python podfam_summary.py --podfam_dir uploads/ --output outputs/
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

from load_podfam import load_pcd_file, PODFAM_FILES
from compress    import (delta_compress, delta_decompress,
                          wavelet_compress, wavelet_decompress,
                          compression_ratio, reconstruction_rmse)

PODFAM_DIR = "/home/ajay/Downloads/"
OUTPUT_DIR = "/home/ajay/linux-home/Videos/THesis_folder_26/outputs/"
PODFAM_OUT = "/home/ajay/linux-home/Videos/THesis_folder_26/outputs/podfam/"


def atp1_denoise(s):
    return gaussian_filter1d(
        median_filter(s.astype(np.float64), size=7), sigma=2.0)


def cross_file_calibration(means, refs):
    """Fit linear calibration curve across files."""
    A = np.vstack([means, np.ones(len(means))]).T
    a, b = np.linalg.lstsq(A, refs, rcond=None)[0]
    preds = a * means + b
    rmse_val = float(np.sqrt(np.mean((preds - refs)**2)))
    return a, b, preds, rmse_val


def run_podfam_summary(podfam_dir, output_dir, podfam_out):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(podfam_out, exist_ok=True)

    # ── Step 1: Load all available PODFAM files ────────────────────────
    print("=" * 65)
    print("  PODFAM Industrial Dataset — Full Analysis")
    print("=" * 65)

    print("\n  Loading PODFAM files...")
    file_data  = []
    cal_points = []  # (ref_temp, pyr1_mean, pyr1_std, pyr2_mean, pyr2_std)

    for fname, ref in PODFAM_FILES:
        path = os.path.join(podfam_dir, fname)
        if not os.path.exists(path):
            dot_fname = fname.replace('_', '.')
            path = os.path.join(podfam_dir, dot_fname)
        if not os.path.exists(path):
            print(f"    [skip] {fname}"); continue

        T1, T2, T_tc, time_s, meta = load_pcd_file(path)
        T1_den = atp1_denoise(T1)
        T2_den = atp1_denoise(T2)

        file_data.append({
            "fname": fname, "ref": ref,
            "T1": T1, "T2": T2, "T1_den": T1_den, "T2_den": T2_den,
            "T_tc": T_tc, "time_s": time_s, "meta": meta,
        })
        cal_points.append((ref, T1.mean(), T1.std(), T2.mean(), T2.std()))
        print(f"    {fname}: TC={ref:.0f}°C  "
              f"Pyr1={T1.mean():.1f}±{T1.std():.1f}°C  "
              f"Pyr2={T2.mean():.1f}±{T2.std():.1f}°C  "
              f"err={T1.mean()-ref:+.1f}°C")

    if not file_data:
        print("  No PODFAM files found.")
        return

    # ── Step 2: YOUR ATP-2 cross-file calibration ──────────────────────
    refs   = np.array([p[0] for p in cal_points])
    means1 = np.array([p[1] for p in cal_points])
    stds1  = np.array([p[2] for p in cal_points])
    means2 = np.array([p[3] for p in cal_points])
    stds2  = np.array([p[4] for p in cal_points])

    a1, b1, p1, e1 = cross_file_calibration(means1, refs)
    a2, b2, p2, e2 = cross_file_calibration(means2, refs)

    print(f"\n  YOUR ATP-2 Cross-file Calibration:")
    print(f"    Pyr1: TC = {a1:.4f} × Pyr1_mean + {b1:.2f}  "
          f"cross-file RMSE = {e1:.2f}°C")
    print(f"    Pyr2: TC = {a2:.4f} × Pyr2_mean + {b2:.2f}  "
          f"cross-file RMSE = {e2:.2f}°C")
    print(f"\n    NOTE: Within-file RMSE≈0 is overfitting to constant TC.")
    print(f"    Cross-file RMSE is the REAL calibration accuracy.")

    print(f"\n    {'TC_ref':>8}  {'Pyr1_mean':>10}  {'Pyr1_pred':>10}  "
          f"{'Pyr1_err':>9}  {'Pyr2_err':>9}")
    print(f"    {'─'*52}")
    for i, (ref, m1, _, m2, _) in enumerate(cal_points):
        print(f"    {ref:>8.0f}  {m1:>10.1f}  {p1[i]:>10.1f}  "
              f"{p1[i]-ref:>+9.1f}°C  {p2[i]-ref:>+9.1f}°C")

    # ── Step 3: YOUR ATP-3 compression on raw denoised signal ──────────
    print(f"\n  YOUR ATP-3 Compression (on raw denoised Pyr1):")
    comp_results = []
    T_den_sample = file_data[0]["T1_den"]

    for step, label in [(None,"delta_lossless"),(0.1,"delta_0.1C"),
                         (0.5,"delta_0.5C"),(1.0,"delta_1.0C")]:
        c   = delta_compress(T_den_sample, quantize_step=step)
        r   = delta_decompress(c)
        rat = compression_ratio(T_den_sample, c)
        err = reconstruction_rmse(T_den_sample, r)
        print(f"    {label:<18} ratio={rat:.1f}×  recon={err:.4f}°C")
        comp_results.append({"Method":label,"Type":"Delta",
                               "Ratio":round(rat,1),"ReconRMSE":round(err,4)})

    for kf, label in [(0.20,"wavelet_20pct"),(0.10,"wavelet_10pct"),
                       (0.05,"wavelet_5pct")]:
        c   = wavelet_compress(T_den_sample, keep_fraction=kf)
        r   = wavelet_decompress(c)
        rat = compression_ratio(T_den_sample, c)
        err = reconstruction_rmse(T_den_sample, r)
        print(f"    {label:<18} ratio={rat:.1f}×  recon={err:.4f}°C")
        comp_results.append({"Method":label,"Type":"Wavelet",
                               "Ratio":round(rat,1),"ReconRMSE":round(err,4)})

    # ── Save summary CSV ───────────────────────────────────────────────
    df_summary = pd.DataFrame({
        "File":       [d["fname"] for d in file_data],
        "TC_ref_C":   [d["ref"] for d in file_data],
        "Pyr1_mean":  [d["T1"].mean().round(2) for d in file_data],
        "Pyr1_std":   [d["T1"].std().round(2) for d in file_data],
        "Pyr1_err":   [(d["T1"].mean()-d["ref"]).round(2) for d in file_data],
        "Pyr2_mean":  [d["T2"].mean().round(2) for d in file_data],
        "Pyr2_std":   [d["T2"].std().round(2) for d in file_data],
        "Pyr2_err":   [(d["T2"].mean()-d["ref"]).round(2) for d in file_data],
        "Pyr1_cal_pred": np.round(p1, 1),
        "Pyr2_cal_pred": np.round(p2, 1),
        "Pyr1_ATP2_err": np.round(p1-refs, 2),
        "Pyr2_ATP2_err": np.round(p2-refs, 2),
    })
    df_summary.to_csv(os.path.join(output_dir, "podfam_summary.csv"), index=False)
    print(f"\n  Saved: podfam_summary.csv")

    # ── FIGURE 1: PODFAM overview ─────────────────────────────────────
    _plot_podfam_summary(file_data, refs, podfam_out)

    # ── FIGURE 2: YOUR ATP-2 calibration curve ────────────────────────
    _plot_atp2_calibration(file_data, refs, means1, means2,
                            a1, b1, p1, e1, a2, b2, p2, e2, podfam_out)

    # ── FIGURE 3: YOUR ATP-3 compression ─────────────────────────────
    _plot_atp3_compression(file_data[0], comp_results, podfam_out)

    # ── FIGURE 4: Both pyrometers per file ────────────────────────────
    _plot_both_pyrometers(file_data, refs, podfam_out)

    print(f"\n  PODFAM analysis complete. Outputs saved to {podfam_out}")


def _plot_podfam_summary(file_data, refs, out):
    """Overview: all 10 files, both pyrometers, TC reference."""
    n = len(file_data)
    cols = min(5, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4*cols, 3.5*rows_n),
                              facecolor='white')
    if rows_n == 1: axes = [axes] if cols == 1 else list(axes)
    else: axes = [ax for row in axes for ax in row]
    fig.suptitle("PODFAM Dataset — Overview: Both Pyrometers vs TC Reference\n"
                 "Primary Dataset (2-Pyrometer + TC Experiment)",
                 fontsize=12, fontweight='bold')

    for ax, d in zip(axes, file_data):
        t = d["time_s"]; ref = d["ref"]
        ax.plot(t, d["T1"], color='#3498DB', lw=0.4, alpha=0.6,
                label=f"Pyr1 {d['T1'].mean():.0f}°C")
        ax.plot(t, d["T2"], color='#E07B39', lw=0.4, alpha=0.6,
                label=f"Pyr2 {d['T2'].mean():.0f}°C")
        ax.axhline(ref, color='#C0392B', lw=1.2, ls='--',
                   label=f"TC={ref:.0f}°C")
        ax.set_title(d["fname"].replace('.pcd',''), fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        ax.legend(fontsize=6)

    for ax in axes[len(file_data):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "podfam_summary.png"),
                dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: podfam_summary.png")


def _plot_atp2_calibration(file_data, refs, means1, means2,
                             a1, b1, p1, e1, a2, b2, p2, e2, out):
    """YOUR ATP-2 cross-file calibration curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.suptitle(
        "YOUR ATP-2 — Cross-File Calibration Curve  |  PODFAM Dataset\n"
        f"Pyr1 RMSE={e1:.1f}°C  |  Pyr2 RMSE={e2:.1f}°C",
        fontsize=12, fontweight='bold')

    pr = np.linspace(min(means1.min(), means2.min())-20,
                     max(means1.max(), means2.max())+20, 100)

    # Panel 1: Calibration curve scatter + fit
    axes[0].scatter(means1, refs, color='#3498DB', s=100, zorder=4,
                    label=f"Pyr1 points  RMSE={e1:.1f}°C", edgecolors='white')
    axes[0].scatter(means2, refs, color='#E07B39', s=100, marker='s', zorder=4,
                    label=f"Pyr2 points  RMSE={e2:.1f}°C", edgecolors='white')
    axes[0].plot(pr, a1*pr+b1, color='#3498DB', lw=1.8, ls='--',
                 label=f"Pyr1 fit: TC={a1:.3f}×Pyr1+{b1:.1f}")
    axes[0].plot(pr, a2*pr+b2, color='#E07B39', lw=1.8, ls='--',
                 label=f"Pyr2 fit: TC={a2:.3f}×Pyr2+{b2:.1f}")
    axes[0].plot([refs.min()-20, refs.max()+20],
                 [refs.min()-20, refs.max()+20],
                 color='#C0392B', lw=0.8, ls=':', label="Perfect fit")
    # Annotate each point
    for i, d in enumerate(file_data):
        axes[0].annotate(f"{d['ref']:.0f}°C",
                         (means1[i], refs[i]),
                         textcoords='offset points', xytext=(5,3), fontsize=7)
    axes[0].set_xlabel("Pyrometer mean reading (°C)")
    axes[0].set_ylabel("TC reference temperature (°C)")
    axes[0].set_title("Cross-file calibration (10 calibration points)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Panel 2: Prediction error per file
    x = np.arange(len(file_data))
    labels = [d["fname"][:7] for d in file_data]
    axes[1].bar(x-0.2, p1-refs, 0.35, color='#3498DB',
                label='Pyr1 prediction error', edgecolor='white')
    axes[1].bar(x+0.2, p2-refs, 0.35, color='#E07B39',
                label='Pyr2 prediction error', edgecolor='white')
    axes[1].axhline(0, color='black', lw=0.8, ls='--')
    axes[1].axhline(+e1, color='#3498DB', lw=0.8, ls=':', alpha=0.6,
                    label=f"±RMSE Pyr1 ({e1:.1f}°C)")
    axes[1].axhline(-e1, color='#3498DB', lw=0.8, ls=':', alpha=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel("Prediction error (°C)")
    axes[1].set_title("YOUR ATP-2: Per-file prediction error")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out, "podfam_calibration.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: podfam_calibration.png")


def _plot_atp3_compression(first_file, comp_results, out):
    """YOUR ATP-3 compression trade-off on PODFAM signal."""
    T_den = first_file["T1_den"]
    t     = first_file["time_s"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
    fig.suptitle(
        f"YOUR ATP-3 — Compression on PODFAM Signal  |  {first_file['fname']}\n"
        f"Raw denoised Pyr1  (TC ref = {first_file['ref']:.0f}°C)",
        fontsize=12, fontweight='bold')

    # Reconstruct signals for plotting
    recon_signals = {}
    for row in comp_results:
        if row['Type'] == 'Delta':
            step = None if 'lossless' in row['Method'] else float(
                row['Method'].replace('delta_','').replace('C',''))
            c = delta_compress(T_den, quantize_step=step)
            recon_signals[row['Method']] = (delta_decompress(c), row['Type'])
        else:
            kf = {'wavelet_20pct':0.20,'wavelet_10pct':0.10,
                  'wavelet_5pct':0.05}[row['Method']]
            c = wavelet_compress(T_den, keep_fraction=kf)
            recon_signals[row['Method']] = (wavelet_decompress(c), row['Type'])

    # Panel 1: Reconstructed signals
    axes[0].plot(t, T_den, color='#333333', lw=1.2, label="Denoised input")
    colors = ['#27AE60','#1E8449','#145A32','#2ECC71',
              '#E07B39','#A85520','#6E3210']
    for i, (name, (r, _)) in enumerate(recon_signals.items()):
        axes[0].plot(t, r, color=colors[i % len(colors)],
                     lw=0.8, ls='--', label=f"{name}")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Temp (°C)")
    axes[0].set_title("Reconstructed signals — all methods")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    # Panel 2: Trade-off scatter
    for row in comp_results:
        col = '#27AE60' if row['Type']=='Delta' else '#E07B39'
        mk  = 'o' if row['Type']=='Delta' else 's'
        axes[1].scatter(row['Ratio'], row['ReconRMSE'],
                        color=col, marker=mk, s=100, zorder=3,
                        edgecolors='white', lw=0.5)
        axes[1].annotate(row['Method'].replace('_','\n'),
                         (row['Ratio'], row['ReconRMSE']),
                         textcoords='offset points', xytext=(4,3), fontsize=8)
    axes[1].set_xlabel("Compression ratio (×)")
    axes[1].set_ylabel("Reconstruction RMSE (°C)")
    axes[1].set_title("YOUR ATP-3: Ratio vs accuracy trade-off")
    axes[1].legend(handles=[mpatches.Patch(color='#27AE60', label='Delta'),
                              mpatches.Patch(color='#E07B39', label='Wavelet')],
                   fontsize=9)
    axes[1].grid(alpha=0.3)

    # Panel 3: RMSE bar
    methods = [r['Method'] for r in comp_results]
    recons  = [r['ReconRMSE'] for r in comp_results]
    cols    = ['#27AE60' if r['Type']=='Delta' else '#E07B39'
               for r in comp_results]
    bars = axes[2].bar(methods, recons, color=cols,
                       edgecolor='white', width=0.6)
    for bar, val in zip(bars, recons):
        axes[2].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.05, f"{val:.3f}",
                     ha='center', fontsize=8)
    axes[2].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel("Reconstruction RMSE (°C)")
    axes[2].set_title("Reconstruction error per method")
    axes[2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out, "podfam_compression.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: podfam_compression.png")


def _plot_both_pyrometers(file_data, refs, out):
    """Both pyrometers side-by-side for each file."""
    n = len(file_data)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3*n), facecolor='white')
    fig.suptitle("PODFAM — Both Pyrometers per File\n"
                 "Raw signal vs TC reference",
                 fontsize=12, fontweight='bold')

    if n == 1: axes = [axes]

    for i, (row_axes, d) in enumerate(zip(axes, file_data)):
        ref = d["ref"]
        t   = d["time_s"]
        row_axes[0].plot(t, d["T1"], color='#3498DB', lw=0.5, alpha=0.8)
        row_axes[0].axhline(ref, color='#C0392B', lw=1.0, ls='--')
        row_axes[0].set_title(f"{d['fname'][:7]}  Pyr1  "
                               f"mean={d['T1'].mean():.1f}°C  "
                               f"err={d['T1'].mean()-ref:+.1f}°C",
                               fontsize=9)
        row_axes[0].set_ylabel("Temp (°C)"); row_axes[0].grid(alpha=0.3)

        row_axes[1].plot(t, d["T2"], color='#E07B39', lw=0.5, alpha=0.8)
        row_axes[1].axhline(ref, color='#C0392B', lw=1.0, ls='--')
        row_axes[1].set_title(f"{d['fname'][:7]}  Pyr2  "
                               f"mean={d['T2'].mean():.1f}°C  "
                               f"err={d['T2'].mean()-ref:+.1f}°C",
                               fontsize=9)
        row_axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "podfam_both_pyr.png"),
                dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: podfam_both_pyr.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PODFAM dataset full analysis and summary")
    parser.add_argument('--podfam_dir', type=str, default=PODFAM_DIR)
    parser.add_argument('--output',     type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    PODFAM_DIR = args.podfam_dir
    OUTPUT_DIR = args.output
    PODFAM_OUT = os.path.join(OUTPUT_DIR, "podfam")
    run_podfam_summary(PODFAM_DIR, OUTPUT_DIR, PODFAM_OUT)
