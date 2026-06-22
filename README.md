# Pyrometer Data Pre-Processing Pipeline
## Automation of pyrometer data pre-processing (denoising, calibration, compression) for metal forming and heat treatment
### Master's Thesis — University West 2026

**Author:** Ajay (Focus: ATP-2 Calibration & ATP-3 Compression)
**Collaborator (Denoising):** Sravya Reddy (Focus: ATP-1 Denoising)

## Overview

This repository contains the complete pre-processing pipeline for pyrometer
data from metal forming and heat treatment processes.

**Your work (ATP-2 + ATP-3):**
- `calibrate.py` — classical calibration: mean offset, linear, polynomial, piecewise
- `ml_calibrate.py` — ML calibration: Random Forest, MLP, Gradient Boosting, SVR
- `compress.py` — classical compression: Delta encoding, Wavelet (Haar)
- `ml_compress.py` — ML compression: VAE, Deep Autoencoder (PyTorch)

**Friend's work (ATP-1, included as prerequisite hook):**
- ATP-1 denoising is called internally as a prerequisite before your calibration

---

## Datasets

| Dataset | Role | Files |
|---------|------|-------|
| **PODFAM .pcd** | **Primary** — real 2-pyrometer + TC experiment | `001_820.pcd` … `010_990.pcd` |
| **NIST AMBench .mat** | **Open testbed** — pipeline validation | `Layer01.mat` … `Layer10.mat` |

**PODFAM:** `sensor0` = Pyrometer 1, `sensor1` = Pyrometer 2, filename number = TC reference (°C).
AP&T proprietary data was unavailable; PODFAM was provided by the university as substitute.

**NIST:** Thermal camera data from laser powder-bed fusion (IN625, 195W, 800mm/s).
Download: https://doi.org/10.18434/M32044  |  No real TC — simulated reference.

---

## Research Questions

**RQ1 (ATP-2 — Calibration):**
How effectively can classical regression methods (mean offset, linear, polynomial, piecewise)
and ML methods (Random Forest, MLP, Gradient Boosting, SVR) correct systematic pyrometer
temperature errors using a thermocouple reference signal, and which method provides
the best trade-off between calibration accuracy and computational complexity?

**RQ2 (ATP-3 — Compression):**
What compression ratio and reconstruction accuracy are achievable using Delta Encoding,
Variational Autoencoder, and Deep Autoencoder for calibrated pyrometer time-series data,
and how do these methods compare in the compression accuracy trade-off?

---

## Installation

```bash
pip install numpy scipy pandas matplotlib scikit-learn
pip install torch   # required for VAE and Deep Autoencoder
```

---

## How to Run

```bash
# Run everything — all phases in correct order
python run_all.py \
  --nist_dir path/to/nist/data/ \
  --podfam_dir path/to/podfam/pcd/files/ \
  --output outputs/

# Individual phases
python run_all.py --phase 1    # NIST: load + your ATP-2 + ATP-3 all methods
python run_all.py --phase 2    # D3: classical vs ML comparison
python run_all.py --phase 3    # D4: visualisation dashboard
python run_all.py --phase 4    # D5: trade-off analysis
python run_all.py --phase 5    # PODFAM: industrial dataset

# Individual scripts
python d3_comparison.py --nist_dir data/ --output outputs/
python d4_visualise.py  --nist_dir data/ --output outputs/
python d5_analysis.py   --nist_dir data/ --output outputs/
python podfam_summary.py --podfam_dir pcd/ --output outputs/
```

---

## File Structure

```
thesis_pipeline/
│
│  ── DATA LOADERS ──
├── load_podfam.py         PODFAM .pcd loader (primary dataset)
├── load_nist.py           NIST .mat loader (open testbed)
│
│  ── YOUR ATP-2: CALIBRATION ──
├── calibrate.py           Classical: mean offset, linear, polynomial, piecewise
├── ml_calibrate.py        ML: Random Forest, MLP, Gradient Boosting, SVR
├── classical_methods.py   All classical methods in one place
│
│  ── YOUR ATP-3: COMPRESSION ──
├── compress.py            Classical: Delta encoding, Wavelet (Haar)
├── ml_compress.py         ML: VAE, Deep Autoencoder (PyTorch)
│
│  ── DELIVERABLE SCRIPTS ──
├── d3_comparison.py       D3: Classical vs ML comparison
├── d4_visualise.py        D4: Visualisation tool
├── d5_analysis.py         D5: Accuracy trade-off analysis
├── podfam_summary.py      PODFAM: full industrial dataset analysis
├── run_all.py             Single entry point — run everything
│
├── README.md              This file
│
└── outputs/
    ├── figures/
    │   ├── Layer01_calibration.png … Layer10_calibration.png  (10 per-layer)
    │   ├── noise_level_lower_better.png
    │   ├── d3_comparison.png
    │   ├── d4_dashboard.png
    │   ├── d5_analysis.png
    │   └── results.png
    ├── models/
    │   └── vae_latent8_layer01.pth … deep_latent8_layer10.pth  (.pth files)
    ├── podfam/
    │   ├── podfam_summary.png
    │   ├── podfam_calibration.png
    │   ├── podfam_compression.png
    │   └── podfam_results.png
    ├── d3_calibration_summary.csv
    ├── d3_compress_summary.csv
    ├── atp2_calibration_summary.csv
    ├── atp3_compression_summary.csv
    ├── classical_summary.csv
    ├── d5_summary.csv
    └── podfam_summary.csv
```

---

## Deliverables Checklist

| # | Deliverable | Script | Output |
|---|-------------|--------|--------|
| D1 | Clean calibrated time-series (PODFAM) | `run_all.py --phase 5` | `podfam_summary.csv` |
| D2 | Modular pipeline with calibration hooks | `run_all.py` | All outputs |
| D3 | ML/AI architecture investigation | `d3_comparison.py` | `d3_comparison.png`, CSVs |
| D4 | Visualisation tool | `d4_visualise.py` | `d4_dashboard.png` |
| D5 | Calibration/compression analysis | `d5_analysis.py` | `d5_analysis.png`, `d5_summary.csv` |
| D6 | Git repository with comments | This repo | All `.py` files |

---

## Key Results

### RQ1 — Calibration (NIST, 10 layers averaged)
| Method | Category | RMSE (°C) | Fit time (s) |
|--------|----------|-----------|-------------|
| linear | Classical | ~86–132 | <0.001 |
| piecewise | Classical | ~88–132 | <0.001 |
| svr | ML | ~91–200 | ~0.01 |
| gradient_boosting | ML | ~95–239 | ~0.15 |
| random_forest | ML | ~101–230 | ~0.15 |
| mlp | ML | ~113–775 | ~1.5 |
| mean_offset | Classical | ~96–372 | <0.001 |

**Conclusion RQ1:** Linear calibration achieves the best accuracy-complexity trade-off.
ML methods add significant compute without consistent accuracy gains.

### RQ2 — Compression (NIST, calibrated signal)
| Method | Category | Ratio | Recon RMSE (°C) |
|--------|----------|-------|-----------------|
| delta_lossless | Classical | 1× | 0.000 |
| delta_0.1C | Classical | 4× | 0.6 |
| wavelet_10pct | Classical | 5-10× | 17-27 |
| vae_latent8 | ML | 8× | 38-176 |
| deep_latent8 | ML | 8× | 31-180 |

**Conclusion RQ2:** Delta encoding at 0.1°C quantization gives 4× compression
with near-zero reconstruction error. Best for real-time monitoring applications.

### PODFAM — Cross-file Calibration (Real Industrial Data)
- Pyr1 cross-file RMSE: **49.88°C**
- Pyr2 cross-file RMSE: **52.68°C**
- Note: Within-file RMSE≈0 is overfitting to constant TC reference —
  cross-file RMSE is the true calibration accuracy on this dataset.

---

## Important Notes

1. **TC in NIST dataset is SIMULATED** — no real thermocouple. Stated as limitation.
2. **PODFAM cross-file evaluation** — within-file RMSE=0 is overfitting.
   Use cross-file calibration curve for real accuracy.
3. **ATP-1 denoising** = friend's module, used as prerequisite only.
   Replace `atp1_denoise()` calls with `from denoise import denoise_signal`.
4. **MLP convergence** — some layers may show convergence warnings.
   Increase `max_iter` in `ml_calibrate.py` if needed.
