"""
=============================================================================
classical_methods.py  —  All Classical Baselines (ATP-2 + ATP-3)
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

WHAT THIS FILE DOES:
    Collects ALL classical (non-ML) methods for ATP-2 and ATP-3 in one
    place for easy comparison.

    ATP-2 Classical Calibration:
        mean_offset   — constant shift
        linear        — least-squares line
        polynomial    — degree-2 curve
        piecewise     — per-temperature-band lines

    ATP-3 Classical Compression:
        delta_lossless  — lossless delta encoding
        delta_0.1C      — delta quantized to 0.1°C
        delta_0.5C      — delta quantized to 0.5°C
        delta_1.0C      — delta quantized to 1.0°C
        wavelet_20pct   — Haar wavelet, keep 20% of coefficients
        wavelet_10pct   — Haar wavelet, keep 10%
        wavelet_5pct    — Haar wavelet, keep 5%

HOW TO USE:
    from classical_methods import run_all_classical_calibration
    from classical_methods import run_all_classical_compression
    cal_results  = run_all_classical_calibration(T_den, T_tc)
    comp_results = run_all_classical_compression(T_cal)
=============================================================================
"""

import time
import numpy as np
import pandas as pd

from calibrate import (mean_offset_calibration, linear_calibration,
                        polynomial_calibration,  piecewise_calibration,
                        remove_drift, rmse, mae)
from compress  import (delta_compress, delta_decompress,
                        wavelet_compress, wavelet_decompress,
                        compression_ratio, reconstruction_rmse)


# =============================================================================
# ATP-2 — ALL CLASSICAL CALIBRATION METHODS
# =============================================================================

CLASSICAL_CALIBRATION_METHODS = [
    ("mean_offset",  mean_offset_calibration, {}),
    ("linear",       linear_calibration,       {}),
    ("polynomial",   polynomial_calibration,   {"degree": 2}),
    ("piecewise",    piecewise_calibration,    {}),
]


def run_all_classical_calibration(T_den: np.ndarray,
                                    T_tc: np.ndarray,
                                    cal_fraction: float = 0.20) -> dict:
    """
    Run all 4 classical ATP-2 calibration methods.

    Parameters
    ----------
    T_den        : denoised pyrometer signal (°C)  — ATP-1 output
    T_tc         : thermocouple reference (°C)
    cal_fraction : calibration window fraction

    Returns
    -------
    dict: {method_name: {T_cal, rmse, mae, fit_time, category}}
    """
    results = {}
    for name, fn, kw in CLASSICAL_CALIBRATION_METHODS:
        t0 = time.perf_counter()
        T_cal, info = fn(T_den, T_tc, cal_fraction=cal_fraction, **kw)
        T_cal = remove_drift(T_cal, T_tc)
        elapsed = time.perf_counter() - t0
        results[name] = {
            "T_cal"    : T_cal,
            "rmse"     : round(float(rmse(T_cal, T_tc)), 3),
            "mae"      : round(float(mae(T_cal,  T_tc)), 3),
            "fit_time" : round(elapsed, 5),
            "category" : "Classical",
            "info"     : info,
        }
    return results


def classical_calibration_summary(results: dict,
                                   label: str = "") -> pd.DataFrame:
    """
    Build a summary DataFrame from run_all_classical_calibration output.

    Parameters
    ----------
    results : dict from run_all_classical_calibration()
    label   : optional label (e.g. layer name)

    Returns
    -------
    pd.DataFrame with columns: Method, Category, RMSE_C, MAE_C, Time_s
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "Label"    : label,
            "Method"   : name,
            "Category" : r["category"],
            "RMSE_C"   : r["rmse"],
            "MAE_C"    : r["mae"],
            "Time_s"   : r["fit_time"],
        })
    return pd.DataFrame(rows).sort_values("RMSE_C").reset_index(drop=True)


# =============================================================================
# ATP-3 — ALL CLASSICAL COMPRESSION METHODS
# =============================================================================

CLASSICAL_COMPRESSION_METHODS = [
    ("delta_lossless", "delta", None),
    ("delta_0.1C",     "delta", 0.1),
    ("delta_0.5C",     "delta", 0.5),
    ("delta_1.0C",     "delta", 1.0),
    ("wavelet_20pct",  "wavelet", 0.20),
    ("wavelet_10pct",  "wavelet", 0.10),
    ("wavelet_5pct",   "wavelet", 0.05),
]


def run_all_classical_compression(T_cal: np.ndarray) -> dict:
    """
    Run all 7 classical ATP-3 compression methods.

    Parameters
    ----------
    T_cal : calibrated signal (°C)  — ATP-2 output

    Returns
    -------
    dict: {method_name: {T_rec, ratio, recon_rmse, category}}
    """
    results = {}
    for name, method_type, param in CLASSICAL_COMPRESSION_METHODS:
        if method_type == "delta":
            c   = delta_compress(T_cal, quantize_step=param)
            r   = delta_decompress(c)
            rat = compression_ratio(T_cal, c)
        else:
            c   = wavelet_compress(T_cal, keep_fraction=param)
            r   = wavelet_decompress(c)
            rat = compression_ratio(T_cal, c)

        results[name] = {
            "T_rec"     : r,
            "ratio"     : round(rat, 1),
            "recon_rmse": round(reconstruction_rmse(T_cal, r), 4),
            "category"  : "Classical",
            "type"      : method_type,
        }
    return results


def classical_compression_summary(results: dict,
                                   label: str = "") -> pd.DataFrame:
    """
    Build a summary DataFrame from run_all_classical_compression output.
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "Label"      : label,
            "Method"     : name,
            "Type"       : r["type"],
            "Category"   : r["category"],
            "Ratio_x"    : r["ratio"],
            "ReconRMSE_C": r["recon_rmse"],
        })
    return (pd.DataFrame(rows)
            .sort_values("ReconRMSE_C")
            .reset_index(drop=True))


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("classical_methods.py — self-test with synthetic data")
    rng    = np.random.default_rng(42)
    n      = 2000
    t      = np.linspace(0, 4, n)
    T_true = 100 + 800 * np.exp(-t / 2.5)
    T_den  = 0.92 * T_true + 15 + rng.normal(0, 3, n)
    T_tc   = T_true + rng.normal(0, 1.5, n)

    print("\n--- ATP-2 Classical Calibration ---")
    cal = run_all_classical_calibration(T_den, T_tc)
    df_cal = classical_calibration_summary(cal, label="synthetic")
    print(df_cal[["Method","RMSE_C","MAE_C","Time_s"]].to_string(index=False))

    best = min(cal, key=lambda k: cal[k]["rmse"])
    T_cal = cal[best]["T_cal"]

    print("\n--- ATP-3 Classical Compression ---")
    comp = run_all_classical_compression(T_cal)
    df_comp = classical_compression_summary(comp, label="synthetic")
    print(df_comp[["Method","Ratio_x","ReconRMSE_C"]].to_string(index=False))
