"""
=============================================================================
compress.py  —  Classical compression module  (ATP-3)
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

RESEARCH QUESTION (RQ2 / ATP-3):
    What compression ratio and reconstruction accuracy are achievable
    using Delta Encoding, Variational Autoencoder, and Deep Autoencoder
    for calibrated pyrometer time-series data, and how do these methods
    compare in the compression accuracy trade-off?

WHAT THIS MODULE DOES:
    Three classical compression methods plus two bonus baselines:

    1. delta_compress     — Delta encoding (PRIMARY — named in RQ2)
    2. wavelet_compress   — Haar wavelet thresholding (bonus baseline)
    3. svd_compress       — Truncated SVD for 2-D data  (bonus baseline)

HOW TO USE:
    from compress import delta_compress, delta_decompress, compression_report
    compressed = delta_compress(T_cal, quantize_step=0.1)
    T_rec      = delta_decompress(compressed)

PIPELINE POSITION:
    Calibrated_C  →  [compress.py]  →  Compressed  →  Reconstructed_C
=============================================================================
"""

import numpy as np


# =============================================================================
# METHOD 1 — DELTA ENCODING  (primary classical method for RQ2)
# =============================================================================

def delta_compress(signal: np.ndarray,
                   quantize_step: float = None):
    """
    Compress a 1-D signal using delta encoding.

    Instead of storing absolute values [800, 798, 795, ...], store
    the first value and then only the differences:
        deltas = [800, -2, -3, ...]

    For a smooth pyrometer cooling curve the differences are tiny
    (often < 1°C per sample) and compress extremely well with
    standard integer encoding.

    Optional quantization:
        If quantize_step is provided, deltas are rounded to multiples
        of that step (e.g. 0.1°C).  This introduces a small lossy
        error but reduces the number of unique delta values, making
        entropy coding even more efficient.

    Parameters
    ----------
    signal        : np.ndarray — 1-D calibrated temperature signal (°C)
    quantize_step : float | None — quantization step in °C.
                    None = lossless (float64 deltas stored as-is)
                    0.1  = round to nearest 0.1°C (recommended trade-off)
                    1.0  = round to nearest 1°C (highest compression)

    Returns
    -------
    compressed : dict with keys:
        'first_value'    — first sample (float, anchor point)
        'deltas'         — array of differences (quantized if requested)
        'quantize_step'  — step used (None if lossless)
        'length'         — original signal length
        'nonzero_deltas' — number of non-zero delta values
        'method'         — 'delta_encoding'
    """
    signal = signal.astype(np.float64)
    deltas = np.diff(signal)          # deltas[i] = signal[i+1] - signal[i]

    if quantize_step is not None:
        # Quantize: round each delta to nearest multiple of quantize_step
        deltas = np.round(deltas / quantize_step) * quantize_step

    compressed = {
        "first_value"    : float(signal[0]),
        "deltas"         : deltas,
        "quantize_step"  : quantize_step,
        "length"         : len(signal),
        "nonzero_deltas" : int((deltas != 0).sum()),
        "method"         : "delta_encoding",
    }
    return compressed


def delta_decompress(compressed: dict) -> np.ndarray:
    """
    Reconstruct the original signal from delta-encoded representation.

    Parameters
    ----------
    compressed : dict — output from delta_compress()

    Returns
    -------
    np.ndarray — reconstructed signal, same length as original
    """
    reconstructed    = np.empty(compressed["length"], dtype=np.float64)
    reconstructed[0] = compressed["first_value"]
    # Cumulative sum of deltas restores absolute values
    reconstructed[1:] = compressed["first_value"] + np.cumsum(compressed["deltas"])
    return reconstructed


# =============================================================================
# METHOD 2 — WAVELET THRESHOLDING  (bonus baseline — not in RQ2)
# =============================================================================

def wavelet_compress(signal: np.ndarray,
                     threshold: float   = None,
                     keep_fraction: float = 0.10):
    """
    Compress a 1-D signal using Haar wavelet thresholding.

    Keeps only the largest wavelet coefficients (the rest → zero).
    Implemented from scratch — no PyWavelets dependency.

    Note: this method is a bonus baseline beyond RQ2.
    The primary classical method for RQ2 is delta_compress().

    Parameters
    ----------
    signal        : 1-D temperature time-series
    threshold     : absolute threshold for zeroing coefficients.
                    If None, auto-set to keep keep_fraction of coefficients.
    keep_fraction : fraction of coefficients to keep (0–1)

    Returns
    -------
    compressed : dict with 'coeffs', 'threshold', 'length', 'nonzero', etc.
    """
    signal   = signal.astype(np.float64)
    n        = len(signal)
    n_padded = int(2 ** np.ceil(np.log2(n)))
    padded   = np.zeros(n_padded)
    padded[:n] = signal

    coeffs = _haar_forward(padded)

    if threshold is None:
        sorted_abs = np.sort(np.abs(coeffs))[::-1]
        k          = max(1, int(keep_fraction * len(coeffs)))
        threshold  = float(sorted_abs[k])

    coeffs_thresh = coeffs.copy()
    coeffs_thresh[np.abs(coeffs_thresh) < threshold] = 0.0

    return {
        "coeffs"   : coeffs_thresh,
        "threshold": threshold,
        "length"   : n,
        "nonzero"  : int((coeffs_thresh != 0).sum()),
        "n_padded" : n_padded,
        "method"   : "wavelet_haar",
    }


def wavelet_decompress(compressed: dict) -> np.ndarray:
    """Reconstruct signal from wavelet compressed dict."""
    return _haar_inverse(compressed["coeffs"])[:compressed["length"]]


def _haar_forward(x: np.ndarray) -> np.ndarray:
    """Iterative Haar wavelet forward transform."""
    x = x.copy()
    n = len(x)
    while n > 1:
        half      = n // 2
        avg       = (x[:n:2] + x[1:n:2]) / 2.0
        diff      = (x[:n:2] - x[1:n:2]) / 2.0
        x[:half]  = avg
        x[half:n] = diff
        n         = half
    return x


def _haar_inverse(x: np.ndarray) -> np.ndarray:
    """Iterative Haar wavelet inverse transform."""
    x = x.copy()
    n = 2
    while n <= len(x):
        half      = n // 2
        avg       = x[:half].copy()
        diff      = x[half:n].copy()
        x[:n:2]   = avg + diff
        x[1:n:2]  = avg - diff
        n        *= 2
    return x


# =============================================================================
# METHOD 3 — TRUNCATED SVD  (bonus baseline — not in RQ2)
# =============================================================================

def svd_compress(data: np.ndarray, rank: int = 50):
    """
    Compress a 2-D matrix using Truncated SVD (rank-k approximation).

    Best for 2-D or 3-D spatial thermal data (frames × pixels).
    For a 3-D array (rows × cols × frames), reshape:
        data_2d = data.reshape(rows * cols, frames)

    Note: bonus baseline beyond RQ2.

    Parameters
    ----------
    data : 2-D matrix (M × N)
    rank : number of singular values to keep

    Returns
    -------
    compressed : dict with 'U', 'S', 'Vt', 'rank', 'shape'
    """
    data       = data.astype(np.float64)
    U, S, Vt   = np.linalg.svd(data, full_matrices=False)
    compressed = {
        "U"    : U[:, :rank],
        "S"    : S[:rank],
        "Vt"   : Vt[:rank, :],
        "rank" : rank,
        "shape": data.shape,
        "method": "svd",
    }
    return compressed


def svd_decompress(compressed: dict) -> np.ndarray:
    """Reconstruct matrix from SVD compressed components."""
    U  = compressed["U"]
    S  = compressed["S"]
    Vt = compressed["Vt"]
    return (U * S) @ Vt


# =============================================================================
# METRICS AND REPORTING
# =============================================================================

def compression_ratio(original: np.ndarray,
                      compressed: dict) -> float:
    """
    Compute compression ratio: original_bytes / compressed_bytes.
    Higher is better (e.g. 10.0 = 10× smaller).
    """
    orig_bytes = original.nbytes
    method     = compressed.get("method", "")

    if method == "delta_encoding":
        # Store: first_value (8 bytes) + deltas array
        # Quantized deltas can be int16 (2 bytes each) if step >= 0.1
        step = compressed.get("quantize_step")
        bytes_per_delta = 2 if (step is not None and step >= 0.1) else 8
        comp_bytes = 8 + len(compressed["deltas"]) * bytes_per_delta

    elif method == "svd":
        comp_bytes = (compressed["U"].nbytes +
                      compressed["S"].nbytes +
                      compressed["Vt"].nbytes)
    else:
        # wavelet — only non-zero values
        comp_bytes = max(1, compressed["nonzero"] * 8)

    return orig_bytes / max(1, comp_bytes)


def reconstruction_rmse(original: np.ndarray,
                        reconstructed: np.ndarray) -> float:
    """RMSE between original and reconstructed signal (°C)."""
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


def reconstruction_mae(original: np.ndarray,
                       reconstructed: np.ndarray) -> float:
    """MAE between original and reconstructed signal (°C)."""
    return float(np.mean(np.abs(original - reconstructed)))


def compression_report(original: np.ndarray,
                       reconstructed: np.ndarray,
                       compressed: dict,
                       label: str = "") -> None:
    """
    Print a one-block compression summary.

    Parameters
    ----------
    original      : original calibrated signal
    reconstructed : signal after compress → decompress
    compressed    : dict from any compress function
    label         : optional display label
    """
    tag    = f"[{label}] " if label else ""
    ratio  = compression_ratio(original, compressed)
    rmse   = reconstruction_rmse(original, reconstructed)
    mae    = reconstruction_mae(original, reconstructed)
    method = compressed.get("method", "?")

    print(f"  {tag}method       : {method}")
    if method == "delta_encoding":
        step = compressed.get("quantize_step")
        print(f"  {tag}quantize_step: {step} °C  ({'lossy' if step else 'lossless'})")
        total   = len(compressed["deltas"])
        nonzero = compressed["nonzero_deltas"]
        print(f"  {tag}non-zero Δ   : {nonzero} / {total} "
              f"({100*nonzero/max(1,total):.1f}%)")
    elif method == "svd":
        print(f"  {tag}rank kept    : {compressed['rank']}")
    else:
        nz = compressed.get("nonzero", 0)
        tot= len(compressed.get("coeffs", []))
        print(f"  {tag}coeffs kept  : {nz} / {tot} ({100*nz/max(1,tot):.1f}%)")

    print(f"  {tag}orig size    : {original.nbytes/1e3:.1f} KB")
    print(f"  {tag}ratio        : {ratio:.1f}×")
    print(f"  {tag}RMSE         : {rmse:.3f} °C")
    print(f"  {tag}MAE          : {mae:.3f} °C")


# =============================================================================
# SELF-TEST  (run: python compress.py)
# =============================================================================

if __name__ == "__main__":
    import pandas as pd

    print("=" * 65)
    print("compress.py — self-test with synthetic cooling data")
    print("Chain: Synthetic Calibrated_C → Compress → Reconstruct")
    print("=" * 65)

    rng    = np.random.default_rng(42)
    n      = 2000
    time_s = np.linspace(0, 4.0, n)
    T_cal  = 100 + 800 * np.exp(-time_s / 2.5) + rng.normal(0, 1.0, n)

    print(f"\n  Signal length : {n} samples")
    print(f"  Temp range    : {T_cal.min():.1f} – {T_cal.max():.1f} °C")
    print(f"  Original size : {T_cal.nbytes} bytes")

    results = []

    # --- Delta encoding (lossless) ---
    print(f"\n{'─'*65}")
    print("  METHOD: Delta encoding (lossless)")
    c  = delta_compress(T_cal, quantize_step=None)
    r  = delta_decompress(c)
    compression_report(T_cal, r, c, label="Delta-lossless")
    results.append({"Method": "Delta (lossless)",
                    "Ratio": round(compression_ratio(T_cal, c), 1),
                    "RMSE_C": round(reconstruction_rmse(T_cal, r), 4)})

    # --- Delta encoding (quantized 0.1°C) ---
    print(f"\n{'─'*65}")
    print("  METHOD: Delta encoding (quantized 0.1°C)")
    c  = delta_compress(T_cal, quantize_step=0.1)
    r  = delta_decompress(c)
    compression_report(T_cal, r, c, label="Delta-0.1C")
    results.append({"Method": "Delta (0.1°C)",
                    "Ratio": round(compression_ratio(T_cal, c), 1),
                    "RMSE_C": round(reconstruction_rmse(T_cal, r), 4)})

    # --- Delta encoding (quantized 1.0°C) ---
    print(f"\n{'─'*65}")
    print("  METHOD: Delta encoding (quantized 1.0°C)")
    c  = delta_compress(T_cal, quantize_step=1.0)
    r  = delta_decompress(c)
    compression_report(T_cal, r, c, label="Delta-1.0C")
    results.append({"Method": "Delta (1.0°C)",
                    "Ratio": round(compression_ratio(T_cal, c), 1),
                    "RMSE_C": round(reconstruction_rmse(T_cal, r), 4)})

    # --- Wavelet (bonus baseline) ---
    for kf in [0.20, 0.10, 0.05]:
        print(f"\n{'─'*65}")
        print(f"  METHOD: Wavelet (keep {int(kf*100)}%)")
        c  = wavelet_compress(T_cal, keep_fraction=kf)
        r  = wavelet_decompress(c)
        compression_report(T_cal, r, c, label=f"Wavelet-{int(kf*100)}%")
        results.append({
            "Method": f"Wavelet keep={int(kf*100)}%",
            "Ratio" : round(compression_ratio(T_cal, c), 1),
            "RMSE_C": round(reconstruction_rmse(T_cal, r), 3),
        })

    print(f"\n{'='*65}")
    print("  COMPARISON TABLE")
    print("=" * 65)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("\n  compress.py self-test complete.")
