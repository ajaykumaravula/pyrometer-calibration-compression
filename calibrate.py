"""
=============================================================================
calibrate.py  —  Classical calibration module  (ATP-2)
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

RESEARCH QUESTION (RQ1 / ATP-2):
    How effectively can classical regression methods (mean offset, linear,
    polynomial, piecewise) correct systematic pyrometer temperature errors
    using a thermocouple reference signal, and which method provides the
    best trade-off between calibration accuracy and computational complexity?

WHAT THIS MODULE DOES:
    Corrects emissivity error in pyrometer signals using a thermocouple (TC)
    as a reference. Four classical methods are implemented:

    1. mean_offset_calibration   — simplest baseline, constant shift
    2. linear_calibration        — straight-line fit  T_cal = a*T_pyr + b
    3. polynomial_calibration    — curved fit, degree 2 or 3
    4. piecewise_calibration     — separate linear fits per temperature band

    Also provides:
    - remove_drift()             — subtract slow linear sensor drift
    - rmse() / mae()             — accuracy metrics
    - calibration_report()       — print before/after comparison
    - print_preview()            — 5-row table preview

HOW TO USE:
    from calibrate import linear_calibration, calibration_report
    T_cal, coeffs = linear_calibration(T_pyr, T_tc, cal_fraction=0.20)
    calibration_report(T_pyr, T_cal, T_tc, coeffs)

SWAP FOR ML LATER:
    Replace any function here with ml_calibrate.py equivalents.
    The function signature stays the same:
        T_cal, coeffs = <method>(T_pyr, T_tc, cal_fraction=0.20)

PIPELINE POSITION:
    Raw_C  →  [denoise.py]  →  Denoised_C  →  [calibrate.py]  →  Calibrated_C
=============================================================================
"""

import numpy as np


# =============================================================================
# METHOD 1 — MEAN OFFSET  (simplest baseline)
# =============================================================================

def mean_offset_calibration(T_pyr: np.ndarray,
                             T_ref: np.ndarray,
                             cal_fraction: float = 0.20):
    """
    Correct the pyrometer by adding a single constant offset.

    The offset is the mean difference between the thermocouple and the
    pyrometer over the calibration window:
        offset = mean(T_ref - T_pyr)  over first cal_fraction of data
        T_cal  = T_pyr + offset

    This is the simplest possible calibration — a good sanity-check
    baseline. Works well when emissivity is stable and the error is
    mainly a fixed offset.

    Parameters
    ----------
    T_pyr        : np.ndarray — denoised pyrometer signal (°C)
    T_ref        : np.ndarray — thermocouple reference signal (°C)
    cal_fraction : float      — fraction of data used for fitting (0–1)

    Returns
    -------
    T_cal  : np.ndarray — calibrated temperature signal
    coeffs : dict       — {'offset': float, 'method': 'mean_offset', ...}
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))
    offset  = float(np.mean(
        T_ref[:cal_end].astype(np.float64) -
        T_pyr[:cal_end].astype(np.float64)
    ))
    T_cal  = T_pyr.astype(np.float64) + offset
    coeffs = {
        "offset"  : offset,
        "method"  : "mean_offset",
        "cal_end" : cal_end,
    }
    return T_cal, coeffs


# =============================================================================
# METHOD 2 — LINEAR REGRESSION
# =============================================================================

def linear_calibration(T_pyr: np.ndarray,
                        T_ref: np.ndarray,
                        cal_fraction: float = 0.20):
    """
    Fit a linear correction  T_ref ≈ a × T_pyr + b  on a calibration
    window, then apply it to the full signal.

    Uses ordinary least-squares (numpy lstsq). Handles both a
    proportional gain error and a fixed offset simultaneously.

    Parameters
    ----------
    T_pyr        : np.ndarray — denoised pyrometer signal (°C)
    T_ref        : np.ndarray — thermocouple reference signal (°C)
    cal_fraction : float      — fraction of data used for fitting

    Returns
    -------
    T_cal  : np.ndarray — calibrated temperature signal
    coeffs : dict       — {'a': slope, 'b': intercept, 'method': 'linear'}
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))
    x = T_pyr[:cal_end].astype(np.float64)
    y = T_ref[:cal_end].astype(np.float64)

    # Build design matrix [x | 1] and solve for [a, b]
    A      = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    a, b   = result[0]

    T_cal  = a * T_pyr.astype(np.float64) + b
    coeffs = {
        "a"       : a,
        "b"       : b,
        "method"  : "linear",
        "cal_end" : cal_end,
    }
    return T_cal, coeffs


# =============================================================================
# METHOD 3 — POLYNOMIAL REGRESSION (degree 2 or 3)
# =============================================================================

def polynomial_calibration(T_pyr: np.ndarray,
                            T_ref: np.ndarray,
                            cal_fraction: float = 0.20,
                            degree: int = 2):
    """
    Fit a polynomial correction of given degree on the calibration window.

    Useful when emissivity changes significantly with temperature —
    the correction curve needs to bend rather than stay straight.
    Degree 2 is recommended; degree 3 for highly nonlinear sensors.

    Parameters
    ----------
    T_pyr        : np.ndarray — denoised pyrometer signal (°C)
    T_ref        : np.ndarray — thermocouple reference signal (°C)
    cal_fraction : float      — fraction of data for calibration window
    degree       : int        — polynomial degree (2 or 3 recommended)

    Returns
    -------
    T_cal  : np.ndarray — calibrated temperature signal
    coeffs : dict       — {'poly': np.poly1d, 'degree': int, ...}
    """
    cal_end     = max(10, int(cal_fraction * len(T_pyr)))
    x           = T_pyr[:cal_end].astype(np.float64)
    y           = T_ref[:cal_end].astype(np.float64)
    poly_coeffs = np.polyfit(x, y, deg=degree)
    poly_fn     = np.poly1d(poly_coeffs)
    T_cal       = poly_fn(T_pyr.astype(np.float64))
    coeffs      = {
        "poly"    : poly_fn,
        "degree"  : degree,
        "method"  : f"polynomial_deg{degree}",
        "cal_end" : cal_end,
    }
    return T_cal, coeffs


# =============================================================================
# METHOD 4 — PIECEWISE LINEAR REGRESSION  (new for RQ1)
# =============================================================================

def piecewise_calibration(T_pyr: np.ndarray,
                           T_ref: np.ndarray,
                           breakpoints: list = None,
                           cal_fraction: float = 0.20):
    """
    Fit separate linear corrections in different temperature bands.

    The temperature axis is divided into bands by breakpoints.
    A linear fit  T_cal = a_k * T_pyr + b_k  is applied within
    each band k.  At the boundaries, the nearest band's fit is used.

    Why piecewise?
        Emissivity in metal forming can shift abruptly at phase-
        transition temperatures (e.g. around 700°C for steel).
        A single global line cannot capture these step changes.

    Parameters
    ----------
    T_pyr        : np.ndarray — denoised pyrometer signal (°C)
    T_ref        : np.ndarray — thermocouple reference signal (°C)
    breakpoints  : list[float] — temperature break values (°C).
                   Default: [400, 700] → three bands:
                   [min–400], [400–700], [700–max]
    cal_fraction : float — fraction of data used for fitting

    Returns
    -------
    T_cal  : np.ndarray — calibrated temperature signal
    coeffs : dict       — {'segments': list of per-band dicts, ...}
    """
    if breakpoints is None:
        breakpoints = [400.0, 700.0]

    cal_end = max(10, int(cal_fraction * len(T_pyr)))
    x_cal   = T_pyr[:cal_end].astype(np.float64)
    y_cal   = T_ref[:cal_end].astype(np.float64)

    # Build band edges: [−∞, bp0, bp1, ..., +∞]
    edges    = [-np.inf] + list(breakpoints) + [np.inf]
    segments = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask   = (x_cal > lo) & (x_cal <= hi)

        if mask.sum() >= 2:
            # Enough points — fit a line in this band
            A      = np.vstack([x_cal[mask], np.ones(mask.sum())]).T
            result = np.linalg.lstsq(A, y_cal[mask], rcond=None)
            a, b   = result[0]
        else:
            # Too few points in this band — fall back to identity (slope=1, b=0)
            a, b = 1.0, 0.0

        segments.append({"lo": lo, "hi": hi, "a": a, "b": b})

    # Apply: for each sample, find its band and apply that segment's fit
    T_cal = np.empty_like(T_pyr, dtype=np.float64)
    for seg in segments:
        mask          = (T_pyr > seg["lo"]) & (T_pyr <= seg["hi"])
        T_cal[mask]   = seg["a"] * T_pyr[mask] + seg["b"]

    # Handle any samples exactly at or below the lowest breakpoint
    mask_lo         = T_pyr <= edges[1]
    T_cal[mask_lo]  = segments[0]["a"] * T_pyr[mask_lo] + segments[0]["b"]

    coeffs = {
        "segments"    : segments,
        "breakpoints" : breakpoints,
        "method"      : "piecewise",
        "cal_end"     : cal_end,
    }
    return T_cal, coeffs


# =============================================================================
# DRIFT CORRECTION  (applied after any calibration method)
# =============================================================================

def remove_drift(T_pyr: np.ndarray,
                 T_ref: np.ndarray) -> np.ndarray:
    """
    Remove a linear drift from a calibrated pyrometer signal.

    Fits a straight line to the residual (T_pyr - T_ref) over the full
    signal and subtracts it.  Corrects a slow sensor offset that grows
    over time — common in long heat-treatment runs where the sensor
    housing warms up gradually.

    Parameters
    ----------
    T_pyr : np.ndarray — calibrated pyrometer signal
    T_ref : np.ndarray — thermocouple reference signal

    Returns
    -------
    np.ndarray — drift-corrected signal
    """
    residual       = T_pyr.astype(np.float64) - T_ref.astype(np.float64)
    t              = np.arange(len(T_pyr), dtype=np.float64)
    slope, intercept = np.polyfit(t, residual, 1)
    drift          = slope * t + intercept
    return T_pyr - drift


# =============================================================================
# ACCURACY METRICS
# =============================================================================

def rmse(T_cal: np.ndarray, T_true: np.ndarray) -> float:
    """Root Mean Square Error between calibrated and true temperature (°C)."""
    return float(np.sqrt(np.mean((T_cal - T_true) ** 2)))


def mae(T_cal: np.ndarray, T_true: np.ndarray) -> float:
    """Mean Absolute Error between calibrated and true temperature (°C)."""
    return float(np.mean(np.abs(T_cal - T_true)))


def max_error(T_cal: np.ndarray, T_true: np.ndarray) -> float:
    """Maximum absolute error (worst-case single-sample error, °C)."""
    return float(np.max(np.abs(T_cal - T_true)))


# =============================================================================
# REPORTING
# =============================================================================

def calibration_report(T_raw: np.ndarray,
                        T_cal: np.ndarray,
                        T_true: np.ndarray,
                        coeffs: dict,
                        label: str = "Pyrometer") -> None:
    """
    Print a short accuracy report comparing raw vs calibrated signal.

    Parameters
    ----------
    T_raw  : raw (pre-calibration) pyrometer signal
    T_cal  : calibrated signal
    T_true : thermocouple reference (ground truth)
    coeffs : dict returned by any calibration function
    label  : sensor name for display
    """
    method = coeffs.get("method", "?")
    print(f"  [{label}] method     : {method}")
    print(f"  [{label}] cal window : first {coeffs.get('cal_end','?')} samples")

    if method == "linear":
        print(f"  [{label}] formula    : T_cal = {coeffs['a']:.5f} x T_raw"
              f" + {coeffs['b']:.2f}")
    elif method == "mean_offset":
        print(f"  [{label}] offset     : {coeffs['offset']:.2f} °C")
    elif method == "piecewise":
        for i, seg in enumerate(coeffs["segments"]):
            print(f"  [{label}] segment {i}  : "
                  f"({seg['lo']:.0f}–{seg['hi']:.0f}°C) "
                  f"a={seg['a']:.4f}  b={seg['b']:.2f}")

    print(f"  [{label}] RMSE before: {rmse(T_raw,  T_true):.2f} °C")
    print(f"  [{label}] RMSE after : {rmse(T_cal,  T_true):.2f} °C")
    print(f"  [{label}] MAE after  : {mae(T_cal,   T_true):.2f} °C")
    print(f"  [{label}] MAX after  : {max_error(T_cal, T_true):.2f} °C")


# =============================================================================
# PREVIEW HELPER
# =============================================================================

def print_preview(T_raw: np.ndarray,
                  T_cal: np.ndarray,
                  T_ref: np.ndarray,
                  time_s: np.ndarray,
                  start: int = 0) -> None:
    """
    Print a 5-row preview table:
        Time | Raw | Calibrated | TC_reference | Error_vs_ref
    """
    import pandas as pd
    idx = list(range(start, start + 5))
    df  = pd.DataFrame({
        "Time_s" : np.round(time_s[idx], 4),
        "Raw_C"  : np.round(T_raw[idx], 2),
        "Cal_C"  : np.round(T_cal[idx], 2),
        "TC_ref" : np.round(T_ref[idx], 2),
        "Err_C"  : np.round(T_cal[idx] - T_ref[idx], 2),
    }, index=[f"t{i}" for i in idx])
    print(df.to_string())


# =============================================================================
# SELF-TEST  (run: python calibrate.py)
# Simulates the full chain using synthetic data so no .mat file is needed.
# =============================================================================

if __name__ == "__main__":
    import pandas as pd

    print("=" * 65)
    print("calibrate.py — self-test with synthetic data")
    print("Chain: Synthetic Raw → Denoise (skipped) → Calibrate")
    print("=" * 65)

    # --- Build synthetic pyrometer + thermocouple signals ---
    rng   = np.random.default_rng(42)
    n     = 2000
    time_s = np.linspace(0, 4.0, n)

    # True temperature: exponential cooling from 900°C to 100°C
    T_true = 100 + 800 * np.exp(-time_s / 2.5)

    # Pyrometer: emissivity gain error (0.92x) + 15°C offset + noise
    T_pyr  = 0.92 * T_true + 15.0 + rng.normal(0, 3, n)

    # Thermocouple: true temperature + small noise
    T_tc   = T_true + rng.normal(0, 1.5, n)

    hot    = int(T_pyr.argmax())
    print(f"\n  Samples       : {n}")
    print(f"  Pyr range     : {T_pyr.min():.1f} – {T_pyr.max():.1f} °C")
    print(f"  TC  range     : {T_tc.min():.1f}  – {T_tc.max():.1f} °C")

    methods = [
        ("Mean offset",       mean_offset_calibration,  {}),
        ("Linear",            linear_calibration,        {}),
        ("Polynomial deg-2",  polynomial_calibration,    {"degree": 2}),
        ("Piecewise",         piecewise_calibration,     {"breakpoints": [400, 700]}),
    ]

    results = []
    for name, fn, kwargs in methods:
        T_cal, coeffs = fn(T_pyr, T_tc, cal_fraction=0.20, **kwargs)
        T_cal         = remove_drift(T_cal, T_tc)
        r = rmse(T_cal, T_tc)
        m = mae(T_cal, T_tc)
        results.append({"Method": name, "RMSE_C": round(r,2), "MAE_C": round(m,2)})
        print(f"\n{'─'*65}")
        print(f"  METHOD: {name}")
        calibration_report(T_pyr, T_cal, T_tc, coeffs, label=name[:8])

    print(f"\n{'='*65}")
    print("  COMPARISON TABLE")
    print("=" * 65)
    df = pd.DataFrame(results)
    df = df.sort_values("RMSE_C")
    print(df.to_string(index=False))
    print("\n  calibrate.py self-test complete.")
