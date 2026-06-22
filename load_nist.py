"""
=============================================================================
load_nist.py  —  NIST AMBench Open Testbed Loader
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

ROLE IN THESIS:
    OPEN DATASET TESTBED — as suggested in thesis description:
    "Open high-temperature datasets (e.g. NIST AMMT) can be used as
    an additional testbed to stress-test the pipeline."
    NOT a substitute for the primary dataset (PODFAM).

DATASET:
    NIST AMBench 2018 — IN625 Laser Powder Bed Fusion
    Source : https://doi.org/10.18434/M32044
    Files  : Layer01.mat – Layer10.mat
    Camera : 126×360 px thermal camera, ~1800 Hz
    Note   : NO real thermocouple — TC reference is simulated.
             Stated as limitation in thesis.

HOW TO USE:
    from load_nist import load_mat_layer, load_all_layers
    T_pyr, T_tc, time_s, meta = load_mat_layer("data/Layer01.mat")
=============================================================================
"""

import os
import numpy as np
from scipy.io      import loadmat
from scipy.ndimage import gaussian_filter1d


def load_mat_layer(mat_path: str,
                   active_thresh: float = 100.0,
                   tc_sigma: float      = 15.0,
                   tc_noise: float      = 8.0,
                   seed: int            = 42) -> tuple:
    """
    Load one NIST AMBench .mat file → pipeline-ready arrays.

    Extraction:
        1. Load RadiantTemp (126×360×N frames)
        2. Sakuma-Hattori: raw ADC → °C  (T = A×raw + B − 273.15)
        3. Per-frame MAX → 1D pyrometer-like signal
        4. Filter to active frames (T > active_thresh)
        5. Simulate TC reference (smoothed signal + noise)

    NOTE: TC is SIMULATED — no real thermocouple in this dataset.

    Parameters
    ----------
    mat_path      : path to Layer0N.mat file
    active_thresh : min temperature for active frames (°C)
    tc_sigma      : Gaussian smoothing for TC simulation
    tc_noise      : noise std for TC simulation (°C)
    seed          : random seed for reproducibility

    Returns
    -------
    T_pyr  : 1D pyrometer signal (active frames, °C)
    T_tc   : simulated TC reference (°C)
    time_s : time axis (seconds)
    meta   : dict with process parameters
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"File not found: {mat_path}\n"
            f"Download from: https://doi.org/10.18434/M32044"
        )

    mat = loadmat(mat_path)
    L   = mat['Layer'][0, 0]

    sh_A   = float(L['SHvariable_A'].flat[0])   # 2.655
    sh_B   = float(L['SHvariable_B'].flat[0])   # -800.7
    rt     = L['RadiantTemp'].astype(np.float32) # (126, 360, N)
    t_all  = L['BuildTime'][:, 2].flatten().astype(np.float64)
    N      = rt.shape[2]

    # Sakuma-Hattori conversion: raw ADC → °C
    T_all  = sh_A * rt.max(axis=(0, 1)).astype(np.float64) + sh_B - 273.15

    # Filter to active frames
    active = T_all > active_thresh
    T_pyr  = T_all[active]
    time_s = t_all[active]

    # Simulate TC reference (NIST has no real TC)
    rng  = np.random.default_rng(seed)
    T_tc = (gaussian_filter1d(T_pyr, sigma=tc_sigma)
            + rng.normal(0, tc_noise, len(T_pyr)))

    meta = {
        'file'          : os.path.basename(mat_path),
        'material'      : str(L['Material'].flat[0]),
        'laser_W'       : float(L['LaserPower'].flat[0]),
        'speed_mmps'    : float(L['ScanSpeed'].flat[0]),
        'n_frames_total': N,
        'n_frames_active': int(active.sum()),
        'duration_s'    : round(float(t_all[-1]), 3),
        'T_max_C'       : round(float(T_pyr.max()), 1),
        'sh_A'          : sh_A,
        'sh_B'          : sh_B,
        'tc_simulated'  : True,   # important: no real TC
    }
    return T_pyr, T_tc, time_s, meta


def load_all_layers(data_dir: str = "data/",
                    verbose: bool = True) -> list:
    """
    Load all available NIST layers from data_dir.
    Returns list of (T_pyr, T_tc, time_s, meta) tuples.
    """
    results = []
    for n in range(1, 11):
        path = os.path.join(data_dir, f"Layer{n:02d}.mat")
        if not os.path.exists(path):
            if verbose:
                print(f"  [skip] Layer{n:02d}.mat not found")
            continue
        T_pyr, T_tc, time_s, meta = load_mat_layer(path)
        results.append((T_pyr, T_tc, time_s, meta))
        if verbose:
            print(f"  Layer{n:02d}: {meta['n_frames_active']} active frames  |  "
                  f"T_max={meta['T_max_C']:.0f}°C  |  "
                  f"TC simulated={meta['tc_simulated']}")
    return results


# Self-test
if __name__ == "__main__":
    import sys, pandas as pd
    d = sys.argv[1] if len(sys.argv) > 1 else "data/"
    print("=" * 60)
    print("NIST AMBench Dataset — Open Testbed")
    print("NOTE: TC reference is SIMULATED in this dataset")
    print("=" * 60)
    rows = []
    for n in range(1, 11):
        p = os.path.join(d, f"Layer{n:02d}.mat")
        if not os.path.exists(p): continue
        T_pyr, T_tc, time_s, meta = load_mat_layer(p)
        rows.append({'Layer': f"Layer{n:02d}",
                     'ActiveFrames': meta['n_frames_active'],
                     'Duration_s': meta['duration_s'],
                     'T_max_C': meta['T_max_C']})
    if rows: print(pd.DataFrame(rows).to_string(index=False))
