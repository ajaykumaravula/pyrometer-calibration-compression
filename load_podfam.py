"""
=============================================================================
load_podfam.py  —  PODFAM Primary Dataset Loader
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

ROLE IN THESIS:
    PRIMARY dataset — substitute for unavailable AP&T proprietary data.
    Provided by the university. Same experimental setup as the thesis
    description: 2 pyrometers + thermocouple reference.

DATASET:
    10 .pcd files: 001_820.pcd to 010_990.pcd
    sensor0  = Pyrometer 1 (°C)
    sensor1  = Pyrometer 2 (°C)
    filename = NNN_TTT.pcd  →  TTT = TC reference temperature (°C)
    Sample rate: ~100,000 Hz  |  Duration: ~1.5 s per file

HOW TO USE:
    from load_podfam import load_pcd_file, load_all_files, PODFAM_FILES
    T_pyr1, T_pyr2, T_tc, time_s, meta = load_pcd_file("001_820.pcd")
    layers = load_all_files("path/to/pcd/folder/")
=============================================================================
"""

import os
import numpy as np

# All 10 PODFAM files with their TC reference temperatures
PODFAM_FILES = [
    ("001_820.pcd", 820), ("002_800.pcd", 800), ("003_850.pcd", 850),
    ("004_830.pcd", 830), ("005_880.pcd", 880), ("006_860.pcd", 860),
    ("007_840.pcd", 840), ("008_820.pcd", 820), ("009_800.pcd", 800),
    ("010_990.pcd", 990),
]


def load_pcd_file(filepath: str, max_rows: int = 200000) -> tuple:
    """
    Load one PODFAM .pcd file → pipeline-ready arrays.

    TC reference is constant per file (isothermal measurement).
    Extracted from filename: 001_820.pcd → TC = 820°C.

    Parameters
    ----------
    filepath : path to .pcd file
    max_rows : rows to read (200k is enough for pipeline dev;
               full file ~2.8M rows)

    Returns
    -------
    T_pyr1 : np.ndarray  Pyrometer 1 signal (°C)
    T_pyr2 : np.ndarray  Pyrometer 2 signal (°C)
    T_tc   : np.ndarray  Thermocouple reference (constant, °C)
    time_s : np.ndarray  Time axis (seconds)
    meta   : dict        File info, sample rate, TC reference
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            f"Expected a PODFAM .pcd file (sensor0=Pyr1, sensor1=Pyr2)."
        )

    # Parse TC reference from filename: NNN_TTT.pcd or NNN.TTT.pcd → TTT
    basename = os.path.basename(filepath)
    try:
        name_no_ext = basename[:-4] if basename.lower().endswith('.pcd') else basename.replace('.pcd', '')
        if '_' in name_no_ext:
            ref_temp = float(name_no_ext.split('_')[1])
        elif '.' in name_no_ext:
            ref_temp = float(name_no_ext.split('.')[1])
        else:
            ref_temp = float(name_no_ext)
    except Exception:
        raise ValueError(
            f"Cannot parse TC ref from '{basename}'. "
            f"Expected format: NNN_TTT.pcd (e.g. 001_820.pcd) or NNN.TTT.pcd"
        )

    # Read PCD header + data
    rows = []
    total_pts = 0
    header_done = False

    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('POINTS'):
                total_pts = int(line.split()[1])
            if line == 'DATA ascii':
                header_done = True
                continue
            if header_done and line:
                try:
                    v = list(map(int, line.split()))
                    if len(v) == 10:
                        rows.append(v)
                    if len(rows) >= max_rows:
                        break
                except Exception:
                    continue

    data = np.array(rows, dtype=np.float64)
    # PCD columns: t, x, y, z, sensor0, sensor1, sensor2, sensor3, state0, state1
    time_s = data[:, 0] / 1e6          # microseconds → seconds
    T_pyr1 = data[:, 4]                # Pyrometer 1 (°C)
    T_pyr2 = data[:, 5]                # Pyrometer 2 (°C)
    T_tc   = np.full(len(T_pyr1), ref_temp)  # TC = constant reference

    dt = np.median(np.diff(time_s))
    meta = {
        'file'           : basename,
        'ref_temp_C'     : ref_temp,
        'total_points'   : total_pts,
        'loaded_points'  : len(data),
        'duration_s'     : round(float(time_s[-1] - time_s[0]), 4),
        'sample_rate_Hz' : round(float(1.0 / dt) if dt > 0 else 0, 0),
        'pyr1_mean_C'    : round(float(T_pyr1.mean()), 2),
        'pyr2_mean_C'    : round(float(T_pyr2.mean()), 2),
        'pyr1_std_C'     : round(float(T_pyr1.std()),  2),
        'pyr2_std_C'     : round(float(T_pyr2.std()),  2),
    }
    return T_pyr1, T_pyr2, T_tc, time_s, meta


def load_all_files(data_dir: str = ".",
                   max_rows: int = 200000,
                   verbose: bool = True) -> list:
    """
    Load all 10 PODFAM files from a directory.

    Returns list of (T_pyr1, T_pyr2, T_tc, time_s, meta) tuples.
    """
    results = []
    for fname, _ in PODFAM_FILES:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            # Try with dot instead of underscore
            dot_fname = fname.replace('_', '.')
            path = os.path.join(data_dir, dot_fname)
        if not os.path.exists(path):
            if verbose:
                print(f"  [skip] {fname} not found in {data_dir}")
            continue
        T1, T2, TC, t, meta = load_pcd_file(path, max_rows=max_rows)
        results.append((T1, T2, TC, t, meta))
        if verbose:
            print(f"  {fname}: {meta['loaded_points']:,} pts  |  "
                  f"Pyr1={meta['pyr1_mean_C']:.1f}°C  "
                  f"Pyr2={meta['pyr2_mean_C']:.1f}°C  |  "
                  f"TC ref={meta['ref_temp_C']:.0f}°C")
    return results


def print_summary(data_dir: str = ".") -> None:
    """Print a summary table of all available PODFAM files."""
    import pandas as pd
    rows = []
    for fname, ref in PODFAM_FILES:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            dot_fname = fname.replace('_', '.')
            path = os.path.join(data_dir, dot_fname)
        if not os.path.exists(path):
            continue
        _, _, _, _, meta = load_pcd_file(path, max_rows=5000)
        rows.append({
            'File'      : fname,
            'TC_ref_C'  : meta['ref_temp_C'],
            'TotalPts'  : meta['total_points'],
            'Duration_s': meta['duration_s'],
            'Rate_Hz'   : meta['sample_rate_Hz'],
            'Pyr1_C'    : meta['pyr1_mean_C'],
            'Pyr2_C'    : meta['pyr2_mean_C'],
        })
    print("=" * 70)
    print("PODFAM Dataset — 2-Pyrometer + Thermocouple Experiment")
    print("  sensor0=Pyr1  |  sensor1=Pyr2  |  TC ref from filename")
    print("=" * 70)
    print(pd.DataFrame(rows).to_string(index=False))


# Self-test
if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "/home/ajay/Downloads/"
    print_summary(d)
