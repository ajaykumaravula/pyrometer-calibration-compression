"""
=============================================================================
ml_calibrate.py  —  Machine-learning calibration module  (ATP-2)
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

RESEARCH QUESTION (RQ1 / ATP-2):
    How effectively can ML calibration methods (Random Forest, MLP,
    Gradient Boosting, SVR) correct systematic pyrometer temperature errors,
    and which method provides the best trade-off between calibration accuracy
    and computational complexity?

WHAT THIS MODULE DOES:
    Four ML regression methods, each with the SAME interface as calibrate.py:
        T_cal, info = <method>(T_pyr, T_tc, cal_fraction=0.20)

    1. rf_calibration   — Random Forest regressor
    2. mlp_calibration  — Multi-Layer Perceptron (scikit-learn)
    3. gb_calibration   — Gradient Boosting regressor
    4. svr_calibration  — Support Vector Regression

HOW TO USE:
    from ml_calibrate import rf_calibration, mlp_calibration
    T_cal, info = rf_calibration(T_pyr, T_tc, cal_fraction=0.20)

DEPENDENCIES:
    pip install scikit-learn numpy pandas

PIPELINE POSITION:
    Denoised_C  →  [ml_calibrate.py]  →  Calibrated_C
    (same position as calibrate.py — swap-in replacement)
=============================================================================
"""

import numpy as np
import time

from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.svm              import SVR
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import mean_squared_error, mean_absolute_error


# =============================================================================
# SHARED HELPER — build (X, y) training arrays from 1-D signals
# =============================================================================

def _build_features(T_pyr: np.ndarray) -> np.ndarray:
    """
    Build a feature matrix from a 1-D pyrometer signal.

    For 1-D temperature regression the input feature is simply
    the pyrometer reading itself.  We also add T² so the model
    can capture a quadratic emissivity-vs-temperature relationship
    without needing extra hidden layers.

    Feature columns: [T_pyr,  T_pyr²]

    Parameters
    ----------
    T_pyr : 1-D array of pyrometer readings (°C)

    Returns
    -------
    X : ndarray of shape (n, 2)
    """
    T = T_pyr.astype(np.float64).reshape(-1, 1)
    return np.hstack([T, T ** 2])


# =============================================================================
# METHOD 1 — RANDOM FOREST
# =============================================================================

def rf_calibration(T_pyr: np.ndarray,
                   T_ref: np.ndarray,
                   cal_fraction: float = 0.20,
                   n_estimators: int   = 100,
                   max_depth: int      = None):
    """
    Calibrate using a Random Forest regressor.

    Builds an ensemble of decision trees on (T_pyr, T_pyr²) → T_tc
    pairs from the calibration window, then predicts over the full signal.

    Why Random Forest?
        - Naturally handles nonlinear emissivity–temperature relationships
        - Robust to outliers and noisy training samples
        - No need for feature scaling
        - Feature importance tells us how much T vs T² matters

    Parameters
    ----------
    T_pyr         : denoised pyrometer signal (°C)
    T_ref         : thermocouple reference (°C)
    cal_fraction  : fraction of signal used for training
    n_estimators  : number of trees in the forest
    max_depth     : maximum tree depth (None = grow fully)

    Returns
    -------
    T_cal : calibrated signal
    info  : dict with model, training metrics, fit time
    """
    cal_end = max(20, int(cal_fraction * len(T_pyr)))
    X_train = _build_features(T_pyr[:cal_end])
    y_train = T_ref[:cal_end].astype(np.float64)
    X_all   = _build_features(T_pyr)

    t0    = time.perf_counter()
    model = RandomForestRegressor(
        n_estimators = n_estimators,
        max_depth    = max_depth,
        random_state = 42,
        n_jobs       = -1,
    )
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    T_cal = model.predict(X_all)

    # Training-set residuals for reporting
    y_pred_train = model.predict(X_train)
    train_rmse   = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

    info = {
        "method"          : "random_forest",
        "model"           : model,
        "cal_end"         : cal_end,
        "n_estimators"    : n_estimators,
        "train_rmse_C"    : round(train_rmse, 3),
        "fit_time_s"      : round(fit_time, 4),
        "feature_importance": model.feature_importances_.tolist(),
    }
    return T_cal, info


# =============================================================================
# METHOD 2 — MLP (Multi-Layer Perceptron)
# =============================================================================

def mlp_calibration(T_pyr: np.ndarray,
                    T_ref: np.ndarray,
                    cal_fraction: float  = 0.20,
                    hidden_layers: tuple = (64, 32),
                    max_iter: int        = 1000):
    """
    Calibrate using a Multi-Layer Perceptron neural network.

    Learns the nonlinear mapping  T_pyr → T_tc  using a small
    fully-connected network.  Input features are scaled to zero mean
    and unit variance before training (required for MLP convergence).

    Architecture:
        Input(2) → Dense(64, relu) → Dense(32, relu) → Output(1)

    Parameters
    ----------
    T_pyr         : denoised pyrometer signal (°C)
    T_ref         : thermocouple reference (°C)
    cal_fraction  : fraction of signal used for training
    hidden_layers : tuple of hidden layer sizes
    max_iter      : maximum training iterations (epochs)

    Returns
    -------
    T_cal : calibrated signal
    info  : dict with model, scaler, training metrics, fit time
    """
    cal_end = max(20, int(cal_fraction * len(T_pyr)))
    X_train = _build_features(T_pyr[:cal_end])
    y_train = T_ref[:cal_end].astype(np.float64)
    X_all   = _build_features(T_pyr)

    # Scale features — MLP is sensitive to input magnitude
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_all_sc   = scaler.transform(X_all)

    t0    = time.perf_counter()
    model = MLPRegressor(
        hidden_layer_sizes = hidden_layers,
        activation         = "relu",
        solver             = "adam",
        max_iter           = max_iter,
        random_state       = 42,
        early_stopping     = True,
        validation_fraction= 0.10,
        n_iter_no_change   = 20,
    )
    model.fit(X_train_sc, y_train)
    fit_time = time.perf_counter() - t0

    T_cal = model.predict(X_all_sc)

    y_pred_train = model.predict(X_train_sc)
    train_rmse   = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

    info = {
        "method"        : "mlp",
        "model"         : model,
        "scaler"        : scaler,
        "cal_end"       : cal_end,
        "hidden_layers" : hidden_layers,
        "n_iter"        : model.n_iter_,
        "train_rmse_C"  : round(train_rmse, 3),
        "fit_time_s"    : round(fit_time, 4),
    }
    return T_cal, info


# =============================================================================
# METHOD 3 — GRADIENT BOOSTING
# =============================================================================

def gb_calibration(T_pyr: np.ndarray,
                   T_ref: np.ndarray,
                   cal_fraction: float = 0.20,
                   n_estimators: int   = 200,
                   learning_rate: float= 0.05,
                   max_depth: int      = 4):
    """
    Calibrate using Gradient Boosting regression.

    Builds trees sequentially — each tree corrects the residual error
    of the previous ensemble.  Often the most accurate tree-based method
    on smooth calibration data, at the cost of longer training time
    compared to Random Forest.

    Parameters
    ----------
    T_pyr          : denoised pyrometer signal (°C)
    T_ref          : thermocouple reference (°C)
    cal_fraction   : fraction of signal used for training
    n_estimators   : number of boosting rounds
    learning_rate  : shrinkage applied to each tree's contribution
    max_depth      : depth of individual trees

    Returns
    -------
    T_cal : calibrated signal
    info  : dict with model, training metrics, fit time
    """
    cal_end = max(20, int(cal_fraction * len(T_pyr)))
    X_train = _build_features(T_pyr[:cal_end])
    y_train = T_ref[:cal_end].astype(np.float64)
    X_all   = _build_features(T_pyr)

    t0    = time.perf_counter()
    model = GradientBoostingRegressor(
        n_estimators  = n_estimators,
        learning_rate = learning_rate,
        max_depth     = max_depth,
        random_state  = 42,
    )
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    T_cal = model.predict(X_all)

    y_pred_train = model.predict(X_train)
    train_rmse   = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

    info = {
        "method"        : "gradient_boosting",
        "model"         : model,
        "cal_end"       : cal_end,
        "n_estimators"  : n_estimators,
        "learning_rate" : learning_rate,
        "train_rmse_C"  : round(train_rmse, 3),
        "fit_time_s"    : round(fit_time, 4),
    }
    return T_cal, info


# =============================================================================
# METHOD 4 — SUPPORT VECTOR REGRESSION (SVR)
# =============================================================================

def svr_calibration(T_pyr: np.ndarray,
                    T_ref: np.ndarray,
                    cal_fraction: float = 0.20,
                    kernel: str         = "rbf",
                    C: float            = 100.0,
                    epsilon: float      = 0.5):
    """
    Calibrate using Support Vector Regression (SVR).

    Fits a regression in a kernel-transformed feature space with a
    margin of tolerance epsilon.  Samples within epsilon of the true
    value are not penalised — this makes the model robust to small
    sensor noise.

    The RBF kernel is recommended for smooth temperature signals.
    Features are scaled before fitting (required for SVR).

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference (°C)
    cal_fraction : fraction of signal used for training
    kernel       : SVR kernel — 'rbf', 'linear', or 'poly'
    C            : regularisation parameter (larger = tighter fit)
    epsilon      : insensitive tube width (°C)

    Returns
    -------
    T_cal : calibrated signal
    info  : dict with model, scaler, training metrics, fit time
    """
    cal_end = max(20, int(cal_fraction * len(T_pyr)))
    X_train = _build_features(T_pyr[:cal_end])
    y_train = T_ref[:cal_end].astype(np.float64)
    X_all   = _build_features(T_pyr)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_all_sc   = scaler.transform(X_all)

    t0    = time.perf_counter()
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train_sc, y_train)
    fit_time = time.perf_counter() - t0

    T_cal = model.predict(X_all_sc)

    y_pred_train = model.predict(X_train_sc)
    train_rmse   = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

    info = {
        "method"       : f"svr_{kernel}",
        "model"        : model,
        "scaler"       : scaler,
        "cal_end"      : cal_end,
        "kernel"       : kernel,
        "C"            : C,
        "epsilon"      : epsilon,
        "train_rmse_C" : round(train_rmse, 3),
        "fit_time_s"   : round(fit_time, 4),
    }
    return T_cal, info


# =============================================================================
# REPORTING
# =============================================================================

def ml_calibration_report(T_raw: np.ndarray,
                           T_cal: np.ndarray,
                           T_true: np.ndarray,
                           info: dict,
                           label: str = "") -> None:
    """
    Print a short accuracy + complexity report for an ML calibration result.

    Parameters
    ----------
    T_raw  : raw pyrometer signal (before calibration)
    T_cal  : calibrated signal
    T_true : thermocouple reference
    info   : dict returned by any ml_calibrate function
    label  : optional display label
    """
    tag    = f"[{label}] " if label else ""
    method = info.get("method", "?")
    rmse_before = float(np.sqrt(np.mean((T_raw  - T_true) ** 2)))
    rmse_after  = float(np.sqrt(np.mean((T_cal  - T_true) ** 2)))
    mae_after   = float(np.mean(np.abs(T_cal - T_true)))

    print(f"  {tag}method      : {method}")
    print(f"  {tag}cal window  : first {info.get('cal_end','?')} samples")
    print(f"  {tag}fit time    : {info.get('fit_time_s','?')} s")
    print(f"  {tag}RMSE before : {rmse_before:.2f} °C")
    print(f"  {tag}RMSE after  : {rmse_after:.2f} °C")
    print(f"  {tag}MAE after   : {mae_after:.2f} °C")


# =============================================================================
# SELF-TEST  (run: python ml_calibrate.py)
# =============================================================================

if __name__ == "__main__":
    import pandas as pd

    print("=" * 65)
    print("ml_calibrate.py — self-test with synthetic cooling data")
    print("=" * 65)

    rng    = np.random.default_rng(42)
    n      = 2000
    time_s = np.linspace(0, 4.0, n)
    T_true = 100 + 800 * np.exp(-time_s / 2.5)
    T_pyr  = 0.92 * T_true + 15.0 + rng.normal(0, 3, n)
    T_tc   = T_true + rng.normal(0, 1.5, n)

    methods = [
        ("Random Forest",     rf_calibration,  {}),
        ("MLP",               mlp_calibration, {}),
        ("Gradient Boosting", gb_calibration,  {}),
        ("SVR (rbf)",         svr_calibration, {}),
    ]

    results = []
    for name, fn, kwargs in methods:
        print(f"\n{'─'*65}  {name}")
        T_cal, info = fn(T_pyr, T_tc, cal_fraction=0.20, **kwargs)
        ml_calibration_report(T_pyr, T_cal, T_tc, info, label=name[:6])
        rmse_val = float(np.sqrt(np.mean((T_cal - T_tc) ** 2)))
        mae_val  = float(np.mean(np.abs(T_cal - T_tc)))
        results.append({
            "Method"    : name,
            "RMSE_C"    : round(rmse_val, 2),
            "MAE_C"     : round(mae_val,  2),
            "FitTime_s" : info["fit_time_s"],
        })

    print(f"\n{'='*65}")
    print("  COMPARISON TABLE  (sorted by RMSE)")
    print("=" * 65)
    df = pd.DataFrame(results).sort_values("RMSE_C")
    print(df.to_string(index=False))
    print("\n  ml_calibrate.py self-test complete.")
