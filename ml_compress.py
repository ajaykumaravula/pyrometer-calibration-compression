"""
=============================================================================
ml_compress.py  —  Machine-learning compression module  (ATP-3)
=============================================================================
Thesis : Automation of pyrometer data pre-processing
         (denoising, calibration, compression) for metal forming
Student: [Your name]  |  University West 2026

RESEARCH QUESTION (RQ2 / ATP-3):
    What compression ratio and reconstruction accuracy are achievable
    using Variational Autoencoder (VAE) and Deep Autoencoder for
    calibrated pyrometer time-series data?

WHAT THIS MODULE DOES:
    Two ML compression models implemented in PyTorch:

    1. DeepAutoencoder   — deterministic encoder → bottleneck → decoder
    2. VAE               — probabilistic encoder (μ, σ) → sample → decoder

    Both follow the same interface:
        model, info = train_autoencoder(T_cal, latent_dim, ...)
        T_rec       = compress_and_reconstruct(model, T_cal, ...)

DEPENDENCIES:
    pip install torch numpy pandas scikit-learn

PIPELINE POSITION:
    Calibrated_C  →  [ml_compress.py]  →  Reconstructed_C
=============================================================================
"""

import numpy as np
import time

import torch
import torch.nn            as nn
import torch.optim         as optim
from torch.utils.data      import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# SHARED HELPER — segment signal into overlapping windows for training
# =============================================================================

def _make_windows(signal: np.ndarray,
                  window_size: int,
                  stride: int = 1) -> np.ndarray:
    """
    Slide a window over the signal and return all windows as rows.

    Parameters
    ----------
    signal      : 1-D array
    window_size : length of each window
    stride      : step between consecutive windows

    Returns
    -------
    windows : ndarray of shape (n_windows, window_size)
    """
    n       = len(signal)
    indices = range(0, n - window_size + 1, stride)
    return np.array([signal[i: i + window_size] for i in indices],
                    dtype=np.float32)


# =============================================================================
# MODEL 1 — DEEP AUTOENCODER
# =============================================================================

class DeepAutoencoder(nn.Module):
    """
    Symmetric encoder-decoder network for 1-D window compression.

    Architecture (window_size=64, latent_dim=8):
        Encoder: 64 → 32 → 16 → 8  (latent)
        Decoder:  8 → 16 → 32 → 64

    The latent vector of dimension latent_dim IS the compressed
    representation.  Compression ratio ≈ window_size / latent_dim.

    Parameters
    ----------
    window_size : int — length of input/output windows
    latent_dim  : int — size of the bottleneck (compressed code)
    """

    def __init__(self, window_size: int = 64, latent_dim: int = 8):
        super().__init__()
        self.window_size = window_size
        self.latent_dim  = latent_dim

        # Encoder: progressively compress
        self.encoder = nn.Sequential(
            nn.Linear(window_size, window_size // 2), nn.ReLU(),
            nn.Linear(window_size // 2, window_size // 4), nn.ReLU(),
            nn.Linear(window_size // 4, latent_dim),
        )

        # Decoder: progressively expand back to original size
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, window_size // 4), nn.ReLU(),
            nn.Linear(window_size // 4, window_size // 2), nn.ReLU(),
            nn.Linear(window_size // 2, window_size),
        )

    def forward(self, x):
        """Encode then decode — returns reconstruction."""
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        """Return only the latent code (compressed representation)."""
        return self.encoder(x)

    def decode(self, z):
        """Decode a latent code back to a window."""
        return self.decoder(z)


# =============================================================================
# MODEL 2 — VARIATIONAL AUTOENCODER (VAE)
# =============================================================================

class VAE(nn.Module):
    """
    Variational Autoencoder for 1-D window compression.

    Difference from DeepAutoencoder:
        The encoder outputs a mean (mu) and log-variance (log_var)
        instead of a single latent vector.  The actual latent code z
        is sampled from  N(mu, exp(log_var)).

    This probabilistic latent space is smoother and more structured —
    nearby latent codes decode to similar-looking windows.

    Loss = reconstruction_loss + KL_divergence_weight × KL_loss

    Parameters
    ----------
    window_size   : int — length of input/output windows
    latent_dim    : int — size of the latent distribution
    kl_weight     : float — weight on the KL divergence term
    """

    def __init__(self,
                 window_size: int = 64,
                 latent_dim: int  = 8,
                 kl_weight: float = 1e-3):
        super().__init__()
        self.window_size = window_size
        self.latent_dim  = latent_dim
        self.kl_weight   = kl_weight

        # Shared encoder backbone
        self.enc_shared = nn.Sequential(
            nn.Linear(window_size, window_size // 2), nn.ReLU(),
            nn.Linear(window_size // 2, window_size // 4), nn.ReLU(),
        )
        # Separate heads for mean and log-variance
        self.fc_mu      = nn.Linear(window_size // 4, latent_dim)
        self.fc_log_var = nn.Linear(window_size // 4, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, window_size // 4), nn.ReLU(),
            nn.Linear(window_size // 4, window_size // 2), nn.ReLU(),
            nn.Linear(window_size // 2, window_size),
        )

    def encode(self, x):
        """Return mu and log_var for the input batch."""
        h       = self.enc_shared(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterise(self, mu, log_var):
        """
        Reparameterisation trick:
            z = mu + epsilon * std   where epsilon ~ N(0, I)
        Allows gradients to flow through the sampling step.
        """
        std     = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):
        """Decode latent code to reconstructed window."""
        return self.decoder(z)

    def forward(self, x):
        """Full pass: encode → reparameterise → decode."""
        mu, log_var = self.encode(x)
        z           = self.reparameterise(mu, log_var)
        x_hat       = self.decode(z)
        return x_hat, mu, log_var

    def vae_loss(self, x, x_hat, mu, log_var):
        """
        Combined VAE loss:
            recon_loss = MSE between input and reconstruction
            kl_loss    = KL divergence  D_KL( N(mu,σ) || N(0,1) )
        """
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl_loss    = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        return recon_loss + self.kl_weight * kl_loss, recon_loss, kl_loss


# =============================================================================
# TRAINING
# =============================================================================

def train_autoencoder(T_cal: np.ndarray,
                      model_type: str   = "deep",
                      latent_dim: int   = 8,
                      window_size: int  = 64,
                      stride: int       = 4,
                      epochs: int       = 100,
                      batch_size: int   = 64,
                      lr: float         = 1e-3,
                      kl_weight: float  = 1e-3):
    """
    Train a Deep Autoencoder or VAE on a calibrated pyrometer signal.

    The signal is sliced into overlapping windows of `window_size`
    samples.  Each window is normalised to [0, 1] before training.

    Parameters
    ----------
    T_cal       : 1-D calibrated pyrometer signal (°C)
    model_type  : 'deep' or 'vae'
    latent_dim  : bottleneck size (smaller = more compression)
    window_size : length of each training window
    stride      : step between windows (smaller = more training data)
    epochs      : training epochs
    batch_size  : mini-batch size
    lr          : Adam learning rate
    kl_weight   : VAE only — weight on KL divergence loss

    Returns
    -------
    model   : trained PyTorch model
    scaler  : fitted MinMaxScaler (needed for reconstruction)
    info    : dict with training metrics and parameters
    """
    # --- Normalise signal to [0, 1] ---
    scaler = MinMaxScaler()
    T_norm = scaler.fit_transform(
        T_cal.astype(np.float32).reshape(-1, 1)
    ).flatten()

    # --- Slice into windows ---
    windows = _make_windows(T_norm, window_size=window_size, stride=stride)
    X       = torch.tensor(windows, dtype=torch.float32)
    loader  = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    # --- Build model ---
    if model_type == "vae":
        model = VAE(window_size=window_size,
                    latent_dim=latent_dim,
                    kl_weight=kl_weight)
    else:
        model = DeepAutoencoder(window_size=window_size,
                                latent_dim=latent_dim)

    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --- Training loop ---
    t0          = time.perf_counter()
    loss_history= []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            if model_type == "vae":
                x_hat, mu, log_var = model(batch)
                loss, _, _         = model.vae_loss(batch, x_hat, mu, log_var)
            else:
                x_hat = model(batch)
                loss  = criterion(x_hat, batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(batch)

        avg_loss = epoch_loss / len(X)
        loss_history.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:4d}/{epochs}  loss={avg_loss:.6f}")

    fit_time = time.perf_counter() - t0
    model.eval()

    info = {
        "method"       : model_type,
        "latent_dim"   : latent_dim,
        "window_size"  : window_size,
        "epochs"       : epochs,
        "final_loss"   : round(loss_history[-1], 6),
        "fit_time_s"   : round(fit_time, 2),
        "loss_history" : loss_history,
    }
    return model, scaler, info


# =============================================================================
# COMPRESS AND RECONSTRUCT
# =============================================================================

def compress_and_reconstruct(model,
                              scaler,
                              T_cal: np.ndarray,
                              window_size: int = 64,
                              stride: int      = 1) -> np.ndarray:
    """
    Run the full compress → decompress cycle on a signal.

    Uses overlap-add averaging: for each output position, multiple
    overlapping windows may contribute; their reconstructions are
    averaged for a smoother result.

    Parameters
    ----------
    model       : trained DeepAutoencoder or VAE
    scaler      : MinMaxScaler fitted during training
    T_cal       : original calibrated signal (°C)
    window_size : must match training window_size
    stride      : step between windows

    Returns
    -------
    T_reconstructed : np.ndarray — signal after compress + decompress (°C)
    """
    T_norm = scaler.transform(
        T_cal.astype(np.float32).reshape(-1, 1)
    ).flatten()

    n        = len(T_norm)
    out_sum  = np.zeros(n, dtype=np.float64)
    out_cnt  = np.zeros(n, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for start in range(0, n - window_size + 1, stride):
            window  = T_norm[start: start + window_size]
            x_in    = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

            if isinstance(model, VAE):
                x_hat, mu, _ = model(x_in)
                # At inference time, use mu (mean) for deterministic output
                x_hat = model.decode(mu)
            else:
                x_hat = model(x_in)

            recon = x_hat.squeeze(0).numpy()
            out_sum[start: start + window_size] += recon
            out_cnt[start: start + window_size] += 1

    # Avoid division by zero at the very end if stride > 1
    out_cnt = np.maximum(out_cnt, 1)
    T_norm_rec = (out_sum / out_cnt).astype(np.float32)

    # Inverse-normalise back to °C
    T_reconstructed = scaler.inverse_transform(
        T_norm_rec.reshape(-1, 1)
    ).flatten()
    return T_reconstructed.astype(np.float64)


def ml_compression_report(T_cal: np.ndarray,
                           T_rec: np.ndarray,
                           info: dict,
                           label: str = "") -> None:
    """
    Print a compression summary for a trained autoencoder model.

    Parameters
    ----------
    T_cal  : original calibrated signal
    T_rec  : reconstructed signal
    info   : dict returned by train_autoencoder
    label  : optional display label
    """
    tag   = f"[{label}] " if label else ""
    rmse  = float(np.sqrt(np.mean((T_cal - T_rec) ** 2)))
    mae   = float(np.mean(np.abs(T_cal - T_rec)))
    ratio = info["window_size"] / info["latent_dim"]

    print(f"  {tag}method       : {info['method']}")
    print(f"  {tag}latent_dim   : {info['latent_dim']}")
    print(f"  {tag}window_size  : {info['window_size']}")
    print(f"  {tag}logical ratio: {ratio:.1f}×  "
          f"(window_size / latent_dim)")
    print(f"  {tag}epochs       : {info['epochs']}")
    print(f"  {tag}final loss   : {info['final_loss']}")
    print(f"  {tag}fit time     : {info['fit_time_s']} s")
    print(f"  {tag}RMSE         : {rmse:.3f} °C")
    print(f"  {tag}MAE          : {mae:.3f} °C")


# =============================================================================
# SELF-TEST  (run: python ml_compress.py)
# =============================================================================

if __name__ == "__main__":
    import pandas as pd

    print("=" * 65)
    print("ml_compress.py — self-test with synthetic cooling data")
    print("=" * 65)

    rng    = np.random.default_rng(42)
    n      = 2000
    time_s = np.linspace(0, 4.0, n)
    T_cal  = 100 + 800 * np.exp(-time_s / 2.5) + rng.normal(0, 1.0, n)

    print(f"  Signal: {n} samples, {T_cal.min():.1f}–{T_cal.max():.1f} °C\n")

    results = []

    for model_type, latent in [("deep", 8), ("vae", 8),
                                ("deep", 4), ("vae", 4)]:
        label = f"{model_type.upper()} latent={latent}"
        print(f"\n{'─'*65}")
        print(f"  Training: {label}")
        model, scaler, info = train_autoencoder(
            T_cal,
            model_type  = model_type,
            latent_dim  = latent,
            window_size = 64,
            stride      = 4,
            epochs      = 60,
            batch_size  = 64,
        )
        T_rec = compress_and_reconstruct(model, scaler, T_cal,
                                         window_size=64, stride=1)
        ml_compression_report(T_cal, T_rec, info, label=label[:10])

        rmse = float(np.sqrt(np.mean((T_cal - T_rec) ** 2)))
        results.append({
            "Model"     : label,
            "LatentDim" : latent,
            "Ratio"     : f"{64/latent:.0f}×",
            "RMSE_C"    : round(rmse, 3),
            "FitTime_s" : info["fit_time_s"],
        })

    print(f"\n{'='*65}")
    print("  COMPARISON TABLE")
    print("=" * 65)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("\n  ml_compress.py self-test complete.")
