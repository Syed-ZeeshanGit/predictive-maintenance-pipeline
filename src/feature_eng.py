"""
src/feature_eng.py
==================
Mathematical Feature Engineering Engine
-----------------------------------------
Implements two core transforms from first principles:

  1. Discrete Fourier Transform (DFT) via numpy.fft
     Extracts Spectral Peak Frequency and Spectral Entropy from the
     one-sided power spectral density of a vibration window.

  2. Principal Component Analysis (PCA) via Singular Value Decomposition
     Standardises the multi-sensor feature matrix, decomposes it via SVD,
     and projects onto the first two principal components to form a
     single scalar Health Index Coefficient per window.

Mathematical Reference
----------------------
FFT Power Spectral Density (one-sided):
    S(fₖ) = |X(fₖ)|² / N²      for k = 0, 1, …, N/2

Spectral Entropy (Shannon):
    Hₛ = − Σₖ p̃ₖ · log(p̃ₖ)    where  p̃ₖ = S(fₖ) / Σⱼ S(fⱼ)

SVD decomposition:
    X_centred = U · Σ · Vᵀ

PCA projection:
    X_projected = X_centred · V[:, :2]

Health Index (scalar, per window):
    HI = ‖ X_projected ‖₂  (Euclidean norm of the 2-D projected vector)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from src.ingestion import SensorWindow


# ---------------------------------------------------------------------------
# 1. DISCRETE FOURIER TRANSFORM ANALYSIS
# ---------------------------------------------------------------------------

def compute_fft_features(
    vibration_window: np.ndarray,
    sampling_rate_hz: float,
) -> dict:
    """
    Apply the Discrete Fourier Transform to one vibration window and extract
    frequency-domain condition indicators.

    Algorithm
    ---------
    1. Apply a Hann window to suppress spectral leakage:
           w(n) = 0.5 · (1 − cos(2π·n / (N−1)))
    2. Compute the one-sided complex DFT:
           X(fₖ) = Σ_{n=0}^{N-1} x(n)·w(n) · e^{−j·2π·k·n/N}
    3. Normalised one-sided Power Spectral Density:
           S(fₖ) = |X(fₖ)|² / N²
    4. Spectral Peak Frequency: fₚ = argmax_k S(fₖ)
    5. Spectral Entropy:        Hₛ = −Σₖ p̃ₖ · ln(p̃ₖ)

    Parameters
    ----------
    vibration_window : 1-D array of vibration samples (g-force units)
    sampling_rate_hz : acquisition rate in Hz

    Returns
    -------
    dict with keys:
        'frequencies_hz'        : frequency bin axis (Hz), 1-D array
        'power_spectral_density': S(fₖ), 1-D array
        'spectral_peak_freq_hz' : dominant frequency in Hz (float)
        'spectral_entropy'      : Shannon entropy of the PSD (float)
        'spectral_peak_power'   : power magnitude at peak frequency (float)
    """
    num_samples = len(vibration_window)

    # Step 1: Hann window to reduce spectral leakage
    hann_window   = np.hanning(num_samples)
    windowed_signal = vibration_window * hann_window

    # Step 2: Full complex FFT via numpy (Cooley-Tukey radix-2)
    complex_spectrum = np.fft.fft(windowed_signal)

    # Step 3: One-sided PSD (only positive frequencies)
    num_positive_bins     = num_samples // 2 + 1
    one_sided_complex     = complex_spectrum[:num_positive_bins]
    power_spectral_density = (np.abs(one_sided_complex) ** 2) / (num_samples ** 2)

    # Double the power for all bins except DC (k=0) and Nyquist (k=N/2)
    # to conserve total energy in the one-sided representation
    power_spectral_density[1:-1] *= 2.0

    # Frequency axis
    frequencies_hz = np.fft.fftfreq(num_samples, d=1.0 / sampling_rate_hz)[
        :num_positive_bins
    ]

    # Step 4: Spectral Peak Frequency — index of maximum PSD bin
    peak_bin_index         = int(np.argmax(power_spectral_density))
    spectral_peak_freq_hz  = float(frequencies_hz[peak_bin_index])
    spectral_peak_power    = float(power_spectral_density[peak_bin_index])

    # Step 5: Spectral Entropy  Hₛ = −Σ p̃ₖ · ln(p̃ₖ)
    psd_sum = np.sum(power_spectral_density)
    if psd_sum > 1e-15:
        normalised_psd  = power_spectral_density / psd_sum          # p̃ₖ
        # Guard against log(0): mask zero-power bins
        nonzero_mask    = normalised_psd > 0.0
        spectral_entropy = float(
            -np.sum(normalised_psd[nonzero_mask] * np.log(normalised_psd[nonzero_mask]))
        )
    else:
        spectral_entropy = 0.0

    return {
        "frequencies_hz":         frequencies_hz,
        "power_spectral_density":  power_spectral_density,
        "spectral_peak_freq_hz":   spectral_peak_freq_hz,
        "spectral_entropy":        spectral_entropy,
        "spectral_peak_power":     spectral_peak_power,
    }


def batch_fft_features(
    windows: List[SensorWindow],
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """
    Run compute_fft_features() over a list of SensorWindows and return
    a tidy DataFrame of scalar FFT features (one row per window).

    Parameters
    ----------
    windows          : list of SensorWindow objects
    sampling_rate_hz : sensor acquisition rate in Hz

    Returns
    -------
    pd.DataFrame with columns:
        window_index, spectral_peak_freq_hz, spectral_entropy,
        spectral_peak_power
    """
    records = []
    for window in windows:
        fft_result = compute_fft_features(window.vibration_raw, sampling_rate_hz)
        records.append(
            {
                "window_index":          window.window_index,
                "spectral_peak_freq_hz": fft_result["spectral_peak_freq_hz"],
                "spectral_entropy":      fft_result["spectral_entropy"],
                "spectral_peak_power":   fft_result["spectral_peak_power"],
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. PCA VIA SINGULAR VALUE DECOMPOSITION (from scratch — no sklearn)
# ---------------------------------------------------------------------------

def standardise_matrix(feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zero-centre and unit-scale each column of the feature matrix X.

    Transformation:
        X_std[:, j] = (X[:, j] − μⱼ) / σⱼ

    Parameters
    ----------
    feature_matrix : shape (N, M) — N windows × M features

    Returns
    -------
    standardised_matrix : shape (N, M)
    column_means        : shape (M,) — μⱼ for each feature
    column_stds         : shape (M,) — σⱼ for each feature
    """
    column_means = np.mean(feature_matrix, axis=0)
    column_stds  = np.std( feature_matrix, axis=0, ddof=1)

    # Replace zero-variance columns with ones to avoid division by zero
    column_stds_safe = np.where(column_stds < 1e-12, 1.0, column_stds)

    standardised_matrix = (feature_matrix - column_means) / column_stds_safe

    return standardised_matrix, column_means, column_stds


def pca_via_svd(
    feature_matrix: np.ndarray,
    n_components: int = 2,
) -> dict:
    """
    Perform Principal Component Analysis by computing the full SVD of the
    standardised feature matrix.  No sklearn — only numpy linear algebra.

    Algorithm
    ---------
    Given X ∈ ℝ^{N×M} (N windows, M features):

    Step 1  Standardise:  X̃ = standardise_matrix(X)

    Step 2  SVD:          X̃ = U · Σ · Vᵀ
            U  ∈ ℝ^{N×N} — left singular vectors  (sample modes)
            Σ  ∈ ℝ^{N×M} — singular values on diagonal
            Vᵀ ∈ ℝ^{M×M} — right singular vectors (feature modes / PCs)

    Step 3  Explained variance:
            λₖ = σₖ² / (N − 1)       (eigenvalues of covariance matrix)
            explained_ratio_k = λₖ / Σⱼ λⱼ

    Step 4  Projection onto first `n_components` PCs:
            X_projected = X̃ · V[:, :n_components]

    Parameters
    ----------
    feature_matrix : shape (N, M) — raw (unstandardised) feature values
    n_components   : number of principal components to retain

    Returns
    -------
    dict with keys:
        'standardised_matrix'   : X̃, shape (N, M)
        'singular_values'       : σ, shape (min(N,M),)
        'right_singular_vectors': V, shape (M, M)  — columns are PCs
        'eigenvalues'           : λ, shape (min(N,M),)
        'explained_variance_ratio': array of per-PC explained variance fractions
        'projected_matrix'      : X_projected, shape (N, n_components)
        'column_means'          : μ used for standardisation
        'column_stds'           : σ used for standardisation
    """
    num_windows, num_features = feature_matrix.shape

    # Step 1: Standardise
    standardised_matrix, column_means, column_stds = standardise_matrix(feature_matrix)

    # Step 2: SVD — numpy returns U, singular_values (1-D), Vt (not V)
    # X̃ = U · diag(singular_values) · Vt
    left_singular_vectors, singular_values, right_singular_vectors_transposed = np.linalg.svd(
        standardised_matrix, full_matrices=False
    )
    # V = Vᵀᵀ  (columns of V are the principal component directions)
    right_singular_vectors = right_singular_vectors_transposed.T  # shape (M, min(N,M))

    # Step 3: Eigenvalues of the sample covariance matrix C = X̃ᵀX̃/(N-1)
    #         σₖ² from SVD of X̃  ↔  λₖ = σₖ²/(N-1)
    eigenvalues             = (singular_values ** 2) / max(num_windows - 1, 1)
    total_variance          = float(np.sum(eigenvalues))
    explained_variance_ratio = eigenvalues / total_variance if total_variance > 0 else eigenvalues

    # Step 4: Project data onto first n_components principal components
    # X_projected = X̃ · V[:, :n_components]
    principal_components = right_singular_vectors[:, :n_components]   # shape (M, n_components)
    projected_matrix     = standardised_matrix @ principal_components  # shape (N, n_components)

    return {
        "standardised_matrix":        standardised_matrix,
        "singular_values":            singular_values,
        "right_singular_vectors":     right_singular_vectors,
        "eigenvalues":                eigenvalues,
        "explained_variance_ratio":   explained_variance_ratio,
        "projected_matrix":           projected_matrix,
        "column_means":               column_means,
        "column_stds":                column_stds,
    }


def compute_health_index(projected_matrix: np.ndarray) -> np.ndarray:
    """
    Reduce the 2-D PCA projection to a single scalar Health Index per window.

    The Health Index (HI) is the Euclidean norm of each row in the projected
    space.  A healthy bearing clusters near the origin; degradation causes the
    projected point to drift outward, increasing ‖·‖₂.

        HI_n = ‖ [PC1_n, PC2_n] ‖₂  = √(PC1_n² + PC2_n²)

    Parameters
    ----------
    projected_matrix : shape (N, 2) — output of pca_via_svd()['projected_matrix']

    Returns
    -------
    health_index_series : shape (N,) — scalar HI for each window
    """
    health_index_series = np.linalg.norm(projected_matrix, axis=1)
    return health_index_series
