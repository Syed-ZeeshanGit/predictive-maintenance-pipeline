"""
data/generator.py
=================
Physics-Based High-Frequency Sensor Simulator
----------------------------------------------
Simulates a rotating industrial bearing system with three sensor channels:
  - Vibration:    Compound sinusoidal wave with harmonic distortion
  - Temperature:  Logarithmic thermal drift from mechanical friction
  - Voltage:      Steady AC supply with stochastic spike anomalies

Mathematical Model
------------------
Vibration signal:
    V(t) = A₁·sin(2π·f₁(δ)·t) + A₂·sin(2π·f₂·t) + ε(t, δ)

where f₁(δ) = 10 + 35·δ Hz  (fundamental shifts with degradation),
      f₂     = 30 Hz          (3rd harmonic, fixed),
      ε(t,δ) = non-Gaussian noise scaled by δ.

Temperature signal:
    T(t) = μ_T + σ_T·𝒩(0,1) + α·ln(1 + δ·t/N)

Voltage signal:
    V_err(t) = 230 + 𝒩(0, 1.5) + spike_mask(t, δ) · ΔV_spike
"""

import numpy as np
import pandas as pd


def generate_sensor_stream(
    degradation_factor: float = 0.0,
    sampling_rate_hz: int = 1000,
    duration_seconds: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a multi-channel sensor time-series for a degrading bearing system.

    Parameters
    ----------
    degradation_factor : float
        Normalized wear index in [0.0, 1.0]. At 0.0 the system is healthy;
        at 1.0 the bearing is near failure.
    sampling_rate_hz : int
        Number of samples per second (Hz). Default 1000 Hz.
    duration_seconds : int
        Length of the simulated recording window in seconds.
    random_seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ['time_s', 'vibration_g', 'temperature_c', 'voltage_v']
        Indexed by integer sample number.
    """
    rng = np.random.default_rng(random_seed)

    num_samples = sampling_rate_hz * duration_seconds
    time_axis = np.linspace(0.0, duration_seconds, num=num_samples, endpoint=False)

    # ------------------------------------------------------------------
    # 1. VIBRATION SIGNAL  V(t)
    # ------------------------------------------------------------------
    # Fundamental frequency shifts from 10 Hz (healthy) → 45 Hz (failure).
    fundamental_freq_hz = 10.0 + 35.0 * degradation_factor   # f₁(δ)
    harmonic_freq_hz    = 30.0                                 # f₂, fixed 3rd harmonic

    amplitude_fundamental = 1.0
    amplitude_harmonic    = 0.4

    vibration_clean = (
        amplitude_fundamental * np.sin(2.0 * np.pi * fundamental_freq_hz * time_axis)
        + amplitude_harmonic  * np.sin(2.0 * np.pi * harmonic_freq_hz    * time_axis)
    )

    # Non-Gaussian noise: mixture of Gaussian and Laplacian (heavy tails under wear)
    gaussian_noise  = rng.normal(loc=0.0, scale=0.05, size=num_samples)
    laplacian_noise = rng.laplace(loc=0.0, scale=0.15 * degradation_factor, size=num_samples)
    noise_signal    = gaussian_noise + laplacian_noise

    vibration_g = vibration_clean + noise_signal

    # ------------------------------------------------------------------
    # 2. TEMPERATURE SIGNAL  T(t)
    # ------------------------------------------------------------------
    # Logarithmic thermal drift: friction heat accumulates with degradation.
    baseline_temp_c   = 65.0
    temp_variance_c   = 1.2    # σ_T: sensor measurement uncertainty
    thermal_drift_coeff = 18.0  # α: degrees C of drift at full degradation

    thermal_drift = thermal_drift_coeff * np.log1p(
        degradation_factor * time_axis / duration_seconds
    )
    temperature_c = (
        baseline_temp_c
        + rng.normal(loc=0.0, scale=temp_variance_c, size=num_samples)
        + thermal_drift
    )

    # ------------------------------------------------------------------
    # 3. VOLTAGE SIGNAL  V_err(t)
    # ------------------------------------------------------------------
    # AC supply at 230 V ± small Gaussian noise + random voltage drops.
    nominal_voltage_v  = 230.0
    voltage_noise_std  = 1.5   # σ: normal supply fluctuation

    # Spike probability scales with degradation (more structural cracks → more EMI)
    spike_probability  = 0.002 + 0.04 * degradation_factor
    spike_mask         = rng.random(size=num_samples) < spike_probability

    # Spike magnitude: voltage DROP (negative transient), Gaussian magnitude
    spike_magnitude    = rng.normal(loc=-18.0, scale=4.0, size=num_samples)

    voltage_v = (
        nominal_voltage_v
        + rng.normal(loc=0.0, scale=voltage_noise_std, size=num_samples)
        + spike_mask * spike_magnitude
    )

    # ------------------------------------------------------------------
    # 4. ASSEMBLE DATAFRAME
    # ------------------------------------------------------------------
    sensor_df = pd.DataFrame(
        {
            "time_s":        time_axis,
            "vibration_g":   vibration_g,
            "temperature_c": temperature_c,
            "voltage_v":     voltage_v,
        }
    )

    return sensor_df


def get_healthy_baseline(
    sampling_rate_hz: int = 1000,
    duration_seconds: int = 5,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Convenience wrapper that returns a clean, healthy-state signal
    (degradation_factor = 0.0) for use as a statistical baseline.

    Returns
    -------
    pd.DataFrame
        Same schema as generate_sensor_stream().
    """
    return generate_sensor_stream(
        degradation_factor=0.0,
        sampling_rate_hz=sampling_rate_hz,
        duration_seconds=duration_seconds,
        random_seed=random_seed,
    )
