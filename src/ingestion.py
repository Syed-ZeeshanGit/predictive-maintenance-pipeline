"""
src/ingestion.py
================
Sliding-Window Signal Processing & Feature Extraction
------------------------------------------------------
Applies a sliding window over raw sensor streams to produce overlapping
time-domain feature windows.  Each window is a candidate for downstream
FFT analysis and PCA projection.

Design
------
- Window length  W : number of samples per segment (e.g. 512 or 1024)
- Hop size       H : stride between consecutive windows (H ≤ W)
- Overlap ratio    : (W - H) / W × 100 %

For each window we extract:
  - raw samples (passed to FFT)
  - time-domain statistics: mean, std, RMS, peak-to-peak, kurtosis, skewness
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class SensorWindow:
    """
    Container for a single sliding-window segment across all sensor channels.

    Attributes
    ----------
    window_index   : sequential position of this window in the stream
    start_sample   : integer index of the first sample
    end_sample     : integer index of the last sample (exclusive)
    time_start_s   : physical timestamp of first sample (seconds)
    time_end_s     : physical timestamp of last sample  (seconds)
    vibration_raw  : raw vibration samples  – shape (W,)
    temperature_raw: raw temperature samples – shape (W,)
    voltage_raw    : raw voltage samples    – shape (W,)
    stats          : dict of per-channel time-domain statistics
    """
    window_index:    int
    start_sample:    int
    end_sample:      int
    time_start_s:    float
    time_end_s:      float
    vibration_raw:   np.ndarray
    temperature_raw: np.ndarray
    voltage_raw:     np.ndarray
    stats:           dict = field(default_factory=dict)


def _compute_time_domain_stats(signal: np.ndarray, channel_name: str) -> dict:
    """
    Compute six standard time-domain condition indicators for one channel.

    Parameters
    ----------
    signal       : 1-D array of raw sensor samples within the window
    channel_name : label prefix for the returned dictionary keys

    Returns
    -------
    dict with keys: {channel_name}_mean, _std, _rms, _peak_to_peak,
                    _kurtosis, _skewness
    """
    n = len(signal)

    mean_val        = float(np.mean(signal))
    std_val         = float(np.std(signal, ddof=1))       # sample std
    rms_val         = float(np.sqrt(np.mean(signal ** 2)))
    peak_to_peak    = float(np.max(signal) - np.min(signal))

    # Kurtosis (excess): κ = E[(X-μ)⁴] / σ⁴  − 3
    centred         = signal - mean_val
    if std_val > 1e-12:
        kurtosis_val = float(np.mean(centred ** 4) / std_val ** 4) - 3.0
        skewness_val = float(np.mean(centred ** 3) / std_val ** 3)
    else:
        kurtosis_val = 0.0
        skewness_val = 0.0

    return {
        f"{channel_name}_mean":         mean_val,
        f"{channel_name}_std":          std_val,
        f"{channel_name}_rms":          rms_val,
        f"{channel_name}_peak_to_peak": peak_to_peak,
        f"{channel_name}_kurtosis":     kurtosis_val,
        f"{channel_name}_skewness":     skewness_val,
    }


def sliding_window_extract(
    sensor_df: pd.DataFrame,
    window_length: int = 512,
    hop_size: int = 256,
) -> List[SensorWindow]:
    """
    Partition a sensor DataFrame into overlapping fixed-length windows.

    Parameters
    ----------
    sensor_df     : DataFrame with columns ['time_s','vibration_g',
                    'temperature_c','voltage_v'] (output of generator.py)
    window_length : number of samples per window  (W)
    hop_size      : samples to advance between windows (H); overlap = W-H

    Returns
    -------
    List[SensorWindow]
        One SensorWindow object per valid non-truncated window.
    """
    time_arr        = sensor_df["time_s"].to_numpy()
    vibration_arr   = sensor_df["vibration_g"].to_numpy()
    temperature_arr = sensor_df["temperature_c"].to_numpy()
    voltage_arr     = sensor_df["voltage_v"].to_numpy()

    total_samples = len(sensor_df)
    windows: List[SensorWindow] = []

    window_index = 0
    start = 0

    while start + window_length <= total_samples:
        end = start + window_length

        vib_window  = vibration_arr[start:end]
        temp_window = temperature_arr[start:end]
        volt_window = voltage_arr[start:end]

        # Collect time-domain statistics for all three channels
        stats: dict = {}
        stats.update(_compute_time_domain_stats(vib_window,  "vibration"))
        stats.update(_compute_time_domain_stats(temp_window, "temperature"))
        stats.update(_compute_time_domain_stats(volt_window, "voltage"))

        window_obj = SensorWindow(
            window_index=window_index,
            start_sample=start,
            end_sample=end,
            time_start_s=float(time_arr[start]),
            time_end_s=float(time_arr[end - 1]),
            vibration_raw=vib_window,
            temperature_raw=temp_window,
            voltage_raw=volt_window,
            stats=stats,
        )
        windows.append(window_obj)

        start        += hop_size
        window_index += 1

    return windows


def windows_to_stats_dataframe(windows: List[SensorWindow]) -> pd.DataFrame:
    """
    Flatten the list of SensorWindow objects into a tidy DataFrame where
    each row is one window and each column is a time-domain statistic or
    metadata field.

    Parameters
    ----------
    windows : output of sliding_window_extract()

    Returns
    -------
    pd.DataFrame  – one row per window, ready for PCA / hypothesis testing
    """
    records = []
    for w in windows:
        row = {
            "window_index": w.window_index,
            "time_start_s": w.time_start_s,
            "time_end_s":   w.time_end_s,
        }
        row.update(w.stats)
        records.append(row)

    return pd.DataFrame(records)
