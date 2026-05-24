"""
src/analytics.py
================
Statistical Engine: Hypothesis Testing & Reliability Estimation
---------------------------------------------------------------
Two core statistical modules:

  1. Welch's T-Test (unequal-variance two-sample test)
     Tests whether the current operational signal has deviated significantly
     from the healthy baseline distribution.

     Test Statistic:
         t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)

     Welch-Satterthwaite degrees of freedom:
         ν = (s₁²/n₁ + s₂²/n₂)²
             ─────────────────────────────────────────
             (s₁²/n₁)²/(n₁−1) + (s₂²/n₂)²/(n₂−1)

  2. Two-Parameter Weibull Reliability Estimation
     Fits a Weibull distribution to the Health Index trajectory and
     estimates the Remaining Useful Life (RUL) curve.

     PDF:    f(t) = (β/λ)·(t/λ)^{β−1}·exp[−(t/λ)^β]
     CDF:    F(t) = 1 − exp[−(t/λ)^β]
     Reliability (Survival):  R(t) = exp[−(t/λ)^β]

     where β > 1 signals wear-out failure mode (increasing hazard rate).
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. WELCH'S TWO-SAMPLE T-TEST (from scratch using scipy.stats primitives)
# ---------------------------------------------------------------------------

def welch_t_test(
    baseline_samples: np.ndarray,
    current_samples: np.ndarray,
) -> dict:
    """
    Perform Welch's T-test to compare two independent samples that may have
    unequal population variances.

    This implementation manually computes the test statistic and degrees of
    freedom according to the Welch-Satterthwaite equations, then calls
    scipy.stats.t.sf for the exact p-value.  It does NOT call scipy.stats.ttest_ind
    so that the mathematical derivation remains transparent.

    Hypotheses
    ----------
    H₀ : μ_baseline = μ_current   (no structural deviation)
    H₁ : μ_baseline ≠ μ_current   (statistically significant shift detected)

    Parameters
    ----------
    baseline_samples : 1-D array — healthy reference measurements
    current_samples  : 1-D array — current operational measurements

    Returns
    -------
    dict with keys:
        'mean_baseline'      : x̄₁ (float)
        'mean_current'       : x̄₂ (float)
        'variance_baseline'  : s₁² (float)
        'variance_current'   : s₂² (float)
        'n_baseline'         : n₁ (int)
        'n_current'          : n₂ (int)
        't_statistic'        : t  (float)
        'degrees_of_freedom' : ν  (float, Welch-Satterthwaite)
        'p_value'            : two-tailed p-value (float)
        'reject_null'        : bool — True if p_value < 0.05
        'deviation_detected' : human-readable status string
    """
    # Sample sizes
    n_baseline = len(baseline_samples)
    n_current  = len(current_samples)

    # Sample means
    mean_baseline = float(np.mean(baseline_samples))
    mean_current  = float(np.mean(current_samples))

    # Unbiased sample variances (Bessel's correction: ddof=1)
    variance_baseline = float(np.var(baseline_samples, ddof=1))
    variance_current  = float(np.var(current_samples,  ddof=1))

    # Standard error of the difference of means
    se_baseline = variance_baseline / n_baseline   # s₁²/n₁
    se_current  = variance_current  / n_current    # s₂²/n₂
    se_combined = se_baseline + se_current         # s₁²/n₁ + s₂²/n₂

    # Guard against degenerate case (identical distributions)
    if se_combined < 1e-15:
        t_statistic         = 0.0
        degrees_of_freedom  = float(n_baseline + n_current - 2)
        p_value             = 1.0
    else:
        # Welch's t-statistic
        t_statistic = (mean_baseline - mean_current) / np.sqrt(se_combined)

        # Welch-Satterthwaite effective degrees of freedom
        #        (s₁²/n₁ + s₂²/n₂)²
        # ν = ─────────────────────────────────────────────
        #     (s₁²/n₁)²/(n₁−1) + (s₂²/n₂)²/(n₂−1)
        numerator   = se_combined ** 2
        denominator = (se_baseline ** 2) / (n_baseline - 1) + \
                      (se_current  ** 2) / (n_current  - 1)
        degrees_of_freedom = numerator / denominator if denominator > 0 else 1.0

        # Two-tailed p-value using the Student-t CDF
        # P(|T| ≥ |t|) = 2 · P(T ≥ |t|) = 2 · sf(|t|, df)
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=degrees_of_freedom))

    reject_null        = p_value < 0.05
    deviation_detected = (
        f"⚠️ DEVIATION DETECTED  (p = {p_value:.4f} < 0.05)"
        if reject_null
        else f"✅ NO SIGNIFICANT DEVIATION  (p = {p_value:.4f} ≥ 0.05)"
    )

    return {
        "mean_baseline":      mean_baseline,
        "mean_current":       mean_current,
        "variance_baseline":  variance_baseline,
        "variance_current":   variance_current,
        "n_baseline":         n_baseline,
        "n_current":          n_current,
        "t_statistic":        float(t_statistic),
        "degrees_of_freedom": float(degrees_of_freedom),
        "p_value":            p_value,
        "reject_null":        reject_null,
        "deviation_detected": deviation_detected,
    }


def rolling_hypothesis_test(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    channel: str = "vibration_g",
    window_size: int = 100,
) -> pd.DataFrame:
    """
    Apply Welch's T-test between a baseline signal and a rolling window of the
    current signal to produce a time-series of p-values.

    Parameters
    ----------
    baseline_df  : DataFrame with a column `channel` — healthy reference
    current_df   : DataFrame with a column `channel` — live operational data
    channel      : sensor column to test (default 'vibration_g')
    window_size  : number of samples per rolling window

    Returns
    -------
    pd.DataFrame with columns:
        window_end_idx, mean_current, t_statistic, p_value, reject_null
    """
    baseline_samples = baseline_df[channel].to_numpy()
    current_samples  = current_df[channel].to_numpy()

    records = []
    for end_idx in range(window_size, len(current_samples) + 1, window_size // 2):
        window_slice = current_samples[max(0, end_idx - window_size):end_idx]
        result       = welch_t_test(baseline_samples[:window_size], window_slice)
        records.append(
            {
                "window_end_idx": end_idx,
                "mean_current":   result["mean_current"],
                "t_statistic":    result["t_statistic"],
                "p_value":        result["p_value"],
                "reject_null":    result["reject_null"],
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. TWO-PARAMETER WEIBULL RELIABILITY ESTIMATION
# ---------------------------------------------------------------------------

def fit_weibull_distribution(
    health_index_series: np.ndarray,
) -> Tuple[float, float]:
    """
    Fit a two-parameter Weibull distribution to the Health Index time series
    using Maximum Likelihood Estimation (MLE).

    The Weibull PDF is:
        f(x; β, λ) = (β/λ) · (x/λ)^{β−1} · exp[−(x/λ)^β]

    Log-likelihood:
        ℓ(β, λ) = N·ln(β) − N·β·ln(λ) + (β−1)·Σln(xᵢ) − Σ(xᵢ/λ)^β

    We use scipy.optimize.minimize (L-BFGS-B) to maximise ℓ, initialising
    β=1.5 and λ=mean(HI) as physically motivated starting points.

    Parameters
    ----------
    health_index_series : 1-D array of Health Index values (must be > 0)

    Returns
    -------
    beta_shape  : Weibull shape parameter β  (β > 1 → wear-out failure mode)
    lambda_scale: Weibull scale parameter λ  (characteristic life, same units as HI)
    """
    # Ensure all HI values are strictly positive (Weibull domain constraint)
    hi_positive = np.clip(health_index_series, a_min=1e-6, a_max=None)
    n_samples   = len(hi_positive)

    def negative_log_likelihood(log_params: np.ndarray) -> float:
        """
        Negative log-likelihood function parameterised in log-space
        (log β, log λ) so that the optimiser is unconstrained.
        """
        log_beta, log_lambda = log_params
        beta_val   = np.exp(log_beta)
        lambda_val = np.exp(log_lambda)

        # ℓ(β,λ) = N·ln(β) − N·β·ln(λ) + (β−1)·Σln(xᵢ) − Σ(xᵢ/λ)^β
        log_likelihood = (
            n_samples * np.log(beta_val)
            - n_samples * beta_val * np.log(lambda_val)
            + (beta_val - 1.0) * np.sum(np.log(hi_positive))
            - np.sum((hi_positive / lambda_val) ** beta_val)
        )
        return -log_likelihood   # minimise the negative

    # Physically motivated initialisations
    initial_beta   = 1.5                     # slight wear-out tendency
    initial_lambda = float(np.mean(hi_positive))

    initial_log_params = np.array([np.log(initial_beta), np.log(initial_lambda)])

    result = optimize.minimize(
        negative_log_likelihood,
        x0=initial_log_params,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 10000},
    )

    beta_shape   = float(np.exp(result.x[0]))
    lambda_scale = float(np.exp(result.x[1]))

    # Clamp β ≥ 1.01 for wear-out semantics (enforced by physics of the model)
    beta_shape   = max(beta_shape, 1.01)

    return beta_shape, lambda_scale


def weibull_reliability(
    time_points: np.ndarray,
    beta_shape: float,
    lambda_scale: float,
) -> np.ndarray:
    """
    Compute the Weibull Reliability (Survival) Function R(t).

        R(t) = exp[−(t/λ)^β]

    This gives the probability that the system survives (no failure) up to
    time t, given shape parameter β and scale parameter λ.

    Parameters
    ----------
    time_points  : 1-D array of time values at which to evaluate R(t)
    beta_shape   : Weibull shape parameter β
    lambda_scale : Weibull scale parameter λ

    Returns
    -------
    reliability  : 1-D array of R(t) values in [0, 1]
    """
    reliability = np.exp(-((time_points / lambda_scale) ** beta_shape))
    return reliability


def weibull_hazard_rate(
    time_points: np.ndarray,
    beta_shape: float,
    lambda_scale: float,
) -> np.ndarray:
    """
    Compute the Weibull Hazard Rate (instantaneous failure rate) h(t).

        h(t) = f(t) / R(t) = (β/λ) · (t/λ)^{β−1}

    For β > 1: h(t) is strictly increasing → wear-out failure mode.
    For β = 1: h(t) is constant → random (memoryless) failure mode.
    For β < 1: h(t) is decreasing → infant mortality / early failure mode.

    Parameters
    ----------
    time_points  : 1-D array of time values
    beta_shape   : Weibull shape parameter β
    lambda_scale : Weibull scale parameter λ

    Returns
    -------
    hazard_rate  : 1-D array of h(t) values
    """
    hazard_rate = (beta_shape / lambda_scale) * (time_points / lambda_scale) ** (beta_shape - 1.0)
    return hazard_rate


def compute_rul_curve(
    health_index_series: np.ndarray,
    failure_threshold_reliability: float = 0.10,
) -> dict:
    """
    Full Remaining Useful Life (RUL) pipeline:
      1. Fit Weibull distribution to the Health Index trajectory.
      2. Compute the Reliability R(t) curve over a normalised time axis.
      3. Identify the B10 life: time at which R(t) = 10% (or user threshold).
      4. Compute the hazard rate h(t) for visualisation.

    Parameters
    ----------
    health_index_series           : 1-D array of HI values over time
    failure_threshold_reliability : R(t) threshold below which the system is
                                    considered failed (default 0.10 = 10% survival)

    Returns
    -------
    dict with keys:
        'beta_shape'              : fitted β
        'lambda_scale'            : fitted λ
        'time_axis'               : normalised time array for plotting
        'reliability_curve'       : R(t), shape (T,)
        'hazard_curve'            : h(t), shape (T,)
        'b10_life'                : time at which R = failure_threshold
        'mean_time_to_failure'    : E[T] = λ · Γ(1 + 1/β)
    """
    beta_shape, lambda_scale = fit_weibull_distribution(health_index_series)

    # Normalised time axis: 0 → 3×λ (covers most of the failure probability mass)
    time_axis = np.linspace(1e-6, 3.0 * lambda_scale, num=500)

    reliability_curve = weibull_reliability(time_axis, beta_shape, lambda_scale)
    hazard_curve      = weibull_hazard_rate(time_axis, beta_shape, lambda_scale)

    # B10 life: smallest t such that R(t) ≤ failure_threshold
    below_threshold_indices = np.where(reliability_curve <= failure_threshold_reliability)[0]
    b10_life = float(time_axis[below_threshold_indices[0]]) if len(below_threshold_indices) > 0 else float(time_axis[-1])

    # Mean Time To Failure: E[T] = λ · Γ(1 + 1/β)
    from scipy.special import gamma as gamma_func
    mean_time_to_failure = lambda_scale * gamma_func(1.0 + 1.0 / beta_shape)

    return {
        "beta_shape":           beta_shape,
        "lambda_scale":         lambda_scale,
        "time_axis":            time_axis,
        "reliability_curve":    reliability_curve,
        "hazard_curve":         hazard_curve,
        "b10_life":             b10_life,
        "mean_time_to_failure": float(mean_time_to_failure),
    }
