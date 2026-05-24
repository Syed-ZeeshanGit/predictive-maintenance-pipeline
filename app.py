import sys
import os
import inspect
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Path resolution — allow running from repo root:  streamlit run app.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generator  import generate_sensor_stream, get_healthy_baseline
from src.ingestion   import sliding_window_extract, windows_to_stats_dataframe
from src.feature_eng import compute_fft_features, batch_fft_features, pca_via_svd, compute_health_index
from src.analytics   import welch_t_test, compute_rul_curve

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Pipeline",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — industrial dark-mode aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global dark canvas ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* ── Header strip ── */
    .main-header {
        background: linear-gradient(135deg, #1a2332 0%, #0d1117 60%, #1a1a2e 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 24px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #f97316, #ef4444, #8b5cf6, #06b6d4);
    }
    .main-header h1 { color: #f0f6fc; font-size: 1.9rem; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { color: #8b949e; margin: 6px 0 0; font-size: 0.9rem; }

    /* ── KPI metric cards ── */
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 18px 20px;
        position: relative;
    }
    .kpi-card .label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-card .value { color: #f0f6fc; font-size: 1.55rem; font-weight: 700; margin-top: 4px; font-family: 'Courier New', monospace; }
    .kpi-card .unit  { color: #58a6ff; font-size: 0.75rem; }
    .kpi-card.alert  { border-color: #f97316; }
    .kpi-card.danger { border-color: #ef4444; }
    .kpi-card.ok     { border-color: #3fb950; }

    /* ── Math equation blocks ── */
    .math-block {
        background: #0d1117;
        border: 1px solid #21262d;
        border-left: 4px solid #58a6ff;
        border-radius: 6px;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .math-block h4 { color: #58a6ff; margin: 0 0 12px; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.06em; }

    /* ── Code alignment blocks ── */
    .code-label { color: #3fb950; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; margin: 16px 0 4px; }

    /* ── Status badge ── */
    .status-ok     { background:#1a3a1a; color:#3fb950; border:1px solid #3fb950; border-radius:6px; padding:8px 16px; font-weight:600; }
    .status-warn   { background:#3a2a0a; color:#f97316; border:1px solid #f97316; border-radius:6px; padding:8px 16px; font-weight:600; }
    .status-danger { background:#3a0a0a; color:#ef4444; border:1px solid #ef4444; border-radius:6px; padding:8px 16px; font-weight:600; }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 4px; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab"]      { color: #8b949e; border-radius: 6px; }
    .stTabs [aria-selected="true"]    { background: #21262d !important; color: #f0f6fc !important; }

    /* ── Dividers ── */
    hr { border-color: #21262d; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# PLOTLY THEME — consistent dark engineering aesthetic
# ===========================================================================
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_VIBRATION  = "#58a6ff"
COLOR_TEMPERATURE= "#f97316"
COLOR_VOLTAGE    = "#a371f7"
COLOR_HEALTH     = "#3fb950"
COLOR_RELIABILITY= "#06b6d4"
COLOR_HAZARD     = "#ef4444"
COLOR_PVALUE     = "#e3b341"


def _dark_fig_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply uniform dark-mode layout to a Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="'Courier New', monospace", color="#c9d1d9", size=12),
        title=dict(text=title, font=dict(size=14, color="#f0f6fc")),
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    )
    return fig


# ===========================================================================
# SIDEBAR — Controls
# ===========================================================================
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")
    st.markdown("---")

    degradation_factor = st.slider(
        label="🔧 Operational Degradation Factor (δ)",
        min_value=0.00,
        max_value=1.00,
        value=0.30,
        step=0.01,
        help=(
            "Simulates bearing wear progression.\n"
            "0.0 = pristine / healthy\n"
            "1.0 = imminent failure"
        ),
    )

    st.markdown("---")
    st.markdown("#### Signal Parameters")

    sampling_rate_hz = st.selectbox(
        "Sampling Rate (Hz)",
        options=[500, 1000, 2000],
        index=1,
    )

    duration_seconds = st.slider(
        "Recording Duration (s)",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
    )

    window_length = st.select_slider(
        "FFT Window Length (samples)",
        options=[256, 512, 1024],
        value=512,
    )

    st.markdown("---")
    st.markdown("#### Reliability Threshold")
    failure_threshold = st.slider(
        "R(t) Failure Threshold",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
        format="%.2f",
        help="System is considered failed when R(t) drops below this probability.",
    )

    st.markdown("---")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.78rem;'>"
        "All backend modules recompute on slider change.<br>"
        "Architecture: numpy · scipy · pandas<br>"
        "No sklearn preprocessing or modelling."
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# BACKEND — Run full pipeline (cached on input parameters)
# ===========================================================================
@st.cache_data(show_spinner=False)
def run_pipeline(
    degradation_factor: float,
    sampling_rate_hz: int,
    duration_seconds: int,
    window_length: int,
    failure_threshold: float,
) -> dict:
    """
    Execute the complete predictive maintenance pipeline and return all
    intermediate and final results needed for the UI.

    Parameters mirror the sidebar controls so Streamlit cache invalidates
    correctly when any slider changes.
    """
    hop_size = window_length // 2   # 50% overlap

    # ── 1. Simulate sensor streams ──────────────────────────────────────────
    current_df  = generate_sensor_stream(
        degradation_factor=degradation_factor,
        sampling_rate_hz=sampling_rate_hz,
        duration_seconds=duration_seconds,
        random_seed=42,
    )
    baseline_df = get_healthy_baseline(
        sampling_rate_hz=sampling_rate_hz,
        duration_seconds=duration_seconds,
        random_seed=0,
    )

    # ── 2. Sliding-window extraction ────────────────────────────────────────
    current_windows  = sliding_window_extract(current_df,  window_length, hop_size)
    baseline_windows = sliding_window_extract(baseline_df, window_length, hop_size)

    current_stats_df  = windows_to_stats_dataframe(current_windows)
    baseline_stats_df = windows_to_stats_dataframe(baseline_windows)

    # ── 3. FFT features (last window for spectrum display) ──────────────────
    last_window   = current_windows[-1]
    fft_result    = compute_fft_features(last_window.vibration_raw, sampling_rate_hz)
    fft_df        = batch_fft_features(current_windows, sampling_rate_hz)

    # ── 4. PCA / SVD Health Index ───────────────────────────────────────────
    # Build feature matrix from time-domain stats (6 cols × 3 channels = 18 features)
    feature_cols = [c for c in current_stats_df.columns
                    if c not in ("window_index", "time_start_s", "time_end_s")]

    current_feature_matrix  = current_stats_df[feature_cols].to_numpy()
    baseline_feature_matrix = baseline_stats_df[feature_cols].to_numpy()

    # Stack baseline + current for global PCA (so the projection is consistent)
    n_baseline_windows = len(baseline_windows)
    combined_matrix    = np.vstack([baseline_feature_matrix, current_feature_matrix])

    pca_result         = pca_via_svd(combined_matrix, n_components=2)
    projected_combined = pca_result["projected_matrix"]

    projected_baseline = projected_combined[:n_baseline_windows]
    projected_current  = projected_combined[n_baseline_windows:]

    health_index_baseline = compute_health_index(projected_baseline)
    health_index_current  = compute_health_index(projected_current)

    # ── 5. Welch's T-test ───────────────────────────────────────────────────
    t_test_vibration = welch_t_test(
        baseline_df["vibration_g"].to_numpy(),
        current_df["vibration_g"].to_numpy(),
    )
    t_test_health = welch_t_test(health_index_baseline, health_index_current)

    # ── 6. Weibull RUL curve ────────────────────────────────────────────────
    rul_result = compute_rul_curve(
        health_index_current,
        failure_threshold_reliability=failure_threshold,
    )

    return dict(
        current_df=current_df,
        baseline_df=baseline_df,
        fft_result=fft_result,
        fft_df=fft_df,
        current_stats_df=current_stats_df,
        pca_result=pca_result,
        projected_current=projected_current,
        projected_baseline=projected_baseline,
        health_index_current=health_index_current,
        health_index_baseline=health_index_baseline,
        t_test_vibration=t_test_vibration,
        t_test_health=t_test_health,
        rul_result=rul_result,
        feature_cols=feature_cols,
    )


# Run pipeline
with st.spinner("🔄 Running pipeline…"):
    pipeline = run_pipeline(
        degradation_factor,
        sampling_rate_hz,
        duration_seconds,
        window_length,
        failure_threshold,
    )

# Unpack pipeline results
current_df           = pipeline["current_df"]
baseline_df          = pipeline["baseline_df"]
fft_result           = pipeline["fft_result"]
fft_df               = pipeline["fft_df"]
pca_result           = pipeline["pca_result"]
projected_current    = pipeline["projected_current"]
projected_baseline   = pipeline["projected_baseline"]
health_index_current = pipeline["health_index_current"]
health_index_baseline= pipeline["health_index_baseline"]
t_test_vibration     = pipeline["t_test_vibration"]
t_test_health        = pipeline["t_test_health"]
rul_result           = pipeline["rul_result"]

beta_shape   = rul_result["beta_shape"]
lambda_scale = rul_result["lambda_scale"]
p_value_vib  = t_test_vibration["p_value"]

# ===========================================================================
# HEADER
# ===========================================================================
st.markdown(
    f"""
    <div class="main-header">
        <h1>⚙️ Predictive Maintenance &amp; Fault Diagnostics Pipeline</h1>
        <p>
            Physics-Based Sensor Simulation  ·  DFT / FFT Spectral Analysis  ·
            SVD/PCA Health Index  ·  Welch's T-Test  ·  Weibull Reliability Estimation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# KPI STRIP
# ===========================================================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

degradation_pct   = int(degradation_factor * 100)
peak_freq         = fft_result["spectral_peak_freq_hz"]
spectral_entropy  = fft_result["spectral_entropy"]
mean_hi           = float(np.mean(health_index_current))

card_class_hi  = "danger" if mean_hi > 2.5 else "alert" if mean_hi > 1.0 else "ok"
card_class_p   = "danger" if p_value_vib < 0.001 else "alert" if p_value_vib < 0.05 else "ok"
card_class_beta= "alert"  if beta_shape > 2 else "ok"

with kpi1:
    st.markdown(
        f'<div class="kpi-card {"alert" if degradation_pct > 60 else "ok"}">'
        f'<div class="label">Degradation Factor</div>'
        f'<div class="value">{degradation_factor:.2f}</div>'
        f'<div class="unit">{degradation_pct}% of failure threshold</div></div>',
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f'<div class="kpi-card {card_class_hi}">'
        f'<div class="label">Mean Health Index</div>'
        f'<div class="value">{mean_hi:.3f}</div>'
        f'<div class="unit">PCA ‖PC₁, PC₂‖₂</div></div>',
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f'<div class="kpi-card {card_class_p}">'
        f'<div class="label">Welch p-value</div>'
        f'<div class="value">{p_value_vib:.2e}</div>'
        f'<div class="unit">{"H₀ rejected" if p_value_vib < 0.05 else "H₀ not rejected"}</div></div>',
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f'<div class="kpi-card {card_class_beta}">'
        f'<div class="label">Weibull β (Shape)</div>'
        f'<div class="value">{beta_shape:.3f}</div>'
        f'<div class="unit">{"Wear-out ▲" if beta_shape > 1 else "Random ●"}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ===========================================================================
# TABS
# ===========================================================================
tab_telemetry, tab_math, tab_diagnostics = st.tabs(
    ["📊  Real-Time Telemetry", "🧮  Mathematical Architecture", "📈  Survival & Diagnostics Analysis"]
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — REAL-TIME TELEMETRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_telemetry:

    # ── Time-series plot: all three channels ──
    fig_ts = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Vibration (g-force)", "Temperature (°C)", "Voltage (V)"),
        vertical_spacing=0.06,
    )

    # Downsample for display performance (every 4th point)
    ds = 4
    t = current_df["time_s"].to_numpy()[::ds]

    fig_ts.add_trace(
        go.Scatter(x=t, y=current_df["vibration_g"].to_numpy()[::ds],
                   name="Vibration",  line=dict(color=COLOR_VIBRATION,   width=1.0)),
        row=1, col=1,
    )
    fig_ts.add_trace(
        go.Scatter(x=t, y=current_df["temperature_c"].to_numpy()[::ds],
                   name="Temperature", line=dict(color=COLOR_TEMPERATURE, width=1.5)),
        row=2, col=1,
    )
    fig_ts.add_trace(
        go.Scatter(x=t, y=current_df["voltage_v"].to_numpy()[::ds],
                   name="Voltage",    line=dict(color=COLOR_VOLTAGE,      width=1.0)),
        row=3, col=1,
    )

    fig_ts.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="'Courier New', monospace", color="#c9d1d9"),
        height=480,
        margin=dict(l=50, r=30, t=60, b=40),
        showlegend=False,
        title=dict(
            text=f"Multi-Channel Sensor Stream  |  δ = {degradation_factor:.2f}  |  fs = {sampling_rate_hz} Hz",
            font=dict(size=14, color="#f0f6fc"),
        ),
    )
    fig_ts.update_xaxes(gridcolor="#21262d", linecolor="#30363d")
    fig_ts.update_yaxes(gridcolor="#21262d", linecolor="#30363d")
    fig_ts.update_xaxes(title_text="Time (s)", row=3, col=1)

    st.plotly_chart(fig_ts, use_container_width=True)

    # ── FFT Spectrum + Spectral summary ──
    col_fft, col_fft_stats = st.columns([3, 1])

    with col_fft:
        freqs = fft_result["frequencies_hz"]
        psd   = fft_result["power_spectral_density"]
        # Show only up to 150 Hz for clarity
        freq_mask = freqs <= 150
        fig_fft = go.Figure()
        fig_fft.add_trace(
            go.Scatter(
                x=freqs[freq_mask],
                y=psd[freq_mask],
                name="PSD",
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.12)",
                line=dict(color=COLOR_VIBRATION, width=1.5),
            )
        )
        # Mark peak
        fig_fft.add_vline(
            x=fft_result["spectral_peak_freq_hz"],
            line=dict(color="#ef4444", dash="dash", width=1.5),
            annotation_text=f"Peak: {fft_result['spectral_peak_freq_hz']:.1f} Hz",
            annotation_font_color="#ef4444",
        )
        _dark_fig_layout(
            fig_fft,
            f"One-Sided Power Spectral Density (Vibration)  |  Window: {window_length} samples, Hann-weighted"
        )
        fig_fft.update_xaxes(title_text="Frequency (Hz)")
        fig_fft.update_yaxes(title_text="PSD")
        fig_fft.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_fft, use_container_width=True)

    with col_fft_stats:
        st.markdown("**FFT Indicators**")
        st.metric("Spectral Peak Freq.", f"{fft_result['spectral_peak_freq_hz']:.1f} Hz")
        st.metric("Spectral Entropy",    f"{fft_result['spectral_entropy']:.4f}")
        st.metric("Peak PSD",            f"{fft_result['spectral_peak_power']:.2e}")
        st.markdown("---")
        expected_healthy_freq = 10.0
        expected_degraded_freq = 10.0 + 35.0 * degradation_factor
        st.markdown(
            f"**Expected f₁:** `{expected_degraded_freq:.1f} Hz`\n\n"
            f"_Healthy baseline:_ `{expected_healthy_freq:.0f} Hz`\n\n"
            f"_Shift at δ={degradation_factor:.2f}:_ `+{expected_degraded_freq - expected_healthy_freq:.1f} Hz`"
        )

    # ── Rolling FFT peak frequency over time ──
    fig_freq_drift = go.Figure()
    window_times   = (fft_df["window_index"] * (window_length // 2)) / sampling_rate_hz
    fig_freq_drift.add_trace(
        go.Scatter(
            x=window_times,
            y=fft_df["spectral_peak_freq_hz"],
            mode="lines+markers",
            marker=dict(size=5, color=COLOR_VIBRATION),
            line=dict(color=COLOR_VIBRATION, width=2),
            name="Peak Frequency",
        )
    )
    fig_freq_drift.add_hline(
        y=10.0, line=dict(color="#3fb950", dash="dot", width=1),
        annotation_text="Healthy baseline (10 Hz)", annotation_font_color="#3fb950"
    )
    _dark_fig_layout(fig_freq_drift, "Spectral Peak Frequency Drift Over Time (Per Window)")
    fig_freq_drift.update_xaxes(title_text="Time (s)")
    fig_freq_drift.update_yaxes(title_text="Peak Frequency (Hz)")
    fig_freq_drift.update_layout(height=270, showlegend=False)
    st.plotly_chart(fig_freq_drift, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — MATHEMATICAL ARCHITECTURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_math:

    st.markdown("## 🧮 Mathematical Architecture")
    st.markdown(
        "Each formula below is immediately followed by the exact Python source that implements it, "
        "demonstrating one-to-one alignment between the mathematics and the code."
    )

    # ────────────────────────────────────────────
    # SECTION A: FFT / DFT
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 1. Discrete Fourier Transform & Spectral Feature Extraction")

    st.markdown(
        '<div class="math-block"><h4>DFT Definition (Cooley-Tukey FFT)</h4></div>',
        unsafe_allow_html=True,
    )
    st.latex(
        r"""
        X(f_k) = \sum_{n=0}^{N-1} x(n) \cdot w(n) \cdot e^{-j 2\pi k n / N}
        \quad k = 0, 1, \ldots, N-1
        """
    )
    st.markdown("where $w(n) = \\frac{1}{2}\\left(1 - \\cos\\frac{2\\pi n}{N-1}\\right)$ is the **Hann window** to suppress spectral leakage.")

    st.latex(
        r"""
        S(f_k) = \frac{|X(f_k)|^2}{N^2}
        \qquad \text{(Normalised one-sided Power Spectral Density)}
        """
    )

    st.latex(
        r"""
        H_s = -\sum_{k} \tilde{p}_k \ln \tilde{p}_k, \qquad
        \tilde{p}_k = \frac{S(f_k)}{\sum_j S(f_j)}
        \qquad \text{(Spectral Entropy)}
        """
    )

    st.markdown('<div class="code-label">↳ Source: src/feature_eng.py — compute_fft_features()</div>', unsafe_allow_html=True)

    fft_source = textwrap.dedent("""\
        # Hann window  w(n) = 0.5·(1 − cos(2πn/(N−1)))
        hann_window    = np.hanning(num_samples)
        windowed_signal = vibration_window * hann_window

        # Complex DFT via NumPy (Cooley-Tukey)
        complex_spectrum = np.fft.fft(windowed_signal)

        # One-sided PSD  S(fₖ) = |X(fₖ)|² / N²
        num_positive_bins      = num_samples // 2 + 1
        power_spectral_density = (np.abs(complex_spectrum[:num_positive_bins]) ** 2) / (num_samples ** 2)
        power_spectral_density[1:-1] *= 2.0   # conserve total energy

        # Spectral Peak Frequency
        peak_bin_index        = int(np.argmax(power_spectral_density))
        spectral_peak_freq_hz = float(frequencies_hz[peak_bin_index])

        # Spectral Entropy  Hₛ = −Σ p̃ₖ · ln(p̃ₖ)
        normalised_psd  = power_spectral_density / np.sum(power_spectral_density)
        nonzero_mask    = normalised_psd > 0.0
        spectral_entropy = -np.sum(normalised_psd[nonzero_mask] * np.log(normalised_psd[nonzero_mask]))
    """)
    st.code(fft_source, language="python")

    # Live computed values
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Spectral Peak  fₚ", f"{fft_result['spectral_peak_freq_hz']:.2f} Hz")
    col_b.metric("Spectral Entropy Hₛ", f"{fft_result['spectral_entropy']:.5f}")
    col_c.metric("Expected fₚ at δ",  f"{10 + 35*degradation_factor:.2f} Hz")

    # ────────────────────────────────────────────
    # SECTION B: SVD / PCA
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2. Principal Component Analysis via Singular Value Decomposition")

    st.markdown(
        '<div class="math-block"><h4>SVD Decomposition of the Standardised Feature Matrix</h4></div>',
        unsafe_allow_html=True,
    )
    st.latex(
        r"""
        \tilde{X} = U \, \Sigma \, V^T
        \qquad
        \begin{cases}
            U  \in \mathbb{R}^{N \times N} & \text{left singular vectors (sample modes)}  \\
            \Sigma \in \mathbb{R}^{N \times M} & \text{diagonal matrix of singular values}  \\
            V  \in \mathbb{R}^{M \times M} & \text{right singular vectors (feature modes / PCs)}
        \end{cases}
        """
    )
    st.latex(
        r"""
        \lambda_k = \frac{\sigma_k^2}{N - 1}
        \qquad \text{(eigenvalue of the sample covariance matrix)}
        """
    )
    st.latex(
        r"""
        X_{\text{projected}} = \tilde{X} \cdot V[:,\, :2]
        \qquad \text{(projection onto first 2 PCs)}
        """
    )
    st.latex(
        r"""
        \text{Health Index}_n = \left\| \left[ \text{PC1}_n,\, \text{PC2}_n \right] \right\|_2
        = \sqrt{\text{PC1}_n^2 + \text{PC2}_n^2}
        """
    )

    st.markdown('<div class="code-label">↳ Source: src/feature_eng.py — pca_via_svd() + compute_health_index()</div>', unsafe_allow_html=True)
    svd_source = textwrap.dedent("""\
        # Step 1: Zero-centre and unit-scale  X̃[:,j] = (X[:,j] − μⱼ) / σⱼ
        column_means         = np.mean(feature_matrix, axis=0)
        column_stds          = np.std( feature_matrix, axis=0, ddof=1)
        standardised_matrix  = (feature_matrix - column_means) / column_stds

        # Step 2: Full SVD — X̃ = U · diag(σ) · Vᵀ
        U, singular_values, Vt = np.linalg.svd(standardised_matrix, full_matrices=False)
        right_singular_vectors  = Vt.T          # V: columns are PC directions

        # Step 3: Eigenvalues of sample covariance  λₖ = σₖ²/(N−1)
        eigenvalues              = (singular_values ** 2) / (N - 1)
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

        # Step 4: Project  X_projected = X̃ · V[:, :2]
        principal_components = right_singular_vectors[:, :2]  # shape (M, 2)
        projected_matrix     = standardised_matrix @ principal_components

        # Health Index  HI_n = ‖ [PC1_n, PC2_n] ‖₂
        health_index_series = np.linalg.norm(projected_matrix, axis=1)
    """)
    st.code(svd_source, language="python")

    # Explained variance bar chart
    ev_ratios = pca_result["explained_variance_ratio"][:6] * 100
    fig_ev = go.Figure(
        go.Bar(
            x=[f"PC{i+1}" for i in range(len(ev_ratios))],
            y=ev_ratios,
            marker_color=[COLOR_HEALTH if i < 2 else "#30363d" for i in range(len(ev_ratios))],
            text=[f"{v:.1f}%" for v in ev_ratios],
            textposition="outside",
        )
    )
    _dark_fig_layout(fig_ev, "Explained Variance Ratio by Principal Component")
    fig_ev.update_yaxes(title_text="Explained Variance (%)")
    fig_ev.update_layout(height=280, showlegend=False)
    st.plotly_chart(fig_ev, use_container_width=True)

    cum_var = float(np.sum(pca_result["explained_variance_ratio"][:2]) * 100)
    st.info(f"**PC1 + PC2 capture {cum_var:.1f}% of total variance** in the {len(pca_result['eigenvalues'])}-dimensional feature space.")

    # ────────────────────────────────────────────
    # SECTION C: Welch's T-Test
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3. Welch's T-Test (Unequal Variances)")

    st.markdown(
        '<div class="math-block"><h4>Welch\'s Test Statistic</h4></div>',
        unsafe_allow_html=True,
    )
    st.latex(
        r"""
        t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}}
        """
    )
    st.latex(
        r"""
        \nu = \frac{\left(\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}\right)^2}
                   {\dfrac{(s_1^2/n_1)^2}{n_1 - 1} + \dfrac{(s_2^2/n_2)^2}{n_2 - 1}}
        \qquad \text{(Welch–Satterthwaite d.o.f.)}
        """
    )
    st.latex(
        r"""
        p\text{-value} = 2 \cdot P\!\left(T_\nu \geq |t|\right) = 2 \cdot \text{sf}(|t|,\, \nu)
        """
    )

    st.markdown('<div class="code-label">↳ Source: src/analytics.py — welch_t_test()</div>', unsafe_allow_html=True)
    welch_source = textwrap.dedent("""\
        # Unbiased sample variances (Bessel's correction)
        variance_baseline = np.var(baseline_samples, ddof=1)     # s₁²
        variance_current  = np.var(current_samples,  ddof=1)     # s₂²

        se_combined = variance_baseline/n_baseline + variance_current/n_current

        # Welch's t-statistic
        t_statistic = (mean_baseline - mean_current) / np.sqrt(se_combined)

        # Welch-Satterthwaite degrees of freedom
        numerator          = se_combined ** 2
        denominator        = (se_baseline**2)/(n_baseline-1) + (se_current**2)/(n_current-1)
        degrees_of_freedom = numerator / denominator

        # Two-tailed p-value  P(|T| ≥ |t|) = 2·sf(|t|, df)
        p_value = 2.0 * scipy.stats.t.sf(abs(t_statistic), df=degrees_of_freedom)
    """)
    st.code(welch_source, language="python")

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("t-statistic",  f"{t_test_vibration['t_statistic']:.4f}")
    col_t2.metric("d.o.f. ν",     f"{t_test_vibration['degrees_of_freedom']:.1f}")
    col_t3.metric("p-value",       f"{t_test_vibration['p_value']:.4e}")
    col_t4.metric("H₀ rejected?", "Yes ⚠️" if t_test_vibration["reject_null"] else "No ✅")

    # ────────────────────────────────────────────
    # SECTION D: Weibull
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 4. Two-Parameter Weibull Reliability Function")

    st.markdown(
        '<div class="math-block"><h4>Weibull PDF, CDF, and Reliability Function</h4></div>',
        unsafe_allow_html=True,
    )
    st.latex(
        r"""
        f(t;\,\beta,\lambda) = \frac{\beta}{\lambda}
            \left(\frac{t}{\lambda}\right)^{\!\beta - 1}
            \exp\!\left[-\left(\frac{t}{\lambda}\right)^{\!\beta}\right]
        """
    )
    st.latex(
        r"""
        R(t) = \exp\!\left[-\left(\frac{t}{\lambda}\right)^{\!\beta}\right]
        \qquad
        h(t) = \frac{\beta}{\lambda}\left(\frac{t}{\lambda}\right)^{\!\beta - 1}
        """
    )
    st.latex(
        r"""
        \mathbb{E}[T] = \lambda \cdot \Gamma\!\left(1 + \frac{1}{\beta}\right)
        \qquad \text{(Mean Time To Failure)}
        """
    )
    st.latex(
        r"""
        \text{Log-likelihood:} \quad
        \ell(\beta,\lambda) = N\ln\beta - N\beta\ln\lambda
          + (\beta-1)\sum_{i}\ln x_i - \sum_{i}\left(\frac{x_i}{\lambda}\right)^{\!\beta}
        """
    )

    st.markdown('<div class="code-label">↳ Source: src/analytics.py — fit_weibull_distribution() + weibull_reliability()</div>', unsafe_allow_html=True)
    weibull_source = textwrap.dedent("""\
        def negative_log_likelihood(log_params):
            beta_val, lambda_val = np.exp(log_params[0]), np.exp(log_params[1])
            # ℓ(β,λ) = N·ln(β) − N·β·ln(λ) + (β−1)·Σln(xᵢ) − Σ(xᵢ/λ)^β
            log_likelihood = (
                N * np.log(beta_val)
                - N * beta_val * np.log(lambda_val)
                + (beta_val - 1.0) * np.sum(np.log(hi_positive))
                - np.sum((hi_positive / lambda_val) ** beta_val)
            )
            return -log_likelihood    # minimise negative ℓ

        result       = scipy.optimize.minimize(negative_log_likelihood, x0=..., method='L-BFGS-B')
        beta_shape   = np.exp(result.x[0])
        lambda_scale = np.exp(result.x[1])

        # Reliability Function  R(t) = exp[−(t/λ)^β]
        reliability = np.exp(-((time_points / lambda_scale) ** beta_shape))

        # Hazard Rate  h(t) = (β/λ)·(t/λ)^{β−1}
        hazard_rate = (beta_shape / lambda_scale) * (time_points / lambda_scale) ** (beta_shape - 1)
    """)
    st.code(weibull_source, language="python")

    col_w1, col_w2, col_w3 = st.columns(3)
    col_w1.metric("Shape β",       f"{beta_shape:.4f}", delta=f"{'Wear-out ▲' if beta_shape>1 else 'Random'}")
    col_w2.metric("Scale λ",       f"{lambda_scale:.4f}")
    col_w3.metric("MTTF E[T]",     f"{rul_result['mean_time_to_failure']:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — SURVIVAL & DIAGNOSTICS ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_diagnostics:

    # ── PCA 2D scatter: baseline vs current ──
    col_pca, col_hi = st.columns([1, 1])

    with col_pca:
        fig_pca = go.Figure()
        fig_pca.add_trace(
            go.Scatter(
                x=projected_baseline[:, 0],
                y=projected_baseline[:, 1],
                mode="markers",
                name="Healthy Baseline",
                marker=dict(color="#3fb950", size=7, opacity=0.7,
                            line=dict(color="#3fb950", width=1)),
            )
        )
        fig_pca.add_trace(
            go.Scatter(
                x=projected_current[:, 0],
                y=projected_current[:, 1],
                mode="markers",
                name=f"Current (δ={degradation_factor:.2f})",
                marker=dict(
                    color=projected_current[:, 0],
                    colorscale="Plasma",
                    size=9,
                    opacity=0.85,
                    line=dict(color="#f0f6fc", width=0.5),
                    showscale=True,
                    colorbar=dict(title="PC1", thickness=12),
                ),
            )
        )
        _dark_fig_layout(fig_pca, "PCA Projection: Health State Space (PC1 vs PC2)")
        fig_pca.update_xaxes(title_text="Principal Component 1")
        fig_pca.update_yaxes(title_text="Principal Component 2")
        fig_pca.update_layout(height=370, legend=dict(x=0.02, y=0.98))
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_hi:
        n_current_windows  = len(health_index_current)
        n_baseline_windows = len(health_index_baseline)
        window_time_current  = np.arange(n_current_windows)  * (window_length // 2) / sampling_rate_hz
        window_time_baseline = np.arange(n_baseline_windows) * (window_length // 2) / sampling_rate_hz

        fig_hi = go.Figure()
        fig_hi.add_trace(
            go.Scatter(
                x=window_time_baseline, y=health_index_baseline,
                name="Baseline HI", mode="lines",
                line=dict(color="#3fb950", width=1.5, dash="dot"),
            )
        )
        fig_hi.add_trace(
            go.Scatter(
                x=window_time_current, y=health_index_current,
                name="Current HI", mode="lines+markers",
                line=dict(color=COLOR_VIBRATION, width=2),
                marker=dict(size=5),
            )
        )
        # Threshold line at mean_hi
        threshold_line = float(np.mean(health_index_baseline)) + 2 * float(np.std(health_index_baseline))
        fig_hi.add_hline(
            y=threshold_line,
            line=dict(color="#ef4444", dash="dash", width=1.5),
            annotation_text="Alert threshold (μ+2σ)",
            annotation_font_color="#ef4444",
        )
        _dark_fig_layout(fig_hi, "Health Index Coefficient Over Time  ‖PC₁, PC₂‖₂")
        fig_hi.update_xaxes(title_text="Time (s)")
        fig_hi.update_yaxes(title_text="Health Index")
        fig_hi.update_layout(height=370)
        st.plotly_chart(fig_hi, use_container_width=True)

    st.markdown("---")

    # ── Weibull Reliability + Hazard Rate ──
    col_rel, col_haz = st.columns([1, 1])

    time_axis       = rul_result["time_axis"]
    reliability_curve = rul_result["reliability_curve"]
    hazard_curve    = rul_result["hazard_curve"]
    b10_life        = rul_result["b10_life"]

    with col_rel:
        fig_rel = go.Figure()
        fig_rel.add_trace(
            go.Scatter(
                x=time_axis,
                y=reliability_curve,
                name=f"R(t)  β={beta_shape:.2f}, λ={lambda_scale:.2f}",
                fill="tozeroy",
                fillcolor="rgba(6,182,212,0.10)",
                line=dict(color=COLOR_RELIABILITY, width=2.5),
            )
        )
        fig_rel.add_hline(
            y=failure_threshold,
            line=dict(color="#ef4444", dash="dash", width=1.5),
            annotation_text=f"Failure threshold R={failure_threshold:.2f}",
            annotation_font_color="#ef4444",
        )
        fig_rel.add_vline(
            x=b10_life,
            line=dict(color="#f97316", dash="dot", width=1.5),
            annotation_text=f"B{int(failure_threshold*100)} life",
            annotation_font_color="#f97316",
        )
        _dark_fig_layout(fig_rel, "Weibull Reliability Function  R(t) = exp[−(t/λ)^β]")
        fig_rel.update_xaxes(title_text="Normalised Time (t / λ units)")
        fig_rel.update_yaxes(title_text="Reliability  R(t)", range=[-0.05, 1.05])
        fig_rel.update_layout(height=340)
        st.plotly_chart(fig_rel, use_container_width=True)

    with col_haz:
        fig_haz = go.Figure()
        fig_haz.add_trace(
            go.Scatter(
                x=time_axis,
                y=hazard_curve,
                name=f"h(t)  β={beta_shape:.2f}",
                line=dict(color=COLOR_HAZARD, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.10)",
            )
        )
        _dark_fig_layout(fig_haz, "Weibull Hazard Rate  h(t) = (β/λ)·(t/λ)^{β−1}")
        fig_haz.update_xaxes(title_text="Normalised Time")
        fig_haz.update_yaxes(title_text="Instantaneous Failure Rate h(t)")
        fig_haz.update_layout(height=340)
        st.plotly_chart(fig_haz, use_container_width=True)

    # ── Hypothesis test p-value panel ──
    st.markdown("---")
    st.markdown("### Welch's T-Test — Structural Deviation Report")

    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        st.markdown("**Vibration Signal — Baseline vs Current**")
        tv = t_test_vibration
        report_df = pd.DataFrame(
            {
                "Parameter": [
                    "Baseline Mean (x̄₁)", "Current Mean (x̄₂)",
                    "Baseline Variance (s₁²)", "Current Variance (s₂²)",
                    "t-Statistic", "Degrees of Freedom (ν)",
                    "p-Value", "Reject H₀ (α=0.05)?",
                ],
                "Value": [
                    f"{tv['mean_baseline']:.5f}",  f"{tv['mean_current']:.5f}",
                    f"{tv['variance_baseline']:.5f}", f"{tv['variance_current']:.5f}",
                    f"{tv['t_statistic']:.4f}",    f"{tv['degrees_of_freedom']:.1f}",
                    f"{tv['p_value']:.4e}",        "YES ⚠️" if tv["reject_null"] else "NO ✅",
                ],
            }
        )
        st.dataframe(report_df, hide_index=True, use_container_width=True)

    with col_stat2:
        st.markdown("**Health Index — Baseline vs Current**")
        th = t_test_health
        report_hi = pd.DataFrame(
            {
                "Parameter": [
                    "Baseline Mean HI", "Current Mean HI",
                    "t-Statistic", "p-Value", "Reject H₀?",
                ],
                "Value": [
                    f"{th['mean_baseline']:.5f}", f"{th['mean_current']:.5f}",
                    f"{th['t_statistic']:.4f}",   f"{th['p_value']:.4e}",
                    "YES ⚠️" if th["reject_null"] else "NO ✅",
                ],
            }
        )
        st.dataframe(report_hi, hide_index=True, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        status_class = "status-danger" if tv["p_value"] < 0.001 else "status-warn" if tv["reject_null"] else "status-ok"
        st.markdown(
            f'<div class="{status_class}">{tv["deviation_detected"]}</div>',
            unsafe_allow_html=True,
        )

    # ── RUL Summary table ──
    st.markdown("---")
    st.markdown("### Remaining Useful Life Summary")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Weibull β (Shape)",    f"{beta_shape:.4f}", help="β > 1 → wear-out failure (increasing hazard rate)")
    col_s2.metric("Weibull λ (Scale)",    f"{lambda_scale:.4f}")
    col_s3.metric("MTTF  E[T]",           f"{rul_result['mean_time_to_failure']:.4f}")
    col_s4.metric(f"B{int(failure_threshold*100)} Life",  f"{b10_life:.4f}", help="Time at which survival probability = threshold")

    st.markdown(
        f"""
        <div style="background:#161b22;border:1px solid #30363d;border-left:4px solid #06b6d4;
                    border-radius:6px;padding:16px 20px;margin-top:12px;color:#c9d1d9;font-size:0.9rem;">
        <b style="color:#06b6d4;">Interpretation:</b>  At degradation factor δ = {degradation_factor:.2f},
        the fitted Weibull shape parameter β = {beta_shape:.3f}
        {"confirms a <b style='color:#ef4444;'>wear-out failure mode</b> (β > 1, monotonically increasing hazard rate)."
         if beta_shape > 1 else
         "indicates a near-random failure mode (β ≈ 1, constant hazard rate)."}
        The Mean Time to Failure is <b style='color:#06b6d4;'>{rul_result['mean_time_to_failure']:.3f}</b> normalised time units.
        The B{int(failure_threshold*100)} characteristic life
        (R(t) = {failure_threshold:.2f}) occurs at t = <b style='color:#f97316;'>{b10_life:.3f}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
