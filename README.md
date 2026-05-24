# ⚙️ Predictive Maintenance & Fault Diagnostics Pipeline

A production-grade, modular Python application demonstrating industrial signal processing, statistical modelling, and reliability engineering — built for deployment via Streamlit.

# Streamlit app: https://predictive-maintenance-pipeline-fajg2in8mkjttxslwtuwzu.streamlit.app/


## Mathematical Core

| Module | Technique | Implementation |
|--------|-----------|----------------|
| `data/generator.py` | Physics-based sensor simulation | Compound sinusoidal vibration, logarithmic thermal drift, stochastic voltage spikes |
| `src/feature_eng.py` | DFT / FFT spectral analysis | Hann-windowed FFT, Spectral Peak Frequency, Spectral Entropy (Shannon) |
| `src/feature_eng.py` | PCA via SVD (no sklearn) | `numpy.linalg.svd`, eigenvalue decomposition, 2-D Health Index projection |
| `src/analytics.py` | Welch's T-test | Manual Welch-Satterthwaite d.o.f., exact p-value via `scipy.stats.t.sf` |
| `src/analytics.py` | Weibull Reliability | MLE via L-BFGS-B, R(t) = exp[−(t/λ)^β], B10 life, MTTF |

## Repository Architecture

```
predictive_maintenance_app/
│
├── data/
│   └── generator.py       # Physics-based multi-sensor simulator
│
├── src/
│   ├── ingestion.py       # Sliding-window signal segmentation
│   ├── feature_eng.py     # FFT features + SVD/PCA Health Index
│   └── analytics.py       # Welch T-test + Weibull RUL estimation
│
├── app.py                 # Streamlit UI (3 tabs, LaTeX math, live plots)
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```


## Equations Implemented

**Welch's T-test statistic:**

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$

**SVD decomposition:**

$$\tilde{X} = U \Sigma V^T \qquad X_{\text{projected}} = \tilde{X} \cdot V[:, :2]$$

**Weibull Reliability Function:**

$$R(t) = \exp\!\left[-\left(\frac{t}{\lambda}\right)^{\!\beta}\right]$$

## Dependencies

| Library | Version | Role |
|---------|---------|------|
| numpy | ≥ 1.24 | Array math, FFT, SVD |
| scipy | ≥ 1.11 | T-distribution, Weibull MLE optimisation |
| pandas | ≥ 2.0 | Tabular feature assembly |
| streamlit | ≥ 1.35 | Interactive web dashboard |
| plotly | ≥ 5.20 | Dark-mode interactive visualisations |
