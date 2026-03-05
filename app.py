import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

st.set_page_config(page_title="FEA vs Experiment", layout="wide")

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_file(f):
    return pd.read_excel(f) if f.name.endswith((".xlsx", ".xls")) else pd.read_csv(f)


def cfc_filter(data, fs, cfc_class):
    """SAE J211 zero-phase 4-pole Butterworth (sosfiltfilt = forward+backward)."""
    fc_map = {60: 100, 180: 300, 600: 1000, 1000: 1650}
    fc = fc_map[cfc_class]
    if fs <= 2.0 * fc:          # Nyquist check
        st.warning(f"CFC {cfc_class}: sample rate {fs:.0f} Hz too low to filter. Skipped.")
        return data
    sos = butter(2, fc, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, data)


def sprague_geers(a, b):
    """a = experiment (reference), b = FEA (prediction)."""
    sum_a2 = np.sum(a ** 2)
    sum_b2 = np.sum(b ** 2)
    sum_ab = np.sum(a * b)
    M = np.sqrt(sum_b2 / sum_a2) - 1
    cos_theta = np.clip(sum_ab / np.sqrt(sum_a2 * sum_b2), -1.0, 1.0)
    P = (1.0 / np.pi) * np.arccos(cos_theta)
    C = np.sqrt(M ** 2 + P ** 2)
    return M, P, C


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def r_squared(a, b):
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def add_peak_annotation(fig, x, y, color, row):
    idx = int(np.argmax(np.abs(y)))
    fig.add_annotation(
        x=x[idx], y=y[idx],
        text=f"<b>{y[idx]:.3g}</b><br>@ {x[idx]:.4g}",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor=color, font=dict(size=10, color=color),
        bgcolor="white", bordercolor=color, borderwidth=1,
        ax=35, ay=-35,
        row=row, col=1,
    )


COLORS = [
    "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF",
]

# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("FEA vs Experiment Dashboard")
st.caption("Upload your FEA results and experimental data, pick axes, and compare.")

SAMPLE_DIR = Path(__file__).parent

with st.sidebar:
    st.header("1 · Upload data")
    fea_file = st.file_uploader("FEA results", type=["csv", "xlsx", "xls"])
    exp_files = st.file_uploader(
        "Experimental data (one or more)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    st.divider()
    use_sample = st.button("▶ Load sample data", use_container_width=True,
                           help="Loads the bundled tensile-coupon sample CSVs")

# ── Guard ──────────────────────────────────────────────────────────────────────

if use_sample:
    fea_df    = pd.read_csv(SAMPLE_DIR / "sample_fea.csv")
    exp_dfs   = [pd.read_csv(SAMPLE_DIR / "sample_experiment_1.csv"),
                 pd.read_csv(SAMPLE_DIR / "sample_experiment_2.csv")]
    exp_names = ["sample_experiment_1.csv", "sample_experiment_2.csv"]
    fea_x_col = exp_x_col = "displacement_mm"
    fea_y_col = exp_y_col = "force_N"

elif not fea_file or not exp_files:
    st.info("Upload at least one FEA file and one experimental file to get started.")
    with st.expander("Expected data format"):
        st.markdown(
            """
Both files should be **CSV or Excel** with at least two numeric columns, e.g.:

| time | displacement |
|------|-------------|
| 0.0  | 0.000       |
| 0.1  | 0.023       |

- Upload **multiple experimental files** to compare several test runs against one FEA result.
- Column names are mapped via the sidebar after upload.
"""
        )
    st.stop()

else:
    fea_df    = load_file(fea_file)
    exp_dfs   = [load_file(f) for f in exp_files]
    exp_names = [f.name for f in exp_files]

    with st.sidebar:
        st.header("2 · Map columns")

        st.subheader("FEA")
        fea_cols  = list(fea_df.columns)
        fea_x_col = st.selectbox("X axis", fea_cols, key="fea_x")
        fea_y_col = st.selectbox("Y axis", fea_cols, index=min(1, len(fea_cols) - 1), key="fea_y")

        st.subheader("Experiment")
        exp_cols  = list(exp_dfs[0].columns)
        exp_x_col = st.selectbox("X axis", exp_cols, key="exp_x")
        exp_y_col = st.selectbox("Y axis", exp_cols, index=min(1, len(exp_cols) - 1), key="exp_y")

# ── Derived base arrays ────────────────────────────────────────────────────────

fea_x = fea_df[fea_x_col].dropna().values
fea_y = fea_df[fea_y_col].dropna().values

x_lo_full = float(max(fea_x.min(), min(e[exp_x_col].dropna().min() for e in exp_dfs)))
x_hi_full = float(min(fea_x.max(), max(e[exp_x_col].dropna().max() for e in exp_dfs)))

# ── Options sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("3 · Options")

    n_interp = st.slider("Interpolation points", 200, 2000, 500, 50)

    st.subheader("Std deviation band")
    show_std_band = st.checkbox("Show std band", value=True)
    std_n = st.select_slider("Band width", options=[1, 2, 3], value=1,
                             format_func=lambda v: f"±{v}σ",
                             disabled=not show_std_band)

    st.subheader("Corridor (min/max envelope)")
    show_corridor = st.checkbox("Show corridor", value=True)

    st.subheader("Peak annotations")
    show_peaks = st.checkbox("Show peak annotations", value=True)

    st.subheader("CFC filter (SAE J211)")
    cfc_choice = st.selectbox(
        "Apply to experiment data",
        ["None", "CFC 60", "CFC 180", "CFC 600", "CFC 1000"],
    )

    st.subheader("Analysis window")
    st.caption("Metrics are computed only within this X range.")
    w_lo = st.number_input("Start", value=x_lo_full,
                           min_value=x_lo_full, max_value=x_hi_full, format="%.5g")
    w_hi = st.number_input("End",   value=x_hi_full,
                           min_value=x_lo_full, max_value=x_hi_full, format="%.5g")

# ── Common interpolation grid ──────────────────────────────────────────────────

x_common   = np.linspace(x_lo_full, x_hi_full, n_interp)
fea_interp = np.interp(x_common, fea_x, fea_y)
win_mask   = (x_common >= w_lo) & (x_common <= w_hi)

cfc_int_map = {"CFC 60": 60, "CFC 180": 180, "CFC 600": 600, "CFC 1000": 1000}
apply_cfc   = cfc_choice != "None"

# ── Build figure ───────────────────────────────────────────────────────────────

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Signal Overlay", "Residual  (FEA − Experiment)"),
    vertical_spacing=0.14,
    shared_xaxes=True,
    specs=[[{}], [{"secondary_y": True}]],
)

# FEA trace
fig.add_trace(
    go.Scatter(x=fea_x, y=fea_y, name="FEA",
               line=dict(color="royalblue", width=2.5)),
    row=1, col=1,
)
if show_peaks:
    add_peak_annotation(fig, fea_x, fea_y, "royalblue", row=1)

# ── Per-dataset loop ───────────────────────────────────────────────────────────

metrics_rows = []
interp_ys    = []

for i, (exp_df, name) in enumerate(zip(exp_dfs, exp_names)):
    color = COLORS[i % len(COLORS)]
    exp_x = exp_df[exp_x_col].dropna().values
    exp_y = exp_df[exp_y_col].dropna().values

    # ① CFC filter
    if apply_cfc:
        dx = np.mean(np.diff(exp_x))
        fs = 1.0 / dx if dx > 0 else 1000.0
        exp_y = cfc_filter(exp_y, fs, cfc_int_map[cfc_choice])

    # Experiment trace
    fig.add_trace(
        go.Scatter(x=exp_x, y=exp_y, name=f"Exp · {name}",
                   line=dict(color=color, width=1.5, dash="dot")),
        row=1, col=1,
    )

    # ② Peak annotation on experiment
    if show_peaks:
        add_peak_annotation(fig, exp_x, exp_y, color, row=1)

    # Interpolate onto common grid
    exp_interp = np.interp(x_common, exp_x, exp_y)
    interp_ys.append(exp_interp)

    # Residual (absolute) — primary Y
    residual = fea_interp - exp_interp
    fig.add_trace(
        go.Scatter(x=x_common, y=residual, name=f"Residual · {name}",
                   line=dict(color=color, width=1.5)),
        row=2, col=1, secondary_y=False,
    )

    # ③ % error — secondary Y (hidden by default, toggle via legend)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_err = np.where(
            np.abs(exp_interp) > 1e-10,
            100.0 * residual / np.abs(exp_interp),
            np.nan,
        )
    fig.add_trace(
        go.Scatter(x=x_common, y=pct_err, name=f"% error · {name}",
                   line=dict(color=color, width=1, dash="dot"),
                   visible="legendonly"),
        row=2, col=1, secondary_y=True,
    )

    # Windowed metrics
    a_w = exp_interp[win_mask]
    b_w = fea_interp[win_mask]
    if len(a_w) < 2:
        continue
    M, P, C = sprague_geers(a_w, b_w)
    metrics_rows.append({
        "Dataset":      name,
        "R²":           round(r_squared(a_w, b_w), 4),
        "RMSE":         round(rmse(a_w, b_w), 6),
        "Max |Error|":  round(float(np.max(np.abs(b_w - a_w))), 6),
        "S&G  M":       round(M, 4),
        "S&G  P":       round(P, 4),
        "S&G  C":       round(C, 4),
    })

# ── ④ Std band & ① corridor ────────────────────────────────────────────────────

if len(interp_ys) > 1:
    stack  = np.vstack(interp_ys)
    mean_y = stack.mean(axis=0)
    std_y  = stack.std(axis=0)

    if show_std_band:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_common, x_common[::-1]]),
                y=np.concatenate([mean_y + std_n * std_y,
                                  (mean_y - std_n * std_y)[::-1]]),
                fill="toself",
                fillcolor="rgba(255,165,0,0.15)",
                line=dict(color="rgba(255,165,0,0)"),
                name=f"Exp ±{std_n}σ",
            ),
            row=1, col=1,
        )

    # ① Corridor
    if show_corridor:
        min_y = stack.min(axis=0)
        max_y = stack.max(axis=0)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_common, x_common[::-1]]),
                y=np.concatenate([max_y, min_y[::-1]]),
                fill="toself",
                fillcolor="rgba(100,180,100,0.12)",
                line=dict(color="rgba(100,180,100,0.7)", width=1, dash="dash"),
                name="Corridor (min/max)",
            ),
            row=1, col=1,
        )

# ── ⑤ Analysis window markers ─────────────────────────────────────────────────

window_active = (w_lo > x_lo_full) or (w_hi < x_hi_full)
if window_active:
    for xval in [w_lo, w_hi]:
        fig.add_vline(x=xval, line_dash="longdash", line_color="dimgray",
                      line_width=1.2)

# Zero reference on residual plot
fig.add_hline(y=0, line_dash="dash", line_color="lightgray", row=2, col=1)

fig.update_layout(
    height=740,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=80),
)
fig.update_xaxes(title_text=fea_x_col, row=2, col=1)
fig.update_yaxes(title_text=fea_y_col, row=1, col=1)
fig.update_yaxes(title_text="Residual", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="% Error", row=2, col=1, secondary_y=True,
                 showgrid=False, zeroline=False)

st.plotly_chart(fig, use_container_width=True)

# ── Metrics table ──────────────────────────────────────────────────────────────

win_label = (f"full range" if not window_active
             else f"X ∈ [{w_lo:.4g}, {w_hi:.4g}]")
st.subheader(f"Error Metrics — {win_label}")

with st.expander("What do these mean?"):
    st.markdown(
        """
| Metric | Description |
|--------|-------------|
| **R²** | Coefficient of determination. 1.0 = perfect match. |
| **RMSE** | Root mean square error — same units as Y. |
| **Max \|Error\|** | Peak absolute residual within the analysis window. |
| **S&G M** | Sprague-Geers **magnitude** error. 0 = no amplitude bias. Positive → FEA over-predicts. |
| **S&G P** | Sprague-Geers **phase** error. 0 = perfectly in phase. |
| **S&G C** | Sprague-Geers **comprehensive** error = √(M²+P²). Combined goodness-of-fit. |

All metrics are computed within the **analysis window** set in the sidebar.
Toggle **% error** traces in the legend to overlay percentage error on the residual plot.
"""
    )

metrics_df = pd.DataFrame(metrics_rows)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

csv_out = metrics_df.to_csv(index=False)
st.download_button("Download metrics CSV", csv_out,
                   file_name="fea_vs_exp_metrics.csv", mime="text/csv")
