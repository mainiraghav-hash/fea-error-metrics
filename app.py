import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="FEA vs Experiment", layout="wide")

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_file(f):
    return pd.read_excel(f) if f.name.endswith((".xlsx", ".xls")) else pd.read_csv(f)


def sprague_geers(a, b):
    """
    Sprague-Geers error metrics.
    a : reference (experiment), b : prediction (FEA)
    Returns M (magnitude), P (phase), C (comprehensive).
    """
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
    fea_df      = pd.read_csv(SAMPLE_DIR / "sample_fea.csv")
    exp_dfs     = [pd.read_csv(SAMPLE_DIR / "sample_experiment_1.csv"),
                   pd.read_csv(SAMPLE_DIR / "sample_experiment_2.csv")]
    exp_names   = ["sample_experiment_1.csv", "sample_experiment_2.csv"]
    fea_x_col   = exp_x_col  = "displacement_mm"
    fea_y_col   = exp_y_col  = "force_N"
    show_error_band = True
    n_interp    = 500
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

- You can upload **multiple experimental files** to compare several test runs against one FEA result.
- Column names are mapped manually via the sidebar after upload.
"""
        )
    st.stop()

else:
    # ── Load & map columns (uploaded files path) ────────────────────────────────
    fea_df    = load_file(fea_file)
    exp_dfs   = [load_file(f) for f in exp_files]
    exp_names = [f.name for f in exp_files]

    with st.sidebar:
        st.header("2 · Map columns")

        st.subheader("FEA")
        fea_cols = list(fea_df.columns)
        fea_x_col = st.selectbox("X axis", fea_cols, key="fea_x")
        fea_y_col = st.selectbox("Y axis", fea_cols, index=min(1, len(fea_cols) - 1), key="fea_y")

        st.subheader("Experiment")
        exp_cols = list(exp_dfs[0].columns)
        exp_x_col = st.selectbox("X axis", exp_cols, key="exp_x")
        exp_y_col = st.selectbox("Y axis", exp_cols, index=min(1, len(exp_cols) - 1), key="exp_y")

        st.header("3 · Options")
        show_error_band = st.checkbox("Show ±1 std band (multi-experiment)", value=True)
        n_interp = st.slider("Interpolation points", 200, 2000, 500, 50)

fea_x = fea_df[fea_x_col].dropna().values
fea_y = fea_df[fea_y_col].dropna().values

# ── Build figure ───────────────────────────────────────────────────────────────

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Signal Overlay", "Residual  (FEA − Experiment)"),
    vertical_spacing=0.14,
    shared_xaxes=True,
)

# FEA trace
fig.add_trace(
    go.Scatter(
        x=fea_x, y=fea_y,
        name="FEA",
        line=dict(color="royalblue", width=2.5),
    ),
    row=1, col=1,
)

# ── Per-dataset metrics & residuals ───────────────────────────────────────────

metrics_rows = []
interp_ys = []   # collect for std-band

x_lo = max(fea_x.min(), min(e[exp_x_col].dropna().min() for e in exp_dfs))
x_hi = min(fea_x.max(), max(e[exp_x_col].dropna().max() for e in exp_dfs))
x_common = np.linspace(x_lo, x_hi, n_interp)
fea_interp = np.interp(x_common, fea_x, fea_y)

for i, (exp_df, name) in enumerate(zip(exp_dfs, exp_names)):
    color = COLORS[i % len(COLORS)]
    exp_x = exp_df[exp_x_col].dropna().values
    exp_y = exp_df[exp_y_col].dropna().values

    # Experiment trace
    fig.add_trace(
        go.Scatter(
            x=exp_x, y=exp_y,
            name=f"Exp · {name}",
            line=dict(color=color, width=1.5, dash="dot"),
        ),
        row=1, col=1,
    )

    # Interpolate experiment onto common grid
    exp_interp = np.interp(x_common, exp_x, exp_y)
    interp_ys.append(exp_interp)

    # Residual
    residual = fea_interp - exp_interp
    fig.add_trace(
        go.Scatter(
            x=x_common, y=residual,
            name=f"Residual · {name}",
            line=dict(color=color, width=1.5),
        ),
        row=2, col=1,
    )

    # Metrics
    M, P, C = sprague_geers(exp_interp, fea_interp)
    metrics_rows.append({
        "Dataset": name,
        "R²": round(r_squared(exp_interp, fea_interp), 4),
        "RMSE": round(rmse(exp_interp, fea_interp), 6),
        "Max |Error|": round(float(np.max(np.abs(residual))), 6),
        "S&G  M": round(M, 4),
        "S&G  P": round(P, 4),
        "S&G  C": round(C, 4),
    })

# Optional ±1 std band across all experiment curves
if show_error_band and len(interp_ys) > 1:
    stack = np.vstack(interp_ys)
    mean_y = stack.mean(axis=0)
    std_y = stack.std(axis=0)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_common, x_common[::-1]]),
            y=np.concatenate([mean_y + std_y, (mean_y - std_y)[::-1]]),
            fill="toself",
            fillcolor="rgba(255,165,0,0.15)",
            line=dict(color="rgba(255,165,0,0)"),
            name="Exp ±1 std",
            showlegend=True,
        ),
        row=1, col=1,
    )

# Zero line on residual plot
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

fig.update_layout(
    height=680,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=80),
)
fig.update_xaxes(title_text=fea_x_col, row=2, col=1)
fig.update_yaxes(title_text=fea_y_col, row=1, col=1)
fig.update_yaxes(title_text="Residual", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── Metrics table ──────────────────────────────────────────────────────────────

st.subheader("Error Metrics")

with st.expander("What do these mean?"):
    st.markdown(
        """
| Metric | Description |
|--------|-------------|
| **R²** | Coefficient of determination. 1.0 = perfect match. |
| **RMSE** | Root mean square error — same units as Y. |
| **Max \|Error\|** | Peak absolute residual. |
| **S&G M** | Sprague-Geers **magnitude** error. 0 = no amplitude bias. Positive → FEA over-predicts. |
| **S&G P** | Sprague-Geers **phase** error. 0 = perfectly in phase. |
| **S&G C** | Sprague-Geers **comprehensive** error = √(M²+P²). Overall goodness-of-fit. |
"""
    )

metrics_df = pd.DataFrame(metrics_rows)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

csv_out = metrics_df.to_csv(index=False)
st.download_button(
    "Download metrics CSV",
    csv_out,
    file_name="fea_vs_exp_metrics.csv",
    mime="text/csv",
)
