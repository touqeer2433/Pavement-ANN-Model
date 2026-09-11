"""
PAVE-ANN v2.0 . interactive design explorer.

    streamlit run app_streamlit.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predictor import PaveANNPredictor

st.set_page_config(page_title="PAVE-ANN", layout="wide")
st.markdown("""
<style>

    /* Main page */
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Main title */
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #12355b;
        margin-bottom: 0.1rem;
    }

    .main-subtitle {
        font-size: 0.95rem;
        color: #667085;
        margin-bottom: 1.5rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        color: #12355b;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e4e7ec;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #475467;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #12355b;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #102f4f;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Sidebar sliders */
    section[data-testid="stSidebar"] .stSlider {
        padding-bottom: 0.3rem;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid #e4e7ec;
        border-radius: 10px;
        background-color: white;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
    }

    /* Warning box */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def _load():
    return PaveANNPredictor.load(Path(__file__).parent)


P = _load()
LAYERS = ["WC", "ACB", "AggB", "SB", "Fill", "SG"]
PRETTY = {"WC": "Wearing course", "ACB": "AC base", "AggB": "Aggregate base",
          "SB": "Subbase", "Fill": "Subgrade fill", "SG": "Natural subgrade"}

st.markdown("""
<div style="
    background: white;
    padding: 18px 24px;
    border-radius: 12px;
    border: 1px solid #e4e7ec;
    margin-bottom: 20px;
">
    <div style="
        font-size: 2rem;
        font-weight: 750;
        color: #12355b;
    ">
        PAVE-ANN
        <span style="
            font-size: 1rem;
            font-weight: 500;
            color: #667085;
            margin-left: 12px;
        ">
            Artificial Neural Network Model for Flexible Pavement Analysis
        </span>
    </div>

    <div style="
        margin-top: 7px;
        font-size: 0.85rem;
        color: #667085;
    ">
        Data-driven mechanistic pavement response prediction
        &nbsp; | &nbsp;
        v""" + str(P.meta["version"]) + """
        &nbsp; | &nbsp;
        5-member ensemble
    </div>
</div>
""", unsafe_allow_html=True)

vals = P.template().iloc[0].to_dict()

with st.sidebar:
    st.header("Loading and environment")
    for f in ["P", "T", "S"]:
        if f in vals:
            vals[f] = st.slider(f"{f} [{P.meta['inputs']['units'][f]}]",
                                float(P.lo[P.features.index(f)]),
                                float(P.hi[P.features.index(f)]),
                                float(vals[f]))
    st.header("Layer properties")
    for L in LAYERS:
        with st.expander(PRETTY[L], expanded=(L in ("WC", "ACB"))):
            for pre in ["t", "E", "ν", "ρ", "σy", "φ", "K", "ψ"]:
                f = f"{pre}_{L}"
                if f in P.features:
                    i = P.features.index(f)
                    vals[f] = st.slider(f"{f} [{P.meta['inputs']['units'][f]}]",
                                        float(P.lo[i]), float(P.hi[i]),
                                        float(vals[f]), key=f)

df = pd.DataFrame([vals])
out = P.predict(df)

st.subheader("Predicted mechanistic responses")
cols = st.columns(len(P.primary))
for c, t in zip(cols, P.primary):
    mu, sd = float(out[t].iloc[0]), float(out[f"{t}_sd"].iloc[0])
    c.metric(P.meta["outputs"]["description"][t],
             f"{mu:.4g} {P.units[t]}", f"± {1.96*sd:.2g} (95%)")

life = P.design_life(out, vals.get("E_ACB", 3000.0))
c1, c2, c3 = st.columns(3)
c1.metric("Allowable repetitions, fatigue", f"{life['N_fatigue_allowable'].iloc[0]:.3g}")
c2.metric("Allowable repetitions, subgrade rutting", f"{life['N_rutting_allowable'].iloc[0]:.3g}")
c3.metric("Governing distress", str(life["governing_mode"].iloc[0]))
st.caption("Transfer functions use the published Asphalt Institute coefficients. "
           "Replace them with locally calibrated values before design use.")

if int(out["n_params_out_of_range"].iloc[0]) > 0:
    st.warning(f"{int(out['n_params_out_of_range'].iloc[0])} parameters sit outside the "
               f"training envelope, worst offender {out['worst_parameter'].iloc[0]}. "
               f"Predictive intervals widen accordingly but treat the result as indicative.")

st.subheader("Deflection basin")
basin = [t for t in P.targets if t.startswith("U2")]
if len(basin) > 1:
    st.line_chart(pd.DataFrame({"|U2| [mm]": [abs(float(out[t].iloc[0])) for t in basin]},
                               index=[t.replace("U2_", "") for t in basin]))

st.subheader("Thickness sweep")
sweep_f = st.selectbox("parameter", ["t_ACB", "t_WC", "t_AggB", "t_SB", "E_ACB", "P"])
sweep_t = st.selectbox("response", P.primary)
i = P.features.index(sweep_f)
grid = np.linspace(P.lo[i], P.hi[i], 40)
S = pd.concat([df] * len(grid), ignore_index=True)
S[sweep_f] = grid
res = P.predict(S)
st.line_chart(pd.DataFrame({sweep_t: res[sweep_t].to_numpy()}, index=np.round(grid, 3)))

with st.expander("Full prediction table"):
    st.dataframe(out.T)
