"""
PAVE-ANN v2.0
Professional Interactive Design Explorer

Run:
    streamlit run app_streamlit.py
"""

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st

from predictor import PaveANNPredictor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PAVE-ANN | Flexible Pavement Analysis",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    dedent("""
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #f5f7fa;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .pave-header {
        background: linear-gradient(135deg, #0d3557, #17628f);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(15, 53, 87, 0.16);
    }

    .pave-title {
        color: #ffffff;
        font-size: 2.15rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
    }

    .pave-subtitle {
        color: #e4eef6;
        font-size: 1rem;
        font-weight: 450;
        margin-top: 7px;
    }

    .pave-meta {
        color: #c5d9e8;
        font-size: 0.78rem;
        margin-top: 13px;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-heading {
        color: #123b5d;
        font-size: 1.28rem;
        font-weight: 750;
        margin-top: 12px;
        margin-bottom: 3px;
    }

    .section-description {
        color: #667085;
        font-size: 0.84rem;
        margin-bottom: 13px;
    }


    /* ========================================================
       OUTPUT CARDS
       ======================================================== */

    .output-card {
        background: #ffffff;
        border: 1px solid #dfe5eb;
        border-radius: 12px;
        padding: 18px 18px 16px 18px;
        min-height: 132px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.045);
    }

    .output-label {
        color: #475467;
        font-size: 0.78rem;
        font-weight: 650;
        line-height: 1.35;
        min-height: 34px;
    }

    .output-value {
        color: #123b5d;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 7px;
        white-space: nowrap;
    }

    .output-uncertainty {
        color: #667085;
        font-size: 0.76rem;
        margin-top: 4px;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-good {
        background: #ecfdf3;
        border: 1px solid #abefc6;
        color: #067647;
        border-radius: 9px;
        padding: 10px 14px;
        font-size: 0.82rem;
        font-weight: 650;
        margin-top: 12px;
        margin-bottom: 15px;
    }

    .status-warning {
        background: #fffaeb;
        border: 1px solid #fedf89;
        color: #b54708;
        border-radius: 9px;
        padding: 10px 14px;
        font-size: 0.82rem;
        font-weight: 650;
        margin-top: 12px;
        margin-bottom: 15px;
    }


    /* ========================================================
       INFORMATION CARDS
       ======================================================== */

    .info-card {
        background: #ffffff;
        border: 1px solid #dfe5eb;
        border-radius: 12px;
        padding: 17px 19px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
        margin-bottom: 12px;
    }

    .info-title {
        color: #123b5d;
        font-size: 0.95rem;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .info-row {
        color: #475467;
        font-size: 0.80rem;
        line-height: 1.65;
    }

    .info-value {
        color: #123b5d;
        font-weight: 650;
    }


    /* ========================================================
       PAVEMENT LAYER SCHEMATIC
       ======================================================== */

    .pavement-box {
        border: 1px solid #dfe5eb;
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
    }

    .layer {
        padding: 12px 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.5);
        font-size: 0.80rem;
        font-weight: 650;
    }

    .layer-name {
        color: #1d2939;
    }

    .layer-value {
        color: #344054;
        font-weight: 700;
    }

    .layer-wc {
        background: #d8e6f0;
    }

    .layer-acb {
        background: #dfe8d7;
    }

    .layer-aggb {
        background: #eadfcf;
    }

    .layer-sb {
        background: #e6dfd5;
    }

    .layer-fill {
        background: #d8d0c4;
    }

    .layer-sg {
        background: #bfa98e;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #102f4f;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.18);
    }

    .sidebar-title {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .sidebar-description {
        color: #c8d8e5;
        font-size: 0.76rem;
        line-height: 1.45;
        margin-bottom: 18px;
    }

    .sidebar-section {
        color: #ffffff;
        font-size: 0.96rem;
        font-weight: 750;
        border-bottom: 1px solid rgba(255,255,255,0.20);
        padding-bottom: 7px;
        margin-bottom: 9px;
    }


    /* ========================================================
       SIDEBAR EXPANDERS
       ======================================================== */

    section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        background-color: #173f61;
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 9px;
        margin-bottom: 7px;
    }

    section[data-testid="stSidebar"]
    div[data-testid="stExpander"] summary {
        color: #ffffff;
    }

    section[data-testid="stSidebar"]
    div[data-testid="stExpander"] summary p {
        color: #ffffff !important;
        font-weight: 650;
    }


    /* ========================================================
       SLIDERS
       ======================================================== */

    section[data-testid="stSidebar"] .stSlider label {
        color: #eaf2f8 !important;
        font-size: 0.74rem;
    }

    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #bcd0df;
    }


    /* ========================================================
       SELECTBOX
       ======================================================== */

    div[data-baseweb="select"] {
        border-radius: 8px;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 700;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 9px;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .dashboard-footer {
        border-top: 1px solid #dfe5eb;
        margin-top: 30px;
        padding-top: 13px;
        color: #98a2b3;
        font-size: 0.70rem;
        text-align: center;
    }

    </style>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# LOAD PAVE-ANN
# ============================================================

@st.cache_resource
def _load():
    return PaveANNPredictor.load(Path(__file__).parent)


P = _load()


# ============================================================
# DEFINITIONS
# ============================================================

LAYERS = [
    "WC",
    "ACB",
    "AggB",
    "SB",
    "Fill",
    "SG",
]

PRETTY = {
    "WC": "Wearing Course",
    "ACB": "AC Base",
    "AggB": "Aggregate Base",
    "SB": "Subbase",
    "Fill": "Subgrade Fill",
    "SG": "Natural Subgrade",
}


# ============================================================
# HEADER
# ============================================================

header_html = f"""
<div class="pave-header">
<div class="pave-title">PAVE-ANN</div>
<div class="pave-subtitle">
Artificial Neural Network Model for Flexible Pavement Analysis
</div>
<div class="pave-meta">
Data-driven mechanistic pavement response prediction
&nbsp; | &nbsp;
Version {P.meta["version"]}
&nbsp; | &nbsp;
{P.meta["n_ensemble_members"]}-member ensemble
&nbsp; | &nbsp;
Fingerprint {P.meta["fingerprint"]}
</div>
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)


# ============================================================
# DEFAULT INPUT TEMPLATE
# ============================================================

vals = P.template().iloc[0].to_dict()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">INPUT PARAMETERS</div>
        <div class="sidebar-description">
        Define loading, environmental and pavement-layer
        properties for response prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">Loading & Environment</div>',
        unsafe_allow_html=True,
    )

    for f in ["P", "T", "S"]:

        if f in vals:

            vals[f] = st.slider(
                f"{f} [{P.meta['inputs']['units'][f]}]",
                float(P.lo[P.features.index(f)]),
                float(P.hi[P.features.index(f)]),
                float(vals[f]),
            )

    st.markdown("---")

    # --------------------------------------------------------
    # Layer properties
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">Pavement Layer Properties</div>',
        unsafe_allow_html=True,
    )

    for L in LAYERS:

        with st.expander(
            PRETTY[L],
            expanded=(L in ("WC", "ACB")),
        ):

            for pre in [
                "t",
                "E",
                "ν",
                "ρ",
                "σy",
                "φ",
                "K",
                "ψ",
            ]:

                f = f"{pre}_{L}"

                if f in P.features:

                    i = P.features.index(f)

                    vals[f] = st.slider(
                        f"{f} [{P.meta['inputs']['units'][f]}]",
                        float(P.lo[i]),
                        float(P.hi[i]),
                        float(vals[f]),
                        key=f,
                    )


# ============================================================
# PREDICTION
# ============================================================

df = pd.DataFrame([vals])

out = P.predict(df)


# ============================================================
# PREDICTED MECHANISTIC RESPONSES
# ============================================================

st.markdown(
    '<div class="section-heading">Prediction Results</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'PAVE-ANN ensemble predictions with uncertainty estimates.'
    '</div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Four output cards
# ------------------------------------------------------------

output_columns = st.columns(4)

for c, t in zip(output_columns, P.primary):

    mu = float(out[t].iloc[0])
    sd = float(out[f"{t}_sd"].iloc[0])

    description = P.meta["outputs"]["description"][t]

    unit = P.units[t]

    c.markdown(
        f"""
        <div class="output-card">
        <div class="output-label">{description}</div>
        <div class="output-value">{mu:.4g} {unit}</div>
        <div class="output-uncertainty">
        95% uncertainty: ± {1.96 * sd:.3g} {unit}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL STATUS
# ============================================================

n_out = int(out["n_params_out_of_range"].iloc[0])

if n_out > 0:

    worst = str(out["worst_parameter"].iloc[0])

    st.markdown(
        f"""
        <div class="status-warning">
        ⚠ Extrapolation warning:
        {n_out} parameter(s) are outside the training envelope.
        Worst offender: {worst}.
        Predictions should be treated as indicative.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="status-good">
        ✓ Model status: All input parameters are within the
        validated training envelope.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DESIGN PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-heading">Design Performance</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Estimated allowable repetitions based on the implemented '
    'transfer functions.'
    '</div>',
    unsafe_allow_html=True,
)


life = P.design_life(
    out,
    vals.get("E_ACB", 3000.0),
)

life_c1, life_c2, life_c3 = st.columns(3)

with life_c1:

    st.metric(
        "Fatigue Allowable Repetitions",
        f"{life['N_fatigue_allowable'].iloc[0]:.3g}",
    )

with life_c2:

    st.metric(
        "Rutting Allowable Repetitions",
        f"{life['N_rutting_allowable'].iloc[0]:.3g}",
    )

with life_c3:

    st.metric(
        "Governing Distress",
        str(life["governing_mode"].iloc[0]),
    )


st.caption(
    "Transfer functions use the published Asphalt Institute coefficients. "
    "Replace them with locally calibrated values before design use."
)


# ============================================================
# ANALYSIS AREA
# ============================================================

left_col, right_col = st.columns([1.35, 1])


# ============================================================
# DEFLECTION RESPONSE PROFILE
# ============================================================

with left_col:

    st.markdown(
        '<div class="section-heading">Deflection Response Profile</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Predicted surface-response values at the model evaluation locations.'
        '</div>',
        unsafe_allow_html=True,
    )

    basin = [
        t for t in P.targets
        if t.startswith("U2")
    ]

    if len(basin) > 1:

        basin_values = [
            abs(float(out[t].iloc[0]))
            for t in basin
        ]

        basin_labels = [
            t.replace("U2_", "")
            for t in basin
        ]

        basin_df = pd.DataFrame(
            {
                "Deflection |U2| [mm]": basin_values
            },
            index=basin_labels,
        )

        st.line_chart(
            basin_df,
            height=330,
        )

    else:

        st.info(
            "Deflection response targets are not available "
            "in the current model configuration."
        )


# ============================================================
# PAVEMENT STRUCTURE
# ============================================================

with right_col:

    st.markdown(
        '<div class="section-heading">Pavement Structure</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Current pavement-layer configuration.'
        '</div>',
        unsafe_allow_html=True,
    )

    def layer_value(layer):

        key = f"t_{layer}"

        if key in vals:

            unit = P.meta["inputs"]["units"].get(
                key,
                "",
            )

            return f"{vals[key]:.3g} {unit}"

        return "—"


    pavement_html = f"""
    <div class="pavement-box">

    <div class="layer layer-wc">
        <span class="layer-name">Asphalt Concrete / Wearing Course</span>
        <span class="layer-value">{layer_value("WC")}</span>
    </div>

    <div class="layer layer-acb">
        <span class="layer-name">AC Base</span>
        <span class="layer-value">{layer_value("ACB")}</span>
    </div>

    <div class="layer layer-aggb">
        <span class="layer-name">Aggregate Base</span>
        <span class="layer-value">{layer_value("AggB")}</span>
    </div>

    <div class="layer layer-sb">
        <span class="layer-name">Subbase</span>
        <span class="layer-value">{layer_value("SB")}</span>
    </div>

    <div class="layer layer-fill">
        <span class="layer-name">Subgrade Fill</span>
        <span class="layer-value">{layer_value("Fill")}</span>
    </div>

    <div class="layer layer-sg">
        <span class="layer-name">Natural Subgrade</span>
        <span class="layer-value">{layer_value("SG")}</span>
    </div>

    </div>
    """

    st.markdown(
        pavement_html,
        unsafe_allow_html=True,
    )


# ============================================================
# PARAMETRIC SWEEP
# ============================================================

st.markdown(
    '<div class="section-heading">Parametric Analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Evaluate the response sensitivity to a selected input parameter '
    'while keeping the remaining inputs fixed.'
    '</div>',
    unsafe_allow_html=True,
)


sweep_c1, sweep_c2 = st.columns(2)

with sweep_c1:

    sweep_f = st.selectbox(
        "Input parameter",
        [
            "t_ACB",
            "t_WC",
            "t_AggB",
            "t_SB",
            "E_ACB",
            "P",
        ],
    )


with sweep_c2:

    sweep_t = st.selectbox(
        "Response variable",
        P.primary,
    )


i = P.features.index(sweep_f)

grid = np.linspace(
    P.lo[i],
    P.hi[i],
    40,
)

S = pd.concat(
    [df] * len(grid),
    ignore_index=True,
)

S[sweep_f] = grid

res = P.predict(S)

sweep_df = pd.DataFrame(
    {
        sweep_t: res[sweep_t].to_numpy()
    },
    index=np.round(grid, 3),
)

st.line_chart(
    sweep_df,
    height=320,
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-heading">Model Information</div>',
    unsafe_allow_html=True,
)

info_c1, info_c2 = st.columns(2)


with info_c1:

    st.markdown(
        f"""
        <div class="info-card">
        <div class="info-title">PAVE-ANN Model</div>

        <div class="info-row">
        Model version:
        <span class="info-value">{P.meta["version"]}</span>
        </div>

        <div class="info-row">
        Ensemble members:
        <span class="info-value">{P.meta["n_ensemble_members"]}</span>
        </div>

        <div class="info-row">
        Input variables:
        <span class="info-value">{len(P.features)}</span>
        </div>

        <div class="info-row">
        Primary outputs:
        <span class="info-value">{len(P.primary)}</span>
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with info_c2:

    st.markdown(
        f"""
        <div class="info-card">
        <div class="info-title">Deployment Information</div>

        <div class="info-row">
        Model format:
        <span class="info-value">TorchScript</span>
        </div>

        <div class="info-row">
        Uncertainty:
        <span class="info-value">Ensemble-based</span>
        </div>

        <div class="info-row">
        Range checking:
        <span class="info-value">Training-envelope validation</span>
        </div>

        <div class="info-row">
        Fingerprint:
        <span class="info-value">{P.meta["fingerprint"]}</span>
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FULL PREDICTION TABLE
# ============================================================

with st.expander("Full Prediction Table"):

    st.dataframe(
        out.T,
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="dashboard-footer">
    PAVE-ANN v2.0
    &nbsp; | &nbsp;
    Interactive Flexible Pavement Response Prediction
    &nbsp; | &nbsp;
    Developed for Pavement Engineering Research
    </div>
    """,
    unsafe_allow_html=True,
)
