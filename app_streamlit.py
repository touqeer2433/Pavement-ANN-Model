"""
PAVE-ANN v2.0 - Professional Interactive Design Explorer

Run locally:
    streamlit run app_streamlit.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predictor import PaveANNPredictor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PAVE-ANN | Flexible Pavement Analysis",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL DASHBOARD STYLING
# ============================================================

st.markdown(
    """
<style>

    /* ======================================================
       GLOBAL PAGE
       ====================================================== */

    .main {
        background-color: #f5f7fa;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }


    /* ======================================================
       HEADER
       ====================================================== */

    .pave-header {
        background: linear-gradient(
            135deg,
            #0f3557 0%,
            #174d78 100%
        );
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(15, 53, 87, 0.16);
    }

    .pave-title {
        color: white;
        font-size: 2.05rem;
        font-weight: 750;
        letter-spacing: 0.2px;
        margin: 0;
    }

    .pave-subtitle {
        color: #dceaf5;
        font-size: 0.98rem;
        margin-top: 5px;
        margin-bottom: 0;
    }

    .pave-meta {
        color: #b9d0e2;
        font-size: 0.78rem;
        margin-top: 12px;
    }


    /* ======================================================
       SECTION HEADINGS
       ====================================================== */

    .section-heading {
        color: #123b5d;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .section-description {
        color: #667085;
        font-size: 0.87rem;
        margin-top: -5px;
        margin-bottom: 14px;
    }


    /* ======================================================
       PREDICTION METRIC CARDS
       ====================================================== */

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #dfe5eb;
        border-radius: 12px;
        padding: 17px 18px;
        min-height: 125px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #475467;
        font-size: 0.84rem;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #123b5d;
        font-weight: 700;
    }

    div[data-testid="stMetricDelta"] {
        color: #667085;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #102f4f;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #c4d4e2;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.18);
    }


    /* Sidebar headers */

    .sidebar-main-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: white;
        margin-bottom: 4px;
    }

    .sidebar-description {
        font-size: 0.78rem;
        color: #c6d7e5;
        line-height: 1.4;
        margin-bottom: 18px;
    }

    .sidebar-section {
        font-size: 0.98rem;
        font-weight: 700;
        color: white;
        padding-top: 8px;
        padding-bottom: 5px;
        border-bottom: 1px solid rgba(255,255,255,0.18);
        margin-bottom: 8px;
    }


    /* Sidebar sliders */

    section[data-testid="stSidebar"] .stSlider {
        padding-bottom: 5px;
    }

    section[data-testid="stSidebar"] [data-testid="stSlider"] label {
        font-size: 0.76rem;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    div[data-testid="stExpander"] {
        border: 1px solid #dfe5eb;
        border-radius: 10px;
        background-color: white;
        margin-bottom: 8px;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 650;
        border: 1px solid #cfd8e3;
    }


    /* ======================================================
       ALERTS
       ====================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* ======================================================
       INFORMATION CARDS
       ====================================================== */

    .info-card {
        background: white;
        border: 1px solid #dfe5eb;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
    }

    .info-card-title {
        color: #123b5d;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .info-card-text {
        color: #667085;
        font-size: 0.82rem;
        line-height: 1.45;
    }


    /* ======================================================
       STATUS BADGES
       ====================================================== */

    .status-good {
        background: #ecfdf3;
        border: 1px solid #abefc6;
        color: #067647;
        padding: 10px 14px;
        border-radius: 9px;
        font-weight: 650;
        font-size: 0.85rem;
    }

    .status-warning {
        background: #fffaeb;
        border: 1px solid #fedf89;
        color: #b54708;
        padding: 10px 14px;
        border-radius: 9px;
        font-weight: 650;
        font-size: 0.85rem;
    }


    /* ======================================================
       TABLE
       ====================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .dashboard-footer {
        margin-top: 28px;
        padding-top: 14px;
        border-top: 1px solid #dfe5eb;
        color: #98a2b3;
        font-size: 0.72rem;
        text-align: center;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def _load():
    return PaveANNPredictor.load(Path(__file__).parent)


P = _load()


# ============================================================
# MODEL / INPUT DEFINITIONS
# ============================================================

LAYERS = ["WC", "ACB", "AggB", "SB", "Fill", "SG"]

PRETTY = {
    "WC": "Wearing Course",
    "ACB": "AC Base",
    "AggB": "Aggregate Base",
    "SB": "Subbase",
    "Fill": "Subgrade Fill",
    "SG": "Natural Subgrade",
}


# ============================================================
# PROFESSIONAL HEADER
# ============================================================

st.markdown(
    f"""
<div class="pave-header">

    <div class="pave-title">
        PAVE-ANN
    </div>

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
""",
    unsafe_allow_html=True,
)


# ============================================================
# DEFAULT INPUT TEMPLATE
# ============================================================

vals = P.template().iloc[0].to_dict()


# ============================================================
# SIDEBAR - INPUT PARAMETERS
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-main-title">
            INPUT PARAMETERS
        </div>

        <div class="sidebar-description">
            Define loading, environmental and pavement-layer
            properties for mechanistic response prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Loading and Environment
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
    # Pavement Layer Properties
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
# CREATE INPUT DATAFRAME
# ============================================================

df = pd.DataFrame([vals])


# ============================================================
# MODEL PREDICTION
# ============================================================

out = P.predict(df)


# ============================================================
# PREDICTION RESULTS
# ============================================================

st.markdown(
    '<div class="section-heading">Predicted Mechanistic Responses</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Predicted pavement responses from the deployed PAVE-ANN ensemble model.'
    '</div>',
    unsafe_allow_html=True,
)


cols = st.columns(len(P.primary))

for c, t in zip(cols, P.primary):

    mu = float(out[t].iloc[0])
    sd = float(out[f"{t}_sd"].iloc[0])

    c.metric(
        P.meta["outputs"]["description"][t],
        f"{mu:.4g} {P.units[t]}",
        f"± {1.96 * sd:.2g} (95%)",
    )


# ============================================================
# MODEL VALIDATION / STATUS
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
            ✓ All input parameters are within the validated training envelope.
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

life = P.design_life(
    out,
    vals.get("E_ACB", 3000.0),
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Allowable Fatigue Repetitions",
    f"{life['N_fatigue_allowable'].iloc[0]:.3g}",
)

c2.metric(
    "Allowable Rutting Repetitions",
    f"{life['N_rutting_allowable'].iloc[0]:.3g}",
)

c3.metric(
    "Governing Distress",
    str(life["governing_mode"].iloc[0]),
)

st.caption(
    "Transfer functions use the published Asphalt Institute coefficients. "
    "Replace them with locally calibrated values before design use."
)


# ============================================================
# TWO-COLUMN ANALYSIS AREA
# ============================================================

left_col, right_col = st.columns([1.45, 1])


# ============================================================
# LEFT - DEFLECTION BASIN
# ============================================================

with left_col:

    st.markdown(
        '<div class="section-heading">Deflection Basin</div>',
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
            "Deflection basin targets are not available "
            "in the current model configuration."
        )


# ============================================================
# RIGHT - PAVEMENT STRUCTURE SUMMARY
# ============================================================

with right_col:

    st.markdown(
        '<div class="section-heading">Pavement Structure</div>',
        unsafe_allow_html=True,
    )

    layer_rows = []

    for L in LAYERS:

        thickness_key = f"t_{L}"

        if thickness_key in vals:

            layer_rows.append(
                {
                    "Layer": PRETTY[L],
                    "Thickness": (
                        f"{vals[thickness_key]:.3g} "
                        f"{P.meta['inputs']['units'].get(thickness_key, '')}"
                    ),
                }
            )

    if layer_rows:

        layer_df = pd.DataFrame(layer_rows)

        st.dataframe(
            layer_df,
            use_container_width=True,
            hide_index=True,
            height=330,
        )


# ============================================================
# THICKNESS SWEEP
# ============================================================

st.markdown(
    '<div class="section-heading">Parametric Thickness / Input Sweep</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Evaluate the sensitivity of a selected pavement response '
    'to one design parameter while keeping other inputs fixed.'
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
# FULL PREDICTION TABLE
# ============================================================

with st.expander("Full Prediction Table"):

    st.dataframe(
        out.T,
        use_container_width=True,
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    info_c1, info_c2 = st.columns(2)

    with info_c1:

        st.markdown(
            """
            **PAVE-ANN v2.0**

            - Ensemble neural surrogate model
            - Mechanistic pavement response prediction
            - Uncertainty-aware predictions
            - Training-envelope range checking
            """
        )

    with info_c2:

        st.markdown(
            f"""
            **Deployment Information**

            - Ensemble members: {P.meta["n_ensemble_members"]}
            - Input variables: {len(P.features)}
            - Primary outputs: {len(P.primary)}
            - Model fingerprint: `{P.meta["fingerprint"]}`
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="dashboard-footer">
        PAVE-ANN v2.0 &nbsp; | &nbsp;
        Interactive Flexible Pavement Response Prediction
    </div>
    """,
    unsafe_allow_html=True,
)
