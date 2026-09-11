"""
PAVE-ANN v2.0 . Interactive Design Explorer

A drop-in replacement for the generated app_streamlit.py. It uses the same
PaveANNPredictor API, so no change to predictor.py is required.

    streamlit run app_streamlit.py
"""

from pathlib import Path
from textwrap import dedent
from typing import Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from predictor import PaveANNPredictor

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="PAVE-ANN | Flexible Pavement Response",
    page_icon="\U0001F6E3",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# THEME
# ===========================================================================
INK = "#0d2b45"
INK_SOFT = "#17628f"
ACCENT = "#1f8a70"
WARN = "#b54708"
MUTED = "#667085"

st.markdown(dedent(f"""
<style>
.stApp {{ background: #f4f6f9; }}
.block-container {{ max-width: 1560px; padding-top: 1rem; padding-bottom: 2.5rem; }}

/* ---------- header ---------- */
.pave-header {{
    background: linear-gradient(135deg, {INK} 0%, {INK_SOFT} 100%);
    border-radius: 16px; padding: 26px 32px; margin-bottom: 8px;
    box-shadow: 0 6px 20px rgba(13,43,69,.18);
}}
.pave-title {{ color:#fff; font-size:2.3rem; font-weight:800; line-height:1.05; margin:0;
              letter-spacing:-.02em; }}
.pave-subtitle {{ color:#dbe9f4; font-size:1.02rem; margin-top:8px; font-weight:400; }}
.pave-meta {{ color:#a9c6dc; font-size:.76rem; margin-top:14px; letter-spacing:.03em; }}

/* ---------- kpi strip ---------- */
.kpi-strip {{ display:flex; gap:10px; margin:10px 0 22px 0; flex-wrap:wrap; }}
.kpi {{ flex:1; min-width:150px; background:#fff; border:1px solid #e3e8ee;
        border-radius:11px; padding:12px 15px; }}
.kpi-label {{ color:{MUTED}; font-size:.70rem; font-weight:650; letter-spacing:.06em;
              text-transform:uppercase; }}
.kpi-value {{ color:{INK}; font-size:1.25rem; font-weight:800; margin-top:3px; }}

/* ---------- section ---------- */
.section-heading {{ color:{INK}; font-size:1.22rem; font-weight:750; margin:14px 0 2px 0; }}
.section-description {{ color:{MUTED}; font-size:.83rem; margin-bottom:14px; }}

/* ---------- output cards ---------- */
.output-card {{ background:#fff; border:1px solid #e3e8ee; border-left:4px solid {INK_SOFT};
                border-radius:12px; padding:16px 18px; min-height:136px;
                box-shadow:0 2px 8px rgba(16,24,40,.05); }}
.output-label {{ color:#475467; font-size:.76rem; font-weight:650; line-height:1.35;
                 min-height:36px; }}
.output-value {{ color:{INK}; font-size:1.7rem; font-weight:800; margin-top:6px;
                 white-space:nowrap; letter-spacing:-.02em; }}
.output-unit {{ font-size:.95rem; font-weight:650; color:{MUTED}; margin-left:3px; }}
.output-uncertainty {{ color:{MUTED}; font-size:.74rem; margin-top:5px; }}
.output-bar {{ height:4px; border-radius:2px; background:#eef1f5; margin-top:9px;
               overflow:hidden; }}
.output-bar span {{ display:block; height:100%; background:{INK_SOFT}; }}

/* ---------- status ---------- */
.status-good {{ background:#ecfdf3; border:1px solid #abefc6; color:#067647;
                border-radius:10px; padding:11px 15px; font-size:.82rem;
                font-weight:650; margin:14px 0; }}
.status-warning {{ background:#fffaeb; border:1px solid #fedf89; color:{WARN};
                   border-radius:10px; padding:11px 15px; font-size:.82rem;
                   font-weight:650; margin:14px 0; }}

/* ---------- info cards ---------- */
.info-card {{ background:#fff; border:1px solid #e3e8ee; border-radius:12px;
              padding:16px 19px; box-shadow:0 2px 8px rgba(16,24,40,.04);
              margin-bottom:12px; }}
.info-title {{ color:{INK}; font-size:.93rem; font-weight:750; margin-bottom:9px; }}
.info-row {{ color:#475467; font-size:.80rem; line-height:1.75;
             display:flex; justify-content:space-between; }}
.info-value {{ color:{INK}; font-weight:650; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background:{INK}; }}
section[data-testid="stSidebar"] * {{ color:#fff; }}
section[data-testid="stSidebar"] hr {{ border-color:rgba(255,255,255,.16); }}
.sidebar-title {{ font-size:1.25rem; font-weight:800; margin-bottom:4px; }}
.sidebar-description {{ color:#b9d0e2; font-size:.75rem; line-height:1.45;
                        margin-bottom:16px; }}
.sidebar-section {{ font-size:.92rem; font-weight:750;
                    border-bottom:1px solid rgba(255,255,255,.18);
                    padding-bottom:6px; margin:4px 0 10px 0; }}
/* expander shell, targeted both ways so it survives Streamlit versions */
section[data-testid="stSidebar"] div[data-testid="stExpander"],
section[data-testid="stSidebar"] details {{
    background:#16405f !important; border:1px solid rgba(255,255,255,.16) !important;
    border-radius:9px; margin-bottom:6px; }}
section[data-testid="stSidebar"] details > summary,
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {{
    background:#16405f !important; border-radius:9px; }}
section[data-testid="stSidebar"] details summary p,
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p {{
    color:#fff !important; font-weight:650; font-size:.84rem; }}
section[data-testid="stSidebar"] details svg {{ fill:#fff !important; }}

/* select boxes and number inputs: dark fill, light text */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"],
section[data-testid="stSidebar"] input {{
    background-color:#16405f !important; color:#fff !important;
    border-color:rgba(255,255,255,.22) !important; }}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{ fill:#cfe0ec; }}

/* buttons */
section[data-testid="stSidebar"] .stButton > button {{
    background:#1f6f9b; color:#fff !important;
    border:1px solid rgba(255,255,255,.28); }}
section[data-testid="stSidebar"] .stButton > button:hover {{ background:#2a86b8; }}

section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] label {{ color:#dbe9f4 !important;
                                          font-size:.73rem; }}

ul[data-baseweb="menu"], div[data-baseweb="popover"] li {{ font-size:.82rem; }}

/* ---------- misc ---------- */
.stButton > button {{ border-radius:9px; min-height:40px; font-weight:700; }}
div[data-testid="stAlert"], div[data-testid="stDataFrame"] {{ border-radius:10px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
.stTabs [data-baseweb="tab"] {{ border-radius:9px 9px 0 0; padding:9px 18px;
                                font-weight:650; }}
.dashboard-footer {{ border-top:1px solid #e3e8ee; margin-top:34px; padding-top:14px;
                     color:#98a2b3; font-size:.70rem; text-align:center; }}
</style>
"""), unsafe_allow_html=True)

# ===========================================================================
# MODEL
# ===========================================================================
@st.cache_resource(show_spinner="Loading PAVE-ANN ...")
def _load():
    return PaveANNPredictor.load(Path(__file__).parent)


P = _load()

LAYERS = ["WC", "ACB", "AggB", "SB", "Fill", "SG"]
PRETTY = {"WC": "Wearing Course", "ACB": "AC Base", "AggB": "Aggregate Base",
          "SB": "Subbase", "Fill": "Subgrade Fill", "SG": "Natural Subgrade"}
LAYER_FILL = {"WC": "#2f3e4e", "ACB": "#4a5a6a", "AggB": "#a98b5f",
              "SB": "#c0a878", "Fill": "#cbbb9c", "SG": "#b39b7d"}
LAYER_TEXT = {"WC": "#ffffff", "ACB": "#ffffff", "AggB": "#20160a",
              "SB": "#20160a", "Fill": "#2b2314", "SG": "#2b2314"}
PROP_GROUPS = [("t", "Thickness"), ("E", "Modulus"), ("\u03bd", "Poisson"),
               ("\u03c1", "Density"), ("\u03c3y", "Yield"), ("\u03c6", "Friction"),
               ("K", "Flow ratio"), ("\u03c8", "Dilation")]


def unit_of(f: str) -> str:
    return P.meta["inputs"]["units"].get(f, "")


def bounds(f: str):
    i = P.features.index(f)
    return float(P.lo[i]), float(P.hi[i])


@st.cache_data(show_spinner=False)
def predict_frame(records: List[Dict[str, float]]) -> pd.DataFrame:
    """Cached prediction. Records are hashable, DataFrames are not."""
    return P.predict(pd.DataFrame(records))


# ===========================================================================
# HEADER
# ===========================================================================
st.markdown(f"""
<div class="pave-header">
  <div class="pave-title">PAVE-ANN</div>
  <div class="pave-subtitle">
     Neural network surrogate for three-dimensional flexible pavement response
  </div>
  <div class="pave-meta">
     VERSION {P.meta['version']} &nbsp;&middot;&nbsp;
     {P.meta['n_ensemble_members']}-MEMBER ENSEMBLE &nbsp;&middot;&nbsp;
     {len(P.features)} INPUTS &nbsp;&rarr;&nbsp; {len(P.targets)} RESPONSES
     &nbsp;&middot;&nbsp; FINGERPRINT {P.meta['fingerprint']}
  </div>
</div>
""", unsafe_allow_html=True)

_tm = P.meta.get("test_metrics", {})
_kpis = []
for t in P.primary:
    r2 = _tm.get(t, {}).get("R2")
    if r2 is not None:
        _kpis.append((P.meta["outputs"]["description"][t].split(",")[0], f"R\u00b2 {r2:.4f}"))
if _kpis:
    st.markdown('<div class="kpi-strip">' + "".join(
        f'<div class="kpi"><div class="kpi-label">{k}</div>'
        f'<div class="kpi-value">{v}</div></div>' for k, v in _kpis
    ) + "</div>", unsafe_allow_html=True)

# ===========================================================================
# SIDEBAR . DESIGN INPUTS
# ===========================================================================
template = P.template().iloc[0].to_dict()

PRESETS = {
    "Median design": {},
    "Thin flexible": {"t_WC": .25, "t_ACB": .20, "t_AggB": .35, "t_SB": .35},
    "Heavy duty": {"t_WC": .80, "t_ACB": .85, "t_AggB": .75, "t_SB": .70},
    "Weak subgrade": {"E_SG": .10, "E_Fill": .20, "t_SB": .75},
    "High tyre pressure": {"P": .95},
}


def apply_preset(name: str) -> Dict[str, float]:
    """Preset fractions are positions within each parameter's sampled range."""
    v = dict(template)
    for f, frac in PRESETS[name].items():
        if f in P.features:
            lo, hi = bounds(f)
            v[f] = lo + frac * (hi - lo)
    return v


with st.sidebar:
    st.markdown('<div class="sidebar-title">DESIGN INPUTS</div>'
                '<div class="sidebar-description">Set loading, environment and '
                'layer properties. Every slider is bounded by the training '
                'envelope of the underlying simulation database.</div>',
                unsafe_allow_html=True)

    preset = st.selectbox("Starting point", list(PRESETS), index=0)
    if st.button("Reset to preset", use_container_width=True):
        for k in list(st.session_state):
            if k.startswith("in_"):
                del st.session_state[k]
        st.rerun()

    base = apply_preset(preset)
    vals: Dict[str, float] = dict(base)

    st.markdown('<div class="sidebar-section">Loading and environment</div>',
                unsafe_allow_html=True)
    for f in ["P", "T", "S"]:
        if f in P.features:
            lo, hi = bounds(f)
            vals[f] = st.slider(f"{f}  [{unit_of(f)}]", lo, hi, float(base[f]),
                                format="%.4g", key=f"in_{f}")

    st.markdown('<div class="sidebar-section">Layer properties</div>',
                unsafe_allow_html=True)
    for L in LAYERS:
        with st.expander(PRETTY[L], expanded=(L in ("WC", "ACB"))):
            for pre, _ in PROP_GROUPS:
                f = f"{pre}_{L}"
                if f in P.features:
                    lo, hi = bounds(f)
                    step = (hi - lo) / 200 if hi > lo else None
                    # densities are of order 1e-9 t/mm^3, so a fixed-decimal
                    # slider would read 0.00 for every value in range
                    fmt = "%.3g" if max(abs(lo), abs(hi)) < 0.01 else "%.4g"
                    vals[f] = st.slider(f"{f}  [{unit_of(f)}]", lo, hi,
                                        float(base[f]), step=step,
                                        format=fmt, key=f"in_{f}")

    st.markdown("---")
    design_esal = st.number_input(
        "Design traffic (ESALs)", min_value=1e4, max_value=1e9, value=1e7,
        format="%.0f", help="Used to compute the utilisation ratio on the "
                            "Design Explorer tab.")

# ===========================================================================
# PREDICTION
# ===========================================================================
out = predict_frame([vals])
life = P.design_life(out, vals.get("E_ACB", 3000.0))

tab_design, tab_sens, tab_batch, tab_card = st.tabs(
    ["  Design Explorer  ", "  Sensitivity  ", "  Batch Analysis  ",
     "  Model Card  "])

# ===========================================================================
# TAB 1 . DESIGN EXPLORER
# ===========================================================================
with tab_design:
    st.markdown('<div class="section-heading">Predicted mechanistic responses</div>'
                '<div class="section-description">Ensemble mean with a 95 per cent '
                'predictive interval. The bar shows the share of that interval coming '
                'from disagreement between ensemble members: a long bar means the design '
                'is unfamiliar to the model, not merely noisy.</div>',
                unsafe_allow_html=True)

    cols = st.columns(len(P.primary))
    for c, t in zip(cols, P.primary):
        mu = float(out[t].iloc[0])
        sd = float(out[f"{t}_sd"].iloc[0])
        desc = P.meta["outputs"]["description"][t]
        unit = P.units[t]
        # the bar shows how much of the predictive variance is ensemble
        # disagreement rather than intrinsic scatter: a long bar means the
        # members disagree, which is the signal that this design is unfamiliar
        ale = float(out.get(f"{t}_aleatoric_sd", pd.Series([np.nan])).iloc[0])
        epi = float(out.get(f"{t}_epistemic_sd", pd.Series([np.nan])).iloc[0])
        if np.isfinite(ale) and np.isfinite(epi) and (ale + epi) > 0:
            epi_share = 100 * epi ** 2 / (ale ** 2 + epi ** 2)
            foot = f"epistemic share {epi_share:.0f}%"
        else:
            epi_share, foot = 0.0, f"relative spread {100*sd/(abs(mu)+1e-30):.1f}%"
        c.markdown(f"""
        <div class="output-card">
          <div class="output-label">{desc}</div>
          <div class="output-value">{mu:,.4g}<span class="output-unit">{unit}</span></div>
          <div class="output-uncertainty">95% interval &plusmn; {1.96*sd:.3g} {unit}
             &nbsp;&middot;&nbsp; {foot}</div>
          <div class="output-bar"><span style="width:{min(epi_share,100):.0f}%"></span></div>
        </div>""", unsafe_allow_html=True)

    n_out = int(out["n_params_out_of_range"].iloc[0])
    if n_out:
        st.markdown(
            f'<div class="status-warning">Extrapolation warning &nbsp;&middot;&nbsp; '
            f'{n_out} parameter(s) lie outside the training envelope, the worst being '
            f'<b>{out["worst_parameter"].iloc[0]}</b> at '
            f'{float(out["max_extrapolation_frac"].iloc[0]):.2f} of the sampled span. '
            f'Treat these predictions as indicative and verify with finite element '
            f'analysis.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-good">All input parameters lie within the '
                    'validated training envelope.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.25])

    # ---------------- pavement cross-section (proportional) ----------------
    with left:
        st.markdown('<div class="section-heading">Pavement cross-section</div>'
                    '<div class="section-description">Layer depths drawn to scale.'
                    '</div>', unsafe_allow_html=True)

        th = {L: float(vals.get(f"t_{L}", 0.0)) for L in LAYERS}
        total = sum(th.values()) or 1.0
        H, W, MIN_H = 430, 430, 26
        free = H - MIN_H * len(LAYERS)
        y, rects = 0.0, []
        for L in LAYERS:
            h = MIN_H + free * (th[L] / total)
            rects.append((L, y, h))
            y += h

        svg = [f'<svg viewBox="0 0 {W} {H+34}" width="100%" '
               f'xmlns="http://www.w3.org/2000/svg" font-family="DejaVu Sans, sans-serif">']
        svg.append(f'<rect x="0" y="0" width="{W}" height="12" fill="#c62828" '
                   f'opacity=".85" rx="2"/>')
        svg.append(f'<text x="{W/2}" y="9.5" fill="#fff" font-size="8.5" '
                   f'font-weight="700" text-anchor="middle">'
                   f'TYRE CONTACT PRESSURE {vals.get("P", 0):.2f} MPa</text>')
        depth = 0.0
        for L, yy, hh in rects:
            top = yy + 18
            svg.append(f'<rect x="0" y="{top:.1f}" width="{W}" height="{hh:.1f}" '
                       f'fill="{LAYER_FILL[L]}" stroke="#ffffff" stroke-width="1"/>')
            svg.append(f'<text x="12" y="{top+hh/2+3.5:.1f}" fill="{LAYER_TEXT[L]}" '
                       f'font-size="11" font-weight="700">{PRETTY[L]}</text>')
            svg.append(f'<text x="{W-12}" y="{top+hh/2+3.5:.1f}" fill="{LAYER_TEXT[L]}" '
                       f'font-size="11" text-anchor="end">{th[L]:.0f} mm'
                       f'&#160;&#160;|&#160;&#160;{vals.get(f"E_{L}", 0):,.0f} MPa</text>')
            depth += th[L]
        svg.append(f'<text x="0" y="{H+30}" fill="{MUTED}" font-size="10">'
                   f'Total profile depth {total:,.0f} mm</text>')
        svg.append("</svg>")
        st.markdown("".join(svg), unsafe_allow_html=True)

    # ---------------- deflection basin against depth ----------------
    with right:
        st.markdown('<div class="section-heading">Deflection with depth</div>'
                    '<div class="section-description">Vertical displacement at each '
                    'layer interface. Depth increases downward, as in the finite '
                    'element model.</div>', unsafe_allow_html=True)

        chain = ["U2_surface"] + [f"U2_{L}" for L in LAYERS[1:]]
        chain = [c for c in chain if c in P.targets]
        if len(chain) > 1:
            depths, d = [], 0.0
            for i, _ in enumerate(chain):
                depths.append(d)
                if i < len(LAYERS):
                    d += float(vals.get(f"t_{LAYERS[i]}", 0.0))
            basin = pd.DataFrame({
                "Depth (mm)": depths,
                "|U2| (mm)": [abs(float(out[c].iloc[0])) for c in chain],
                "Interface": ["Surface"] + [PRETTY[L] for L in LAYERS[1:len(chain)]],
            })
            ch = (alt.Chart(basin)
                  .mark_line(point=alt.OverlayMarkDef(size=70, filled=True),
                             strokeWidth=2.6, color=INK_SOFT)
                  .encode(
                      x=alt.X("|U2| (mm):Q", title="Vertical deflection |U\u2082| (mm)",
                              scale=alt.Scale(zero=True)),
                      y=alt.Y("Depth (mm):Q", title="Depth below surface (mm)",
                              scale=alt.Scale(reverse=True)),
                      tooltip=["Interface", alt.Tooltip("|U2| (mm):Q", format=".4f"),
                               alt.Tooltip("Depth (mm):Q", format=".0f")])
                  .properties(height=420))
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info("Depth-profile channels are not present in this model.")

    # ---------------- design life ----------------
    st.markdown('<div class="section-heading">Design life screening</div>'
                '<div class="section-description">Allowable repetitions from the '
                'Asphalt Institute transfer functions, compared against the design '
                'traffic set in the sidebar.</div>', unsafe_allow_html=True)

    nf = float(life["N_fatigue_allowable"].iloc[0])
    nd = float(life["N_rutting_allowable"].iloc[0])
    ndes = min(nf, nd)
    util = design_esal / ndes if ndes > 0 else np.inf

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Allowable repetitions, fatigue", f"{nf:.3g}")
    g2.metric("Allowable repetitions, rutting", f"{nd:.3g}")
    g3.metric("Governing distress", str(life["governing_mode"].iloc[0]))
    g4.metric("Utilisation, demand / capacity", f"{util:.2f}",
              delta="adequate" if util <= 1 else "inadequate",
              delta_color="normal" if util <= 1 else "inverse")

    st.caption("Transfer functions carry the published Asphalt Institute default "
               "coefficients and are provided for screening only. Replace them with "
               "locally calibrated coefficients before design use.")

    with st.expander("Full prediction table, all nine channels"):
        st.dataframe(out.T, use_container_width=True)

# ===========================================================================
# TAB 2 . SENSITIVITY
# ===========================================================================
with tab_sens:
    st.markdown('<div class="section-heading">One-parameter sweep</div>'
                '<div class="section-description">One input is varied across its '
                'sampled range while every other input is held at the current design. '
                'The shaded band is the 95 per cent predictive interval.</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    sweepable = [f for f in P.features
                 if f.startswith(("t_", "E_")) or f in ("P", "T", "S")]
    with c1:
        sweep_f = st.selectbox("Input parameter", sweepable,
                               index=sweepable.index("t_ACB")
                               if "t_ACB" in sweepable else 0)
    with c2:
        sweep_t = st.selectbox("Response", P.primary)
    with c3:
        n_grid = st.slider("Grid resolution", 15, 80, 40)

    lo, hi = bounds(sweep_f)
    grid = np.linspace(lo, hi, n_grid)
    recs = []
    for g in grid:
        r = dict(vals)
        r[sweep_f] = float(g)
        recs.append(r)
    res = predict_frame(recs)

    sw = pd.DataFrame({
        sweep_f: grid,
        "mean": res[sweep_t].to_numpy(),
        "lo": res[sweep_t].to_numpy() - 1.96 * res[f"{sweep_t}_sd"].to_numpy(),
        "hi": res[sweep_t].to_numpy() + 1.96 * res[f"{sweep_t}_sd"].to_numpy(),
    })
    band = (alt.Chart(sw).mark_area(opacity=.18, color=INK_SOFT)
            .encode(x=alt.X(f"{sweep_f}:Q", title=f"{sweep_f}  [{unit_of(sweep_f)}]"),
                    y=alt.Y("lo:Q", title=f"{sweep_t}  [{P.units[sweep_t]}]"),
                    y2="hi:Q"))
    line = (alt.Chart(sw).mark_line(strokeWidth=2.8, color=INK)
            .encode(x=f"{sweep_f}:Q", y="mean:Q",
                    tooltip=[alt.Tooltip(f"{sweep_f}:Q", format=".3f"),
                             alt.Tooltip("mean:Q", format=".5g")]))
    marker = (alt.Chart(pd.DataFrame({"x": [vals[sweep_f]]}))
              .mark_rule(strokeDash=[5, 4], color=WARN, strokeWidth=1.6)
              .encode(x="x:Q"))
    st.altair_chart((band + line + marker).properties(height=380),
                    use_container_width=True)
    st.caption("The dashed line marks the current design.")

    # ---------------- tornado ----------------
    st.markdown('<div class="section-heading">Local sensitivity, tornado</div>'
                '<div class="section-description">Each parameter is perturbed by '
                '\u00b110 per cent about the current design, with all others held '
                'fixed. Bars show the resulting percentage change in the selected '
                'response.</div>', unsafe_allow_html=True)

    tor_t = st.selectbox("Response for the tornado", P.primary, key="tor")
    tor_feats = [f for f in P.features if f.startswith(("t_", "E_")) or f == "P"]
    recs, labels = [dict(vals)], []
    for f in tor_feats:
        lo_f, hi_f = bounds(f)
        for sgn in (-0.10, +0.10):
            r = dict(vals)
            r[f] = float(np.clip(vals[f] * (1 + sgn), lo_f, hi_f))
            recs.append(r)
        labels.append(f)
    tr = predict_frame(recs)
    base_v = float(tr[tor_t].iloc[0])
    rows = []
    for i, f in enumerate(labels):
        dn = float(tr[tor_t].iloc[1 + 2 * i])
        up = float(tr[tor_t].iloc[2 + 2 * i])
        rows.append({"Parameter": f,
                     "low": 100 * (dn - base_v) / (abs(base_v) + 1e-30),
                     "high": 100 * (up - base_v) / (abs(base_v) + 1e-30)})
    tdf = pd.DataFrame(rows)
    tdf["span"] = (tdf["high"] - tdf["low"]).abs()
    tdf = tdf.sort_values("span", ascending=False).head(14)
    tlong = tdf.melt(id_vars=["Parameter", "span"], value_vars=["low", "high"],
                     var_name="direction", value_name="change")
    tor = (alt.Chart(tlong)
           .mark_bar(height=13)
           .encode(x=alt.X("change:Q", title=f"Change in {tor_t} (%)"),
                   y=alt.Y("Parameter:N", sort=tdf["Parameter"].tolist(), title=None),
                   color=alt.Color("direction:N",
                                   scale=alt.Scale(domain=["low", "high"],
                                                   range=[WARN, INK_SOFT]),
                                   legend=alt.Legend(title="\u00b110% perturbation")),
                   tooltip=["Parameter", alt.Tooltip("change:Q", format=".2f")])
           .properties(height=28 * len(tdf)))
    st.altair_chart(tor, use_container_width=True)

# ===========================================================================
# TAB 3 . BATCH
# ===========================================================================
with tab_batch:
    st.markdown('<div class="section-heading">Batch scoring</div>'
                '<div class="section-description">Upload a CSV with one row per '
                'design and the 43 parameter columns in any order. Predictions, '
                'uncertainties and envelope checks are returned for every row.</div>',
                unsafe_allow_html=True)

    b1, b2 = st.columns([1, 2])
    with b1:
        st.download_button("Download input template",
                           P.template().to_csv(index=False).encode(),
                           "pave_ann_template.csv", "text/csv",
                           use_container_width=True)
    with b2:
        up = st.file_uploader("CSV of designs", type=["csv"],
                              label_visibility="collapsed")

    if up is not None:
        try:
            bdf = pd.read_csv(up)
            bout = P.predict(bdf)
            merged = pd.concat([bdf.reset_index(drop=True),
                                bout.reset_index(drop=True)], axis=1)
            flagged = int((bout["n_params_out_of_range"] > 0).sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Designs scored", f"{len(bdf):,}")
            m2.metric("Outside training envelope", f"{flagged:,}")
            m3.metric("Share flagged", f"{flagged/max(len(bdf),1):.1%}")
            if flagged:
                st.markdown('<div class="status-warning">Some designs fall outside '
                            'the training envelope. Their predictions are '
                            'extrapolations.</div>', unsafe_allow_html=True)

            st.dataframe(merged.head(200), use_container_width=True, height=340)
            st.download_button("Download predictions",
                               merged.to_csv(index=False).encode(),
                               "pave_ann_predictions.csv", "text/csv")

            st.markdown('<div class="section-heading">Distribution of predicted '
                        'responses</div>', unsafe_allow_html=True)
            hc = st.columns(len(P.primary))
            for col, t in zip(hc, P.primary):
                h = (alt.Chart(pd.DataFrame({t: bout[t]}))
                     .mark_bar(color=INK_SOFT, opacity=.85)
                     .encode(x=alt.X(f"{t}:Q", bin=alt.Bin(maxbins=28),
                                     title=f"{t} [{P.units[t]}]"),
                             y=alt.Y("count()", title="designs"))
                     .properties(height=200))
                col.altair_chart(h, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")

# ===========================================================================
# TAB 4 . MODEL CARD
# ===========================================================================
with tab_card:
    st.markdown('<div class="section-heading">Model card</div>'
                '<div class="section-description">Held-out accuracy, provenance and '
                'the conditions under which these predictions are valid.</div>',
                unsafe_allow_html=True)

    if _tm:
        rows = []
        for t in P.targets:
            m = _tm.get(t, {})
            rows.append({"Response": P.meta["outputs"]["description"].get(t, t),
                         "Unit": P.units.get(t, ""),
                         "R\u00b2": m.get("R2"), "RMSE": m.get("RMSE"),
                         "MAE": m.get("MAE"), "MAPE (%)": m.get("MAPE_%")})
        st.dataframe(pd.DataFrame(rows).round(5), use_container_width=True,
                     hide_index=True)

    i1, i2 = st.columns(2)
    with i1:
        st.markdown(f"""
        <div class="info-card">
          <div class="info-title">Provenance</div>
          <div class="info-row"><span>Version</span>
             <span class="info-value">{P.meta['version']}</span></div>
          <div class="info-row"><span>Created</span>
             <span class="info-value">{P.meta.get('created','n/a')}</span></div>
          <div class="info-row"><span>Ensemble members</span>
             <span class="info-value">{P.meta['n_ensemble_members']}</span></div>
          <div class="info-row"><span>Input parameters</span>
             <span class="info-value">{len(P.features)}</span></div>
          <div class="info-row"><span>Output channels</span>
             <span class="info-value">{len(P.targets)}</span></div>
          <div class="info-row"><span>Fingerprint</span>
             <span class="info-value">{P.meta['fingerprint']}</span></div>
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="info-card">
          <div class="info-title">Conditions of validity</div>
          <div class="info-row"><span>Interpolation only, inside the sampled
             envelope</span></div>
          <div class="info-row"><span>Interfaces fully bonded, an upper bound on
             coupling</span></div>
          <div class="info-row"><span>Stationary load, 150 mm contact radius</span></div>
          <div class="info-row"><span>Predictive intervals are under-dispersed and
             rank difficulty rather than give probabilities</span></div>
          <div class="info-row"><span>Tensile strain is the least reliable
             channel</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Training envelope</div>'
                '<div class="section-description">Sliders are bounded by these '
                'ranges. Predictions outside them are extrapolations.</div>',
                unsafe_allow_html=True)
    env = pd.DataFrame({
        "Parameter": P.features,
        "Unit": [unit_of(f) for f in P.features],
        "Minimum": P.lo, "Median": P.median, "Maximum": P.hi,
    })
    st.dataframe(env, use_container_width=True, hide_index=True, height=420)

# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown(f"""
<div class="dashboard-footer">
PAVE-ANN v{P.meta['version']} &nbsp;&middot;&nbsp; neural surrogate for flexible
pavement response &nbsp;&middot;&nbsp; predictions are interpolations of a finite
element database and are not a substitute for design verification
</div>""", unsafe_allow_html=True)
