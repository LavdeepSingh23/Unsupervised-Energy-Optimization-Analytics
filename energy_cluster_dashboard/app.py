import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Enerlytics AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#RRGGBB' → 'rgba(r,g,b,alpha)' — required by Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def sil_color(score: float) -> str:
    if score >= 0.5: return "#63E6BE"
    if score >= 0.3: return "#FBBF24"
    return "#F87171"

def sil_label(score: float) -> str:
    if score >= 0.5: return "Strong"
    if score >= 0.3: return "Reasonable"
    if score >= 0.1: return "Weak"
    return "Poor"

PALETTE = ["#63E6BE","#818CF8","#FBBF24","#F87171",
           "#38BDF8","#A78BFA","#FB923C","#34D399","#E879F9","#94A3B8"]

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #080B12 !important;
    color: #E8EAF0 !important;
    font-family: 'Instrument Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 60% 40% at 20% 10%, rgba(99,230,190,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 35% at 80% 80%, rgba(130,100,255,0.08) 0%, transparent 60%),
        #080B12 !important;
}
[data-testid="stSidebar"] {
    background: #0D1017 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
.block-container { padding: 2rem 3rem !important; max-width: 1440px !important; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.03em !important; }

[data-testid="stRadio"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E8EAF0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #63E6BE !important;
    border-bottom-color: #63E6BE !important;
}
hr { border-color: rgba(255,255,255,0.06) !important; }

/* Hide Streamlit branding — DO NOT hide full header (kills sidebar toggle) */
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Always show sidebar collapse/expand arrow */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #63E6BE !important;
}

/* Make sidebar toggle arrow visible on dark bg */
[data-testid="stSidebarNav"] { display: none; }
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ── KPI Cards ── */
.e-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px 22px;
    transition: border-color 0.3s, transform 0.3s;
    position: relative;
    overflow: hidden;
    height: 100%;
    margin-bottom: 4px;
}
.e-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #63E6BE, #818CF8);
    opacity: 0; transition: opacity 0.3s;
}
.e-card:hover { border-color: rgba(99,230,190,0.3); transform: translateY(-2px); }
.e-card:hover::before { opacity: 1; }
.e-card .label {
    font-family: 'DM Mono', monospace; font-size: 0.64rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: #4B5563; margin-bottom: 10px;
}
.e-card .value {
    font-family: 'Syne', sans-serif; font-size: 2.1rem;
    font-weight: 800; color: #E8EAF0; line-height: 1;
}
.e-card .sub { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #63E6BE; margin-top: 8px; }

/* ── Page headers ── */
.sec-header {
    font-family: 'Syne', sans-serif; font-size: 2rem;
    font-weight: 800; color: #E8EAF0; letter-spacing: -0.04em; margin-bottom: 4px;
}
.sec-sub { font-size: 0.87rem; color: #4B5563; margin-bottom: 1.6rem; }

/* ── Pill tags ── */
.tag {
    display: inline-block; background: rgba(99,230,190,0.1); color: #63E6BE;
    border: 1px solid rgba(99,230,190,0.2); border-radius: 999px;
    padding: 3px 14px; font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 14px;
}

/* ── Insight cards ── */
.insight-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 16px 18px; margin-bottom: 10px;
}
.insight-card .i-label {
    font-family: 'DM Mono', monospace; font-size: 0.63rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6B7280; margin-bottom: 4px;
}
.insight-card .i-value { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #E8EAF0; }
.insight-card .i-desc { font-size: 0.77rem; color: #4B5563; margin-top: 4px; line-height: 1.5; }

/* ── Recommendation cards ── */
.rec-card {
    border-radius: 14px; padding: 20px 22px; margin-bottom: 14px;
    border-left: 3px solid; background: rgba(255,255,255,0.025);
}
.rec-card .rec-priority {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px;
}
.rec-card .rec-title {
    font-family: 'Syne', sans-serif; font-size: 1rem;
    font-weight: 700; color: #E8EAF0; margin-bottom: 6px;
}
.rec-card .rec-body { font-size: 0.82rem; color: #6B7280; line-height: 1.6; }
.rec-card .rec-impact {
    display: inline-block; margin-top: 12px;
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    background: rgba(99,230,190,0.1); color: #63E6BE;
    border: 1px solid rgba(99,230,190,0.2); border-radius: 999px; padding: 3px 12px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SHARED PLOTLY DARK THEME
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Instrument Sans, sans-serif", color="#6B7280", size=12),
    margin=dict(l=20, r=20, t=48, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    X_pca = pd.read_csv("pca_transformed_data.csv")
    df    = pd.read_csv("final_feature_engineered.csv")
    return X_pca, df

X_pca_raw, original_df_raw = load_data()

pc_cols = [c for c in X_pca_raw.columns if c.startswith("PC")]

# ─────────────────────────────────────────────
# PCA VARIANCE (proxy from column variances)
# ─────────────────────────────────────────────
@st.cache_data
def compute_pca_variance(X_pca: pd.DataFrame):
    cols = [c for c in X_pca.columns if c.startswith("PC")]
    if not cols:
        return {}, 0.0
    var        = X_pca[cols].var()
    total      = var.sum()
    pct        = (var / total * 100).round(2)
    cumulative = float(pct.cumsum().iloc[-1])
    return pct.to_dict(), cumulative

pca_var_dict, pca_cumulative = compute_pca_variance(X_pca_raw)

# ─────────────────────────────────────────────
# KMEANS (cached per k)
# ─────────────────────────────────────────────
@st.cache_data
def run_kmeans_k(X_arr: np.ndarray, k: int):
    km     = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_arr)
    return labels, km.inertia_

@st.cache_data
def compute_cluster_metrics(X_arr: np.ndarray, labels: np.ndarray):
    sil_avg = float(silhouette_score(X_arr, labels))
    sil_smp = silhouette_samples(X_arr, labels)
    ch      = float(calinski_harabasz_score(X_arr, labels))
    return sil_avg, sil_smp, ch

@st.cache_data
def compute_k_range_metrics(X_arr: np.ndarray, k_min: int = 2, k_max: int = 10):
    rows = []
    for k in range(k_min, k_max + 1):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_arr)
        rows.append({
            "k":       k,
            "inertia": km.inertia_,
            "sil":     silhouette_score(X_arr, lbl),
            "ch":      calinski_harabasz_score(X_arr, lbl),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────
def build_recommendations(seg_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> list:
    recs       = []
    port_means = portfolio_df.mean(numeric_only=True)
    seg_means  = seg_df.mean(numeric_only=True)

    def sv(name): return float(seg_means.get(name, 0))
    def bv(name): return float(port_means.get(name, 0))

    avg_cost   = seg_df["Energy_Cost"].mean()
    avg_saving = seg_df["Potential_Savings_Value"].mean()
    n          = len(seg_df)

    # Each rule: (feature, threshold_ratio, above_or_below, priority, color, title, body, saving_pct)
    rules = [
        ("HVAC_Ratio",      1.15, "above", "HIGH",   "#F87171",
         "Reduce HVAC Energy Consumption",
         lambda: (f"HVAC ratio {sv('HVAC_Ratio'):.2f} is "
                  f"{(sv('HVAC_Ratio')-bv('HVAC_Ratio'))/max(bv('HVAC_Ratio'),1e-9)*100:.1f}% "
                  f"above portfolio avg ({bv('HVAC_Ratio'):.2f}). "
                  "Scheduling setbacks, AHU recommissioning, and variable-speed drives are fastest levers."),
         lambda: avg_cost * 0.08 * n),

        ("Renewable_Ratio", 0.85, "below", "HIGH",   "#F87171",
         "Increase Renewable Energy Mix",
         lambda: (f"Renewable penetration {sv('Renewable_Ratio'):.1%} is below portfolio avg "
                  f"{bv('Renewable_Ratio'):.1%}. Green tariff or on-site solar are strong candidates."),
         lambda: avg_cost * 0.06 * n),

        ("Carbon_Load",     1.20, "above", "MEDIUM", "#FBBF24",
         "Carbon Reduction Programme",
         lambda: (f"Carbon load {sv('Carbon_Load'):.2f} is "
                  f"{(sv('Carbon_Load')-bv('Carbon_Load'))/max(bv('Carbon_Load'),1e-9)*100:.1f}% "
                  "above average. Electrification and fuel switching are primary interventions."),
         lambda: avg_cost * 0.05 * n),

        ("Efficiency_Gap",  1.10, "above", "HIGH",   "#F87171",
         "Close the Efficiency Gap",
         lambda: (f"Efficiency gap {sv('Efficiency_Gap'):.2f} is "
                  f"{(sv('Efficiency_Gap')-bv('Efficiency_Gap'))/max(bv('Efficiency_Gap'),1e-9)*100:.1f}% "
                  "above baseline. IoT fault detection typically closes 30–50% within 18 months."),
         lambda: avg_saving * 0.4 * n),

        ("Lighting_Ratio",  1.20, "above", "MEDIUM", "#FBBF24",
         "LED Lighting Retrofit",
         lambda: (f"Lighting ratio {sv('Lighting_Ratio'):.2f} vs avg {bv('Lighting_Ratio'):.2f}. "
                  "LED + occupancy sensors deliver 40–60% lighting reduction, payback 2–4 years."),
         lambda: avg_cost * 0.04 * n),

        ("Grid_Stress",     1.15, "above", "MEDIUM", "#FBBF24",
         "Demand Flexibility / Peak Shaving",
         lambda: (f"Grid stress {sv('Grid_Stress'):.2f} signals heavy peak draws. "
                  "Battery storage or demand-response can shift 10–20% of peak load."),
         lambda: avg_cost * 0.03 * n),

        ("Temp_Deviation",  1.25, "above", "LOW",    "#63E6BE",
         "Thermal Envelope Improvements",
         lambda: (f"Temp deviation {sv('Temp_Deviation'):.2f} vs avg {bv('Temp_Deviation'):.2f}. "
                  "Insulation and glazing upgrades reduce weather-driven load swings."),
         lambda: avg_cost * 0.04 * n),

        ("Savings_Gap",     1.05, "above", "HIGH",   "#F87171",
         "Immediate Operational Optimisation",
         lambda: (f"Savings gap {sv('Savings_Gap'):.2f} above average. "
                  "A 90-day sprint on set-points, schedules and controls recovers 20–35% near-term."),
         lambda: avg_saving * n),
    ]

    for feat, ratio, direction, priority, color, title, body_fn, saving_fn in rules:
        if feat not in seg_df.columns:
            continue
        b  = bv(feat)
        cv = sv(feat)
        if direction == "above":
            fired = cv > b * ratio
        else:
            fired = cv < b * ratio
        if fired:
            saving = saving_fn()
            recs.append({
                "priority":     priority,
                "color":        color,
                "title":        title,
                "body":         body_fn(),
                "impact":       f"Est. saving: ${saving/1e6:.2f}M across segment",
                "saving_value": saving,
            })

    recs.sort(key=lambda x: x["saving_value"], reverse=True)

    if not recs:
        recs.append({
            "priority": "LOW", "color": "#63E6BE",
            "title": "Maintain & Monitor — Segment Performing Well",
            "body": ("This segment tracks at or below portfolio averages across all dimensions. "
                     "Sustain current practices and use it as a benchmark."),
            "impact": "No immediate interventions required",
            "saving_value": 0,
        })
    return recs

# ─────────────────────────────────────────────
# SIDEBAR  — K slider + navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 8px 16px;'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#E8EAF0;'>⚡ Enerlytics</div>
        <div style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.1em;
                    color:#374151;text-transform:uppercase;margin-top:4px;'>AI · Energy Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:DM Mono,monospace;font-size:0.65rem;letter-spacing:0.1em;
                text-transform:uppercase;color:#4B5563;margin-bottom:6px;padding:0 2px;'>
        Cluster Count (K)
    </div>
    """, unsafe_allow_html=True)

    k = st.slider("K clusters", min_value=2, max_value=10, value=4, step=1,
                  label_visibility="collapsed")

    st.markdown(f"""
    <div style='font-family:DM Mono,monospace;font-size:0.67rem;color:#63E6BE;
                letter-spacing:0.06em;margin-bottom:18px;'>→ KMeans k={k} active</div>
    """, unsafe_allow_html=True)

    section = st.radio("Navigation", [
        "Overview",
        "Cluster Quality",
        "Cluster Explorer",
        "Segment Profiles",
        "Recommendations",
        "Business Impact",
        "Anomaly Detector",
        "Export Report",
    ], label_visibility="collapsed")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    n_pc = len(pc_cols)
    st.markdown(f"""
    <div style='padding:0 2px;font-family:DM Mono,monospace;font-size:0.64rem;
                color:#374151;letter-spacing:0.08em;line-height:2.4;'>
        BUILDINGS &nbsp;·&nbsp;<span style='color:#63E6BE'>{len(original_df_raw):,}</span><br>
        K &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;·&nbsp;<span style='color:#63E6BE'>{k}</span><br>
        PCA DIMS &nbsp;&nbsp;·&nbsp;<span style='color:#63E6BE'>{n_pc}</span><br>
        STATUS &nbsp;&nbsp;&nbsp;&nbsp;·&nbsp;<span style='color:#63E6BE'>● Live</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RUN KMEANS + METRICS FOR CURRENT K
# ─────────────────────────────────────────────
X_arr = X_pca_raw[pc_cols].values  # numpy array — hashable by cache

cluster_labels, inertia = run_kmeans_k(X_arr, k)
sil_avg, sil_smp, ch_score = compute_cluster_metrics(X_arr, cluster_labels)

# Build working dataframes
X_pca       = X_pca_raw.copy()
original_df = original_df_raw.copy()
X_pca["Cluster"]       = cluster_labels
original_df["Cluster"] = cluster_labels

# Dynamic names / colours for any k
if k == 4:
    cluster_names = {
        0: "Low-Cost Sustainable",
        1: "Climate Sensitive",
        2: "Optimisation Target",
        3: "High HVAC / Renewables",
    }
    cluster_colors = {
        "Low-Cost Sustainable":   "#63E6BE",
        "Climate Sensitive":      "#818CF8",
        "Optimisation Target":    "#FBBF24",
        "High HVAC / Renewables": "#F87171",
    }
else:
    cluster_names  = {i: f"Segment {i}" for i in range(k)}
    cluster_colors = {f"Segment {i}": PALETTE[i % len(PALETTE)] for i in range(k)}

original_df["Cluster_Name"] = original_df["Cluster"].map(cluster_names)

# ── Derived financials with smart scaling ──
_ec_col = "Energy Consumption (kWh)"
_ep_col = "Energy Price ($/kWh)"
_sg_col = "Savings_Gap"

if _ec_col in original_df.columns and _ep_col in original_df.columns:
    # If median price looks like a fraction of a cent (data stored as $/Wh or similar), rescale
    median_price = original_df[_ep_col].median()
    price_scale  = 1.0
    if median_price < 0.001:          # stored as $/Wh — multiply by 1000
        price_scale = 1000.0
    elif median_price > 10:           # stored as cents/kWh — divide by 100
        price_scale = 0.01

    original_df["Energy_Cost"] = (
        original_df[_ec_col] * original_df[_ep_col] * price_scale
    )
else:
    original_df["Energy_Cost"] = 0.0

if _sg_col in original_df.columns and _ec_col in original_df.columns:
    # Savings_Gap is a ratio (0–1) — multiply by consumption to get kWh savings, then by price
    if "Energy_Cost" in original_df.columns and original_df["Energy_Cost"].median() > 0:
        original_df["Potential_Savings_Value"] = (
            original_df[_sg_col].clip(0, 1) * original_df["Energy_Cost"]
        )
    else:
        original_df["Potential_Savings_Value"] = (
            original_df[_sg_col] * original_df[_ec_col]
        )
else:
    original_df["Potential_Savings_Value"] = 0.0

# ════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════
if section == "Overview":
    st.markdown('<div class="tag">Portfolio Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Energy Portfolio Overview</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sec-sub">KMeans k={k} · Silhouette {sil_avg:.3f} '
        f'({sil_label(sil_avg)}) · CH {ch_score:,.0f} — live from your data</div>',
        unsafe_allow_html=True,
    )

    total_cost   = original_df["Energy_Cost"].sum()
    total_saving = original_df["Potential_Savings_Value"].sum()
    # Clamp ratio: if savings > cost something is wrong with data scaling, show raw number
    raw_ratio     = (total_saving / total_cost * 100) if total_cost > 0 else 0.0
    savings_ratio = min(raw_ratio, 999.0)   # cap display at 999% — flag if > 100%
    ratio_display = f"{savings_ratio:.1f}%" if raw_ratio <= 100 else f"~{savings_ratio:.0f}%⚠"
    avg_intensity = (original_df["Energy_Intensity"].mean()
                     if "Energy_Intensity" in original_df.columns else 0.0)

    # Row 1 — business KPIs
    c1, c2, c3, c4 = st.columns(4)
    for col_st, lbl, val, sub in zip(
        [c1, c2, c3, c4],
        ["Total Buildings", "Portfolio Energy Cost", "Identifiable Savings", "Savings Ratio"],
        [f"{len(original_df):,}", f"${total_cost/1e6:.1f}M", f"${total_saving/1e6:.1f}M", ratio_display],
        ["Across all segments", "Annualised spend", "Based on efficiency gap", "Of spend recoverable"],
    ):
        with col_st:
            st.markdown(f"""
            <div class="e-card">
                <div class="label">{lbl}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Row 2 — cluster quality KPIs
    q1, q2, q3, q4 = st.columns(4)
    sc = sil_color(sil_avg)
    with q1:
        st.markdown(f"""
        <div class="e-card">
            <div class="label">Silhouette Score</div>
            <div class="value" style="color:{sc}">{sil_avg:.3f}</div>
            <div class="sub">{sil_label(sil_avg)} separation</div>
        </div>""", unsafe_allow_html=True)
    with q2:
        st.markdown(f"""
        <div class="e-card">
            <div class="label">Calinski-Harabasz</div>
            <div class="value">{ch_score:,.0f}</div>
            <div class="sub">Higher = better clusters</div>
        </div>""", unsafe_allow_html=True)
    with q3:
        pc1_v = pca_var_dict.get("PC1", 0)
        pc2_v = pca_var_dict.get("PC2", 0)
        st.markdown(f"""
        <div class="e-card">
            <div class="label">PCA Variance PC1+PC2</div>
            <div class="value">{pc1_v+pc2_v:.1f}%</div>
            <div class="sub">{pca_cumulative:.1f}% cumulative all PCs</div>
        </div>""", unsafe_allow_html=True)
    with q4:
        st.markdown(f"""
        <div class="e-card">
            <div class="label">Active Clusters</div>
            <div class="value">{k}</div>
            <div class="sub">Adjust K in the sidebar</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Data diagnostic (helps catch scaling issues) ──
    with st.expander("🔍 Data Diagnostics — click to inspect raw column stats", expanded=False):
        diag_cols = [c for c in [
            "Energy Consumption (kWh)", "Energy Price ($/kWh)",
            "Savings_Gap", "Energy_Cost", "Potential_Savings_Value",
        ] if c in original_df.columns]
        diag_df = original_df[diag_cols].describe().T[["mean","min","50%","max"]]
        diag_df.columns = ["Mean","Min","Median","Max"]
        st.dataframe(diag_df.style.format("{:.4f}"))
        st.caption(
            "If Energy_Cost looks wrong, check that 'Energy Price ($/kWh)' is in $/kWh "
            "(typical range 0.08–0.35). If values are very small (<0.001) or large (>10), "
            "the app auto-scales but you may want to fix the source data."
        )

    col_bar, col_pie = st.columns([3, 2])
    with col_bar:
        cp = original_df["Cluster_Name"].value_counts()
        colors_bar = [cluster_colors.get(n, "#94A3B8") for n in cp.index]
        fig_bar = go.Figure(go.Bar(
            x=cp.index.tolist(), y=cp.values.tolist(),
            marker=dict(color=colors_bar, cornerradius=8),
            hovertemplate="<b>%{x}</b><br>Buildings: %{y:,}<extra></extra>",
        ))
        fig_bar.update_layout(
            title=dict(text="Buildings per Segment",
                       font=dict(family="Syne,sans-serif", size=15, color="#E8EAF0")),
            height=340, showlegend=False, **PLOT_LAYOUT,
        )
        st.plotly_chart(fig_bar, width="stretch")

    with col_pie:
        colors_pie = [cluster_colors.get(n, "#94A3B8") for n in cp.index]
        fig_pie = go.Figure(go.Pie(
            labels=cp.index.tolist(), values=cp.values.tolist(),
            marker=dict(colors=colors_pie),
            hole=0.55,
            hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            textinfo="none",
        ))
        fig_pie.update_layout(
            title=dict(text="Portfolio Share",
                       font=dict(family="Syne,sans-serif", size=15, color="#E8EAF0")),
            paper_bgcolor="rgba(0,0,0,0)", height=340,
            legend=dict(font=dict(family="DM Mono,monospace", size=10, color="#9CA3AF"),
                        bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=48, b=10),
        )
        st.plotly_chart(fig_pie, width="stretch")

# ════════════════════════════════════════════
# CLUSTER QUALITY
# ════════════════════════════════════════════
elif section == "Cluster Quality":
    st.markdown('<div class="tag">Cluster Validation · Silhouette · CH · PCA · Elbow</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Cluster Quality Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sec-sub">Live metrics for k={k} — move the K slider to compare</div>',
        unsafe_allow_html=True,
    )

    # Score cards
    sc = sil_color(sil_avg)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="e-card">
            <div class="label">Silhouette Score (avg)</div>
            <div class="value" style="color:{sc}">{sil_avg:.4f}</div>
            <div class="sub">−1 → 1 &nbsp;·&nbsp; {sil_label(sil_avg)}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="e-card">
            <div class="label">Calinski-Harabasz Index</div>
            <div class="value">{ch_score:,.0f}</div>
            <div class="sub">Between / within variance ratio</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        pc1_v = pca_var_dict.get("PC1", 0)
        pc2_v = pca_var_dict.get("PC2", 0)
        st.markdown(f"""
        <div class="e-card">
            <div class="label">2D Plot Fidelity (PC1+PC2)</div>
            <div class="value">{pc1_v+pc2_v:.1f}%</div>
            <div class="sub">{pca_cumulative:.1f}% cumulative all PCs</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── PCA Variance Explained chart ──
    st.markdown('<div class="tag">PCA Variance Explained</div>', unsafe_allow_html=True)

    if pca_var_dict:
        pc_names  = list(pca_var_dict.keys())
        pc_vals   = list(pca_var_dict.values())
        cum_vals  = list(pd.Series(pc_vals).cumsum())

        fig_pca = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pca.add_trace(go.Bar(
            x=pc_names, y=pc_vals,
            name="Individual %",
            marker=dict(
                color=[PALETTE[i % len(PALETTE)] for i in range(len(pc_names))],
                cornerradius=6,
            ),
            hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
        ), secondary_y=False)
        fig_pca.add_trace(go.Scatter(
            x=pc_names, y=cum_vals,
            name="Cumulative %",
            line=dict(color="#818CF8", width=2.5, dash="dot"),
            marker=dict(size=6, color="#818CF8"),
            hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>",
        ), secondary_y=True)

        fig_pca.update_layout(
            title=dict(text="Variance Explained per Principal Component",
                       font=dict(family="Syne,sans-serif", size=15, color="#E8EAF0")),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Instrument Sans, sans-serif", color="#6B7280", size=12),
            margin=dict(l=20, r=60, t=48, b=20),
            legend=dict(font=dict(family="DM Mono,monospace", size=11, color="#9CA3AF"),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        )
        fig_pca.update_yaxes(
            title_text="Individual Variance (%)", secondary_y=False,
            gridcolor="rgba(255,255,255,0.05)", color="#6B7280", zeroline=False,
        )
        fig_pca.update_yaxes(
            title_text="Cumulative (%)", secondary_y=True,
            gridcolor="rgba(0,0,0,0)", color="#818CF8", zeroline=False,
        )
        st.plotly_chart(fig_pca, width="stretch")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── K-range analysis ──
    st.markdown('<div class="tag">Optimal K Analysis · k=2 to 10</div>', unsafe_allow_html=True)

    with st.spinner("Computing metrics across k=2–10 (cached after first run)…"):
        k_df = compute_k_range_metrics(X_arr)

    col_e, col_s, col_c = st.columns(3)

    def _k_marker_colors(df, current_k):
        return ["#F87171" if r["k"] == current_k else "#63E6BE" for _, r in df.iterrows()]

    with col_e:
        fig_elbow = go.Figure(go.Scatter(
            x=k_df["k"], y=k_df["inertia"],
            mode="lines+markers",
            line=dict(color="#63E6BE", width=2.5),
            marker=dict(size=8, color=_k_marker_colors(k_df, k)),
            hovertemplate="k=%{x}<br>Inertia: %{y:,.0f}<extra></extra>",
        ))
        fig_elbow.update_layout(
            title=dict(text="Elbow (Inertia)",
                       font=dict(family="Syne,sans-serif", size=14, color="#E8EAF0")),
            height=300, showlegend=False,
            **{**PLOT_LAYOUT, "xaxis": dict(gridcolor="rgba(255,255,255,0.05)",
                                             zeroline=False, title="k", dtick=1)},
        )
        st.plotly_chart(fig_elbow, width="stretch")

    with col_s:
        fig_sil_k = go.Figure(go.Scatter(
            x=k_df["k"], y=k_df["sil"],
            mode="lines+markers",
            line=dict(color="#818CF8", width=2.5),
            marker=dict(size=8, color=_k_marker_colors(k_df, k)),
            hovertemplate="k=%{x}<br>Silhouette: %{y:.4f}<extra></extra>",
        ))
        fig_sil_k.update_layout(
            title=dict(text="Silhouette vs K",
                       font=dict(family="Syne,sans-serif", size=14, color="#E8EAF0")),
            height=300, showlegend=False,
            **{**PLOT_LAYOUT, "xaxis": dict(gridcolor="rgba(255,255,255,0.05)",
                                             zeroline=False, title="k", dtick=1)},
        )
        st.plotly_chart(fig_sil_k, width="stretch")

    with col_c:
        fig_ch_k = go.Figure(go.Scatter(
            x=k_df["k"], y=k_df["ch"],
            mode="lines+markers",
            line=dict(color="#FBBF24", width=2.5),
            marker=dict(size=8, color=_k_marker_colors(k_df, k)),
            hovertemplate="k=%{x}<br>CH: %{y:,.0f}<extra></extra>",
        ))
        fig_ch_k.update_layout(
            title=dict(text="Calinski-Harabasz vs K",
                       font=dict(family="Syne,sans-serif", size=14, color="#E8EAF0")),
            height=300, showlegend=False,
            **{**PLOT_LAYOUT, "xaxis": dict(gridcolor="rgba(255,255,255,0.05)",
                                             zeroline=False, title="k", dtick=1)},
        )
        st.plotly_chart(fig_ch_k, width="stretch")

    # Summary bar
    cur = k_df[k_df["k"] == k].iloc[0]
    best_sil_k = int(k_df.loc[k_df["sil"].idxmax(), "k"])
    best_ch_k  = int(k_df.loc[k_df["ch"].idxmax(), "k"])
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:18px 22px;margin-top:8px;
                font-family:DM Mono,monospace;font-size:0.72rem;color:#6B7280;line-height:2.4;'>
        <span style='color:#E8EAF0;font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;'>
            Current k={k}
        </span><br>
        Silhouette &nbsp;·&nbsp;
            <span style='color:{sil_color(cur["sil"])}'>{cur["sil"]:.4f}</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        CH Index &nbsp;·&nbsp;
            <span style='color:#FBBF24'>{cur["ch"]:,.0f}</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Inertia &nbsp;·&nbsp;
            <span style='color:#63E6BE'>{cur["inertia"]:,.0f}</span><br>
        Best Silhouette at &nbsp;<span style='color:#818CF8'>k={best_sil_k}</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Best CH at &nbsp;<span style='color:#FBBF24'>k={best_ch_k}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Live Silhouette Plot ──
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="tag">Live Silhouette Plot · Per-Building</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Bars sorted within each cluster by coefficient. '
        'Red = negative (potentially misclassified). Dotted line = average.</div>',
        unsafe_allow_html=True,
    )

    # Build per-cluster sorted silhouette data
    traces      = []
    y_offset    = 0
    tick_pos    = []
    tick_lbl    = []
    max_pts     = 4000   # downsample to keep render fast
    total_pts   = len(cluster_labels)
    sample_rate = min(1.0, max_pts / total_pts)

    rng = np.random.default_rng(42)

    for cid in sorted(np.unique(cluster_labels)):
        mask      = cluster_labels == cid
        seg_name  = cluster_names.get(cid, f"Segment {cid}")
        color     = cluster_colors.get(seg_name, PALETTE[cid % len(PALETTE)])
        vals_full = np.sort(sil_smp[mask])

        # Optional downsample
        if sample_rate < 1.0:
            n_keep = max(1, int(len(vals_full) * sample_rate))
            idx    = rng.choice(len(vals_full), size=n_keep, replace=False)
            vals   = np.sort(vals_full[idx])
        else:
            vals = vals_full

        tick_pos.append(y_offset + len(vals) / 2)
        tick_lbl.append(seg_name[:22])

        pos_mask = vals >= 0
        neg_mask = ~pos_mask
        y_coords = np.arange(y_offset, y_offset + len(vals))

        if pos_mask.any():
            traces.append(go.Bar(
                x=vals[pos_mask], y=y_coords[pos_mask],
                orientation="h",
                marker=dict(color=color, opacity=0.85, line=dict(width=0)),
                name=seg_name,
                showlegend=True,
                hovertemplate=f"{seg_name}<br>Silhouette: %{{x:.3f}}<extra></extra>",
            ))
        if neg_mask.any():
            traces.append(go.Bar(
                x=vals[neg_mask], y=y_coords[neg_mask],
                orientation="h",
                marker=dict(color="#F87171", opacity=0.6, line=dict(width=0)),
                name="Negative (misclassified)",
                showlegend=False,
                hovertemplate="Negative silhouette: %{x:.3f}<extra></extra>",
            ))

        y_offset += len(vals) + 10  # gap between clusters

    fig_sil = go.Figure(data=traces)
    fig_sil.add_vline(
        x=sil_avg,
        line=dict(color="#E8EAF0", width=1.5, dash="dot"),
        annotation=dict(
            text=f"  avg={sil_avg:.3f}",
            font=dict(color="#E8EAF0", family="DM Mono,monospace", size=11),
            xanchor="left",
        ),
    )
    fig_sil.add_vline(x=0, line=dict(color="#6B7280", width=1))

    fig_sil.update_layout(
        title=dict(text=f"Silhouette Plot — k={k}",
                   font=dict(family="Syne,sans-serif", size=15, color="#E8EAF0")),
        height=max(480, min(900, y_offset // 2)),
        barmode="overlay",
        bargap=0,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans, sans-serif", color="#6B7280", size=12),
        margin=dict(l=160, r=20, t=48, b=40),
        xaxis=dict(
            title="Silhouette coefficient",
            range=[-0.25, 1.0],
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        yaxis=dict(
            tickvals=tick_pos,
            ticktext=tick_lbl,
            tickfont=dict(family="DM Mono,monospace", size=10, color="#9CA3AF"),
            gridcolor="rgba(0,0,0,0)",
        ),
        legend=dict(font=dict(family="DM Mono,monospace", size=10, color="#9CA3AF"),
                    bgcolor="rgba(0,0,0,0)", tracegroupgap=0),
    )
    st.plotly_chart(fig_sil, width="stretch")

# ════════════════════════════════════════════
# CLUSTER EXPLORER
# ════════════════════════════════════════════
elif section == "Cluster Explorer":
    st.markdown('<div class="tag">Unsupervised Clustering · KMeans</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-header">Cluster Visualisation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sec-sub">PCA 2D embedding — k={k} · '
        f'Silhouette {sil_avg:.3f} · CH {ch_score:,.0f}</div>',
        unsafe_allow_html=True,
    )

    plot_df = X_pca.copy()
    plot_df["Segment"] = plot_df["Cluster"].map(cluster_names)

    fig_scatter = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="Segment",
        color_discrete_map=cluster_colors,
        opacity=0.6,
        hover_data={"PC1": ":.2f", "PC2": ":.2f", "Cluster": False},
    )
    fig_scatter.update_traces(marker=dict(size=5))
    fig_scatter.update_layout(
        height=580,
        legend=dict(font=dict(family="DM Mono,monospace", size=11, color="#9CA3AF"),
                    bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"PCA Cluster Map (k={k})",
                   font=dict(family="Syne,sans-serif", size=15, color="#E8EAF0")),
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig_scatter, width="stretch")

    # Per-segment summary table
    breakdown = original_df.groupby("Cluster_Name").agg(
        Buildings             =("Cluster", "count"),
        Avg_Energy_Cost       =("Energy_Cost", "mean"),
        Avg_Potential_Savings =("Potential_Savings_Value", "mean"),
    ).reset_index()
    breakdown.columns = ["Segment", "Buildings", "Avg Energy Cost ($)", "Avg Potential Savings ($)"]
    breakdown["Avg Energy Cost ($)"]       = breakdown["Avg Energy Cost ($)"].map("${:,.0f}".format)
    breakdown["Avg Potential Savings ($)"] = breakdown["Avg Potential Savings ($)"].map("${:,.0f}".format)
    st.dataframe(breakdown, hide_index=True)

# ════════════════════════════════════════════
# SEGMENT PROFILES
# ════════════════════════════════════════════
elif section == "Segment Profiles":
    st.markdown('<div class="tag">Radar Profiles · Standardised Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Segment Deep-Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Normalised radar fingerprint — σ-deviations from portfolio mean</div>',
                unsafe_allow_html=True)

    seg_options = sorted(original_df["Cluster_Name"].dropna().unique().tolist())
    if not seg_options:
        st.error("No segments found. Check clustering ran correctly.")
        st.stop()

    selected = st.selectbox("Choose a segment", seg_options)

    filtered = original_df[original_df["Cluster_Name"] == selected]
    if filtered.empty:
        st.warning("No buildings found for this segment.")
        st.stop()

    cluster_id = int(filtered["Cluster"].iloc[0])

    centroid_features = [c for c in [
        "Energy_Intensity", "HVAC_Ratio", "Lighting_Ratio", "Renewable_Ratio",
        "Carbon_Load", "Efficiency_Gap", "Savings_Gap", "Grid_Stress", "Temp_Deviation",
    ] if c in original_df.columns]

    if not centroid_features:
        st.error("Feature columns not found in dataset.")
        st.stop()

    centroids = original_df.groupby("Cluster")[centroid_features].mean()
    scaler    = StandardScaler()
    scaled    = pd.DataFrame(
        scaler.fit_transform(centroids),
        columns=centroids.columns,
        index=centroids.index,
    )

    values    = scaled.loc[cluster_id].values.tolist()
    labels_r  = scaled.columns.tolist()
    accent    = cluster_colors.get(selected, "#63E6BE")
    fill_rgba = hex_to_rgba(accent, alpha=0.15)

    col_radar, col_stats = st.columns([3, 2])

    with col_radar:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels_r + [labels_r[0]],
            fill="toself",
            fillcolor=fill_rgba,
            line=dict(color=accent, width=2.5),
            hovertemplate="<b>%{theta}</b><br>%{r:.2f} σ<extra></extra>",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, showticklabels=False,
                                gridcolor="rgba(255,255,255,0.07)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="#6B7280",
                                 tickfont=dict(family="DM Mono,monospace", size=10)),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=460, showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_radar, width="stretch")

    with col_stats:
        n_seg      = len(filtered)
        pct        = n_seg / len(original_df) * 100
        avg_cost   = filtered["Energy_Cost"].mean()
        avg_saving = filtered["Potential_Savings_Value"].mean()

        # Cluster-level silhouette
        seg_mask = cluster_labels == cluster_id
        seg_sil  = float(np.mean(sil_smp[seg_mask]))

        st.markdown(f"""
        <div class="insight-card">
            <div class="i-label">Segment</div>
            <div class="i-value" style="color:{accent}">{selected}</div>
        </div>
        <div class="insight-card">
            <div class="i-label">Portfolio Share</div>
            <div class="i-value">{n_seg:,} buildings</div>
            <div class="i-desc">{pct:.1f}% of total</div>
        </div>
        <div class="insight-card">
            <div class="i-label">Avg Annual Energy Cost</div>
            <div class="i-value">${avg_cost:,.0f}</div>
        </div>
        <div class="insight-card">
            <div class="i-label">Avg Identifiable Savings</div>
            <div class="i-value">${avg_saving:,.0f}</div>
        </div>
        <div class="insight-card">
            <div class="i-label">Cluster Silhouette</div>
            <div class="i-value" style="color:{sil_color(seg_sil)}">{seg_sil:.4f}</div>
            <div class="i-desc">{sil_label(seg_sil)} cohesion</div>
        </div>
        """, unsafe_allow_html=True)

        top_feats = scaled.loc[cluster_id].abs().nlargest(3).index.tolist()
        feat_html = ""
        for f in top_feats:
            v         = scaled.loc[cluster_id, f]
            direction = "↑ High" if v > 0 else "↓ Low"
            feat_html += f"""
            <div class="insight-card">
                <div class="i-label">Dominant Feature</div>
                <div class="i-value">{f.replace("_", " ")}</div>
                <div class="i-desc">{direction} vs portfolio avg ({v:+.2f} σ)</div>
            </div>"""
        st.markdown(feat_html, unsafe_allow_html=True)

# ════════════════════════════════════════════
# RECOMMENDATIONS
# ════════════════════════════════════════════
elif section == "Recommendations":
    st.markdown('<div class="tag">AI Recommendations · Rule-Based · Data-Driven</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Action Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Segment-level interventions ranked by estimated financial impact</div>',
        unsafe_allow_html=True,
    )

    seg_options = sorted(original_df["Cluster_Name"].dropna().unique().tolist())
    tab_all, tab_seg = st.tabs(["All Segments", "By Segment"])

    with tab_all:
        all_recs = []
        for seg in seg_options:
            seg_df = original_df[original_df["Cluster_Name"] == seg]
            for r in build_recommendations(seg_df, original_df):
                all_recs.append({
                    "Segment":     seg,
                    "Priority":    r["priority"],
                    "Action":      r["title"],
                    "Est. Impact": r["impact"],
                    "_v":          r["saving_value"],
                })

        summary_df = (pd.DataFrame(all_recs)
                      .sort_values("_v", ascending=False)
                      .drop(columns=["_v"])
                      .reset_index(drop=True))
        total_opp  = sum(r["_v"] for r in all_recs)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="e-card">
                <div class="label">Total Actions</div>
                <div class="value">{len(summary_df)}</div>
                <div class="sub">Across all segments</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            high_n = len(summary_df[summary_df["Priority"] == "HIGH"])
            st.markdown(f"""<div class="e-card">
                <div class="label">High Priority</div>
                <div class="value">{high_n}</div>
                <div class="sub">Require immediate attention</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="e-card">
                <div class="label">Total Est. Opportunity</div>
                <div class="value">${total_opp/1e6:.1f}M</div>
                <div class="sub">Cumulative across all actions</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.dataframe(summary_df, hide_index=True)

    with tab_seg:
        selected_seg = st.selectbox("Select segment", seg_options, key="rec_seg")
        seg_df  = original_df[original_df["Cluster_Name"] == selected_seg]
        recs    = build_recommendations(seg_df, original_df)
        accent  = cluster_colors.get(selected_seg, "#63E6BE")
        high_n  = sum(1 for r in recs if r["priority"] == "HIGH")

        st.markdown(f"""
        <div style='margin-bottom:18px;padding:14px 18px;background:rgba(255,255,255,0.03);
                    border:1px solid rgba(255,255,255,0.07);border-radius:12px;
                    font-family:DM Mono,monospace;font-size:0.72rem;color:#6B7280;line-height:2;'>
            <span style='color:{accent};font-weight:600;'>{selected_seg}</span>
            &nbsp;·&nbsp; {len(seg_df):,} buildings
            &nbsp;·&nbsp; Avg cost ${seg_df["Energy_Cost"].mean():,.0f}
            &nbsp;·&nbsp;
            <span style='color:#F87171;'>{high_n} HIGH priority action{"s" if high_n != 1 else ""}</span>
        </div>
        """, unsafe_allow_html=True)

        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_recs    = sorted(recs, key=lambda r: priority_order.get(r["priority"], 3))

        for i, rec in enumerate(sorted_recs):
            emoji = "🔴" if rec["priority"] == "HIGH" else ("🟡" if rec["priority"] == "MEDIUM" else "🟢")
            st.markdown(f"""
            <div class="rec-card" style="border-left-color:{rec['color']};">
                <div class="rec-priority" style="color:{rec['color']};">
                    {emoji} &nbsp;{rec['priority']} &nbsp;·&nbsp; Action {i+1} of {len(sorted_recs)}
                </div>
                <div class="rec-title">{rec['title']}</div>
                <div class="rec-body">{rec['body']}</div>
                <div class="rec-impact">💰 {rec['impact']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Saving impact bar chart
        chart_recs = [r for r in recs if r["saving_value"] > 0]
        if chart_recs:
            rec_df = pd.DataFrame([{
                "Action": (r["title"][:40] + "…") if len(r["title"]) > 42 else r["title"],
                "Est. Saving ($M)": round(r["saving_value"] / 1e6, 3),
                "color": r["color"],
            } for r in chart_recs]).sort_values("Est. Saving ($M)", ascending=True)

            fig_rec = go.Figure(go.Bar(
                x=rec_df["Est. Saving ($M)"],
                y=rec_df["Action"],
                orientation="h",
                marker=dict(color=rec_df["color"].tolist(), cornerradius=6),
                hovertemplate="<b>%{y}</b><br>$%{x:.3f}M<extra></extra>",
            ))
            fig_rec.update_layout(
                title=dict(text="Estimated Saving by Action",
                           font=dict(family="Syne,sans-serif", size=14, color="#E8EAF0")),
                height=max(280, len(rec_df) * 52),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Instrument Sans, sans-serif", color="#6B7280", size=12),
                margin=dict(l=20, r=20, t=48, b=20),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            )
            st.plotly_chart(fig_rec, width="stretch")

# ════════════════════════════════════════════
# BUSINESS IMPACT
# ════════════════════════════════════════════
elif section == "Business Impact":
    st.markdown('<div class="tag">Financial Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Business Impact</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">All values derived from your actual dataset</div>',
                unsafe_allow_html=True)

    total_cost   = original_df["Energy_Cost"].sum()
    total_saving = original_df["Potential_Savings_Value"].sum()
    roi_pct      = (total_saving / total_cost * 100) if total_cost > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    for col_st, lbl, val, sub in zip(
        [c1, c2, c3],
        ["Total Energy Spend", "Realisable Savings", "Savings / Cost"],
        [f"${total_cost/1e6:.1f}M", f"${total_saving/1e6:.1f}M", f"{roi_pct:.1f}%"],
        ["All segments", "Efficiency-gap driven", "Improvement headroom"],
    ):
        with col_st:
            st.markdown(f"""<div class="e-card">
                <div class="label">{lbl}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    seg_agg = original_df.groupby("Cluster_Name").agg(
        Energy_Cost=("Energy_Cost", "mean"),
        Savings    =("Potential_Savings_Value", "mean"),
    ).reset_index()

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = go.Figure(go.Bar(
            x=seg_agg["Cluster_Name"], y=seg_agg["Energy_Cost"],
            marker=dict(color=[cluster_colors.get(n, "#94A3B8") for n in seg_agg["Cluster_Name"]],
                        cornerradius=6),
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
        ))
        fig1.update_layout(
            title=dict(text="Avg Energy Cost per Segment",
                       font=dict(family="Syne,sans-serif", size=14, color="#E8EAF0")),
            height=340, showlegend=False, **PLOT_LAYOUT,
        )
        st.plotly_chart(fig1, width="stretch")

    with col_b:
        fig2 = go.Figure(go.Bar(
            x=seg_agg["Cluster_Name"], y=seg_agg["Savings"],
            marker=dict(color=[cluster_colors.get(n, "#94A3B8") for n in seg_agg["Cluster_Name"]],
                        cornerradius=6, opacity=0.8),
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
        ))
        fig2.update_layout(
            title=dict(text="Avg Identifiable Savings per Segment",
                       font=dict(family="Syne,sans-serif", size=14, color="#E8EAF0")),
            height=340, showlegend=False, **PLOT_LAYOUT,
        )
        st.plotly_chart(fig2, width="stretch")

    # Cost vs Savings scatter
    sample_n = min(2000, len(original_df))
    fig3 = px.scatter(
        original_df.sample(sample_n, random_state=1),
        x="Energy_Cost", y="Potential_Savings_Value",
        color="Cluster_Name",
        color_discrete_map=cluster_colors,
        opacity=0.5,
        labels={"Energy_Cost": "Annual Energy Cost ($)",
                "Potential_Savings_Value": "Potential Savings ($)"},
    )
    fig3.update_traces(marker=dict(size=5))
    fig3.update_layout(
        title=dict(text=f"Cost vs Savings (n={sample_n:,} sample)",
                   font=dict(family="Syne,sans-serif", size=14, color="#E8EAF0")),
        height=380,
        legend=dict(font=dict(family="DM Mono,monospace", size=11, color="#9CA3AF"),
                    bgcolor="rgba(0,0,0,0)"),
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig3, width="stretch")

# ════════════════════════════════════════════
# ANOMALY DETECTOR
# ════════════════════════════════════════════
elif section == "Anomaly Detector":
    st.markdown('<div class="tag">Outlier Detection · Z-Score</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Anomaly Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Buildings whose energy cost exceeds Nσ above their segment mean</div>',
                unsafe_allow_html=True)

    threshold = st.slider("σ threshold", min_value=1.5, max_value=4.0, value=2.0, step=0.25)

    anomaly_rows = []
    for seg, grp in original_df.groupby("Cluster_Name"):
        mu  = grp["Energy_Cost"].mean()
        std = grp["Energy_Cost"].std()
        if std == 0:
            continue
        out             = grp[(grp["Energy_Cost"] - mu) / std > threshold].copy()
        out["Z_Score"]  = ((out["Energy_Cost"] - mu) / std).round(2)
        out["Segment"]  = seg
        anomaly_rows.append(out)

    if anomaly_rows:
        anomaly_df = pd.concat(anomaly_rows, ignore_index=True)

        st.markdown(f"""
        <div class="insight-card">
            <div class="i-label">Flagged Buildings</div>
            <div class="i-value">{len(anomaly_df):,}</div>
            <div class="i-desc">{len(anomaly_df)/len(original_df)*100:.2f}% of portfolio · Z > {threshold}</div>
        </div>
        """, unsafe_allow_html=True)

        fig_anom = go.Figure()
        for seg, grp in original_df.groupby("Cluster_Name"):
            normal = grp[~grp.index.isin(anomaly_df.index)]
            fig_anom.add_trace(go.Scatter(
                x=normal["Energy_Cost"], y=normal["Potential_Savings_Value"],
                mode="markers",
                marker=dict(color=cluster_colors.get(seg, "#94A3B8"), size=4, opacity=0.3),
                name=seg,
            ))
        fig_anom.add_trace(go.Scatter(
            x=anomaly_df["Energy_Cost"], y=anomaly_df["Potential_Savings_Value"],
            mode="markers",
            marker=dict(color="#FF4D4D", size=8, symbol="x"),
            name=f"Anomaly (Z > {threshold})",
        ))
        fig_anom.update_layout(
            title=dict(text="Anomalous Buildings Highlighted",
                       font=dict(family="Syne,sans-serif", size=14, color="#E8EAF0")),
            height=440,
            legend=dict(font=dict(family="DM Mono,monospace", size=11, color="#9CA3AF"),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis_title="Energy Cost ($)",
            yaxis_title="Potential Savings ($)",
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig_anom, width="stretch")

        show_cols = [c for c in ["Segment", "Energy_Cost", "Potential_Savings_Value", "Z_Score"]
                     if c in anomaly_df.columns]
        display   = (anomaly_df[show_cols]
                     .sort_values("Z_Score", ascending=False)
                     .reset_index(drop=True))
        display["Energy_Cost"]             = display["Energy_Cost"].map("${:,.0f}".format)
        display["Potential_Savings_Value"] = display["Potential_Savings_Value"].map("${:,.0f}".format)
        display.columns = ["Segment", "Energy Cost ($)", "Potential Savings ($)", "Z-Score"]
        st.dataframe(display, hide_index=True)
    else:
        st.info("No anomalies at this threshold. Try lowering the σ slider.")

# ════════════════════════════════════════════
# EXPORT REPORT
# ════════════════════════════════════════════
elif section == "Export Report":
    st.markdown('<div class="tag">Data Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-header">Export Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Download your segmented portfolio as CSV</div>',
                unsafe_allow_html=True)

    export_cols = [c for c in [
        "Cluster_Name", "Energy Consumption (kWh)", "Energy Price ($/kWh)",
        "Energy_Cost", "Potential_Savings_Value",
        "Energy_Intensity", "HVAC_Ratio", "Lighting_Ratio", "Renewable_Ratio",
        "Carbon_Load", "Efficiency_Gap", "Savings_Gap", "Grid_Stress", "Temp_Deviation",
    ] if c in original_df.columns]

    export_df = original_df[export_cols].copy()
    export_df.rename(columns={"Cluster_Name": "Segment"}, inplace=True)

    seg_filter = st.multiselect(
        "Filter by segment (leave empty = all)",
        sorted(export_df["Segment"].unique()),
    )
    if seg_filter:
        export_df = export_df[export_df["Segment"].isin(seg_filter)]

    st.markdown(f"""
    <div class="insight-card">
        <div class="i-label">Rows to export</div>
        <div class="i-value">{len(export_df):,}</div>
        <div class="i-desc">Columns: {len(export_df.columns)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(export_df.head(50), hide_index=True)
    st.caption("Preview: first 50 rows — full dataset in download")

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="enerlytics_portfolio.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-top:60px;padding-top:24px;
            border-top:1px solid rgba(255,255,255,0.06);
            display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;gap:12px;'>
    <span style='font-family:Syne,sans-serif;font-size:0.85rem;
                 color:#1F2937;font-weight:800;'>⚡ Enerlytics AI</span>
    <span style='font-family:DM Mono,monospace;font-size:0.65rem;
                 color:#374151;letter-spacing:0.08em;'>
        © 2026 · ENERGY SEGMENTATION · KMeans · PCA
    </span>
</div>
""", unsafe_allow_html=True)