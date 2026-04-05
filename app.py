# =============================================================================
# HYBRID VEHICLE ENERGY INTELLIGENCE DASHBOARD
# FILE: app.py  —  Main Streamlit Entry Point
#
# HOW IT WORKS:
#   1. User uploads a vehicle telemetry CSV (same schema as eVED dataset)
#   2. App validates required columns and caches the DataFrame in session state
#   3. If ML models are not yet trained, they are auto-trained on the built-in
#      reference dataset (eVED_181031_week.csv) — runs ONCE, ~5–10 min
#   4. Each feature tab runs its preprocessing pipeline on the uploaded CSV,
#      then calls the saved model to predict every row and display results
#
# HOW TO RUN:
#   .\\venv\\Scripts\\activate
#   streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, sys, subprocess, time
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp{background:#0d1117;color:#e6edf3;}
[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d;}
.card{background:linear-gradient(135deg,#1f2937,#111827);border:1px solid #374151;
      border-radius:12px;padding:18px 22px;margin:8px 0;}
.mval{font-size:2rem;font-weight:700;color:#60a5fa;}
.mlbl{font-size:.85rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;}
.grad{background:linear-gradient(90deg,#60a5fa,#a78bfa);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;font-size:2.2rem;font-weight:800;}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "eVED_181031_week.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

REQUIRED_MODELS = [
    "f1_hvac_model.pkl","f1_scaler.pkl",
    "f2_regen_model.pkl","f2_scaler.pkl",
    "f3_behavior_model.pkl","f3_scaler.pkl",
    "f4_hazard_model.pkl","f4_scaler.pkl",
    "f5_battery_model.pkl","f5_scaler.pkl",
]

def models_exist():
    return all(os.path.exists(os.path.join(MODEL_DIR, m)) for m in REQUIRED_MODELS)

# ── Auto-training ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def auto_train():
    """Train all 5 models from the reference dataset. Runs once per server session."""
    if models_exist():
        return True

    st.markdown("""
    <div style='text-align:center;padding:40px 0'>
    <div style='font-size:3rem'>⚡</div>
    <h2 style='color:#60a5fa'>One-Time Model Training</h2>
    <p style='color:#9ca3af'>Training all 5 ML models on the reference dataset.
    This runs once (~5–10 min) and is never repeated.</p></div>""",
    unsafe_allow_html=True)

    steps = [
        ("features/feature1_hvac_optimizer/preprocessing.py",  "F1 HVAC — Preprocessing"),
        ("features/feature1_hvac_optimizer/ml_model.py",        "F1 HVAC — Training"),
        ("features/feature2_regen_braking/preprocessing.py",   "F2 Regen — Preprocessing"),
        ("features/feature2_regen_braking/ml_model.py",         "F2 Regen — Training"),
        ("features/feature3_driver_behavior/preprocessing.py", "F3 Behavior — Preprocessing"),
        ("features/feature3_driver_behavior/ml_model.py",       "F3 Behavior — Training"),
        ("features/feature4_road_hazard/preprocessing.py",     "F4 Hazard — Preprocessing"),
        ("features/feature4_road_hazard/ml_model.py",           "F4 Hazard — Training"),
        ("features/feature5_battery_health/preprocessing.py",  "F5 Battery — Preprocessing"),
        ("features/feature5_battery_health/ml_model.py",        "F5 Battery — Training"),
    ]

    bar = st.progress(0, text="Starting…")
    placeholders = [st.empty() for _ in steps]

    for i, (rel, label) in enumerate(steps):
        placeholders[i].markdown(f"<span style='color:#fbbf24'>⏳ {label}</span>",
                                  unsafe_allow_html=True)
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, rel.replace("/", os.sep))],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            placeholders[i].markdown(f"<span style='color:#f87171'>❌ {label}</span>",
                                      unsafe_allow_html=True)
            st.error(f"**{label}** failed:\n```\n{result.stderr[-2000:]}\n```")
            st.stop()
        else:
            placeholders[i].markdown(f"<span style='color:#34d399'>✅ {label}</span>",
                                      unsafe_allow_html=True)
        bar.progress((i + 1) / len(steps), text=f"{i+1}/{len(steps)} — {label}")

    st.success("🎉 All models trained! Reloading…")
    st.balloons()
    st.rerun()
    return True

auto_train()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Vehicle Energy Intelligence")
    st.markdown("---")

    # ── CSV Upload ────────────────────────────────────────────────────────────
    st.markdown("#### 📂 Upload Sensor CSV")
    uploaded = st.file_uploader(
        "Upload vehicle telemetry CSV",
        type=["csv"],
        help="Upload a CSV with the same columns as the eVED dataset.",
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded, low_memory=False)
            st.session_state["df_upload"] = df_upload
            st.success(f"✅ Loaded {len(df_upload):,} rows × {df_upload.shape[1]} cols")
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")

    if "df_upload" not in st.session_state:
        st.info("👆 Upload a CSV to run predictions")

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "❄️ HVAC Optimizer", "🔋 Regen Braking",
         "🚗 Driver Behavior", "⚠️ Road Hazard", "🔬 Battery Health",
         "📡 Live Streaming"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("eVED Dataset · Ann Arbor, MI")
    st.caption("Pipeline: Upload CSV → Preprocess → ML Predict")

# ── Helper: require uploaded data ─────────────────────────────────────────────
def need_data():
    if "df_upload" not in st.session_state:
        st.warning("👈 Upload a vehicle telemetry CSV from the sidebar to begin.")
        st.stop()
    return st.session_state["df_upload"]

# ── Helper: styled metric card ────────────────────────────────────────────────
def card(label, value, color="#60a5fa"):
    return f"""<div class='card'><div class='mlbl'>{label}</div>
    <div class='mval' style='color:{color}'>{value}</div></div>"""

# ── Dark Plotly theme ─────────────────────────────────────────────────────────
DARK = dict(paper_bgcolor="#1f2937", plot_bgcolor="#111827",
            font_color="#e6edf3",
            xaxis=dict(gridcolor="#374151"), yaxis=dict(gridcolor="#374151"))

# =============================================================================
# PAGE: HOME
# =============================================================================
if page == "🏠 Home":
    st.markdown('<div class="grad">⚡ Vehicle Energy Intelligence</div>', unsafe_allow_html=True)
    st.markdown("**Real-Time Electric Vehicle Sensor Analytics** — Upload your telemetry CSV to begin.")
    st.markdown("---")

    st.markdown("### 🚀 5 Intelligent Features")
    features_info = [
        ("❄️", "HVAC Energy Optimizer",
         "Predicts total HVAC power draw (AC + Heater Watts) from ambient conditions "
         "enabling proactive climate throttling before steep climbs.",
         "Random Forest Regressor", "#1d4ed8"),
        ("🔋", "Gradient-Aware Regen Braking",
         "Estimates optimal regenerative braking intensity from road gradient + battery SOC "
         "— pre-sets regen level before a descent to maximise energy recovery.",
         "Gradient Boosting Regressor", "#065f46"),
        ("🚗", "Driver Behavior & Eco-Score",
         "Discovers driving archetypes via KMeans clustering, then classifies each "
         "moment as Eco / Moderate / Aggressive with a real-time Eco-Score.",
         "KMeans + Random Forest", "#7c2d12"),
        ("⚠️", "Road Hazard Risk Predictor",
         "Classifies road-segment hazard risk (Low/Medium/High) from OBD + road metadata "
         "alone — no camera, no radar required.",
         "Random Forest Classifier", "#4c1d95"),
        ("🔬", "Battery Health Monitor",
         "Predicts instantaneous battery stress level and long-term degradation risk "
         "from discharge patterns, SOC extremes, and thermal conditions.",
         "XGBoost Classifier", "#1e3a5f"),
    ]
    for icon, title, desc, model, color in features_info:
        st.markdown(f"""
        <div class='card' style='border-left:4px solid {color}'>
            <strong>{icon} {title}</strong>
            &nbsp;<span style='background:{color};color:white;padding:2px 8px;
            border-radius:12px;font-size:.75rem'>{model}</span>
            <p style='margin-top:8px;color:#9ca3af;font-size:.9rem'>{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Upload** a vehicle telemetry CSV from the sidebar (same schema as the eVED dataset)
    2. **Navigate** to any of the 5 feature tabs
    3. The pipeline automatically **preprocesses** & **predicts** on every row
    4. For real-time simulation, open **📡 Live Streaming**
    """)
    if models_exist():
        st.success("✅ All ML models are loaded and ready.")


# =============================================================================
# PAGE: FEATURE 1 — HVAC OPTIMIZER
# =============================================================================
elif page == "❄️ HVAC Optimizer":
    st.markdown("## ❄️ Predictive HVAC Energy Optimizer")
    st.markdown("""**Goal:** Predict total HVAC power draw (W) from ambient + vehicle state
    so the system can proactively throttle HVAC before battery-critical zones.""")
    st.markdown("---")

    df_raw = need_data()

    # Run preprocessing pipeline
    sys.path.insert(0, BASE_DIR)
    from features.feature1_hvac_optimizer.preprocessing import run_preprocessing as f1_prep

    with st.spinner("Running F1 preprocessing pipeline…"):
        try:
            result = f1_prep(df_raw.copy())
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

    df = result["df"]
    model  = joblib.load(os.path.join(MODEL_DIR, "f1_hvac_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "f1_scaler.pkl"))
    features = result["features"]

    X_all = scaler.transform(df[features].values)
    preds = model.predict(X_all)
    preds = np.maximum(preds, 0)
    df = df.copy()
    df["HVAC_Predicted_W"] = preds

    # ── Summary metrics ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Avg HVAC Load", f"{preds.mean():.0f} W", "#fbbf24"), unsafe_allow_html=True)
    c2.markdown(card("Max HVAC Load", f"{preds.max():.0f} W", "#f87171"), unsafe_allow_html=True)
    c3.markdown(card("HVAC > 1500W Rows", f"{(preds>1500).sum():,}", "#f87171"), unsafe_allow_html=True)
    st.markdown("---")

    col_l, col_r = st.columns(2)

    # Distribution of predictions
    with col_l:
        st.markdown("### 📊 HVAC Load Distribution")
        fig = px.histogram(df, x="HVAC_Predicted_W", nbins=60,
                           color_discrete_sequence=["#60a5fa"],
                           labels={"HVAC_Predicted_W": "Predicted HVAC Load (W)"})
        fig.update_layout(**DARK, height=320, margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: OAT vs predicted HVAC
    with col_r:
        st.markdown("### 🌡️ Temperature vs HVAC Load")
        if "OAT[DegC]" in df.columns:
            fig2 = px.scatter(df.sample(min(3000, len(df))),
                              x="OAT[DegC]", y="HVAC_Predicted_W",
                              color="HVAC_Predicted_W",
                              color_continuous_scale="Turbo",
                              labels={"OAT[DegC]":"Outside Temp (°C)",
                                      "HVAC_Predicted_W":"HVAC Load (W)"})
            fig2.update_layout(**DARK, height=320, margin=dict(t=30, b=20),
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    # Timeline
    st.markdown("### 📈 HVAC Load Over Uploaded Data (First 2000 rows)")
    fig3 = go.Figure()
    n = min(2000, len(df))
    fig3.add_trace(go.Scatter(y=preds[:n], mode="lines", name="HVAC Load (W)",
                               line=dict(color="#60a5fa", width=1.5)))
    fig3.add_hline(y=1500, line_dash="dash", line_color="#fbbf24",
                   annotation_text="High Load Threshold (1500W)")
    fig3.update_layout(**DARK, height=300, xaxis_title="Row Index",
                       yaxis_title="HVAC Load (W)", margin=dict(t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    # Recommendations
    st.markdown("### 💡 Recommendations")
    high_pct = (preds > 1500).mean() * 100
    if high_pct > 30:
        st.error(f"🔴 **{high_pct:.0f}%** of driving time has HIGH HVAC load (>1500W). "
                 "Pre-condition the cabin before trips and check HVAC system efficiency.")
    elif high_pct > 10:
        st.warning(f"⚠️ **{high_pct:.0f}%** of data is in high-HVAC territory. "
                   "Consider reducing AC intensity during uphill segments.")
    else:
        st.success(f"✅ Only **{high_pct:.0f}%** of data shows high HVAC load. HVAC usage is efficient.")

    with st.expander("📄 Prediction Table (first 500 rows)"):
        show_cols = [c for c in ["OAT[DegC]","Vehicle Speed[km/h]",
                                   "HV Battery SOC[%]","Gradient","HVAC_Predicted_W"] if c in df.columns]
        st.dataframe(df[show_cols].head(500).round(2), use_container_width=True)


# =============================================================================
# PAGE: FEATURE 2 — REGEN BRAKING
# =============================================================================
elif page == "🔋 Regen Braking":
    st.markdown("## 🔋 Gradient-Aware Regenerative Braking Predictor")
    st.markdown("""**Goal:** Predict HV Battery Current (A) during deceleration.
    Negative = regen charging active. Pre-set regen intensity before descents.""")
    st.markdown("---")

    df_raw = need_data()
    from features.feature2_regen_braking.preprocessing import run_preprocessing as f2_prep

    with st.spinner("Running F2 preprocessing pipeline…"):
        try:
            result = f2_prep(df_raw.copy())
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

    df     = result["df"]
    model  = joblib.load(os.path.join(MODEL_DIR, "f2_regen_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "f2_scaler.pkl"))
    features = result["features"]

    X_all = scaler.transform(df[features].values)
    preds = model.predict(X_all)
    df = df.copy()
    df["Predicted_Current_A"] = preds
    df["Regen_Active"] = preds < 0

    regen_rows = df[df["Regen_Active"]]
    regen_pct  = len(regen_rows) / max(len(df), 1) * 100
    avg_regen  = regen_rows["Predicted_Current_A"].mean() if len(regen_rows) else 0
    energy_est = abs(avg_regen) * 360 / 1000  # kW approx at 360V

    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Regen Active", f"{regen_pct:.1f}%", "#34d399"), unsafe_allow_html=True)
    c2.markdown(card("Avg Regen Current", f"{avg_regen:.1f} A", "#34d399"), unsafe_allow_html=True)
    c3.markdown(card("Est. Regen Power", f"{energy_est:.2f} kW", "#60a5fa"), unsafe_allow_html=True)
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 📊 Predicted Battery Current Distribution")
        fig = px.histogram(df, x="Predicted_Current_A", nbins=80,
                           color_discrete_sequence=["#34d399"],
                           labels={"Predicted_Current_A":"Battery Current (A)"})
        fig.add_vline(x=0, line_dash="dash", line_color="#f87171",
                      annotation_text="Regen Threshold (0A)")
        fig.update_layout(**DARK, height=320, margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### 🏔️ Gradient vs Regen Current")
        if "Gradient" in df.columns:
            s = df.sample(min(3000, len(df)))
            fig2 = px.scatter(s, x="Gradient", y="Predicted_Current_A",
                              color="Predicted_Current_A",
                              color_continuous_scale="RdYlGn_r",
                              labels={"Gradient":"Road Gradient",
                                      "Predicted_Current_A":"Battery Current (A)"})
            fig2.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
            fig2.update_layout(**DARK, height=320, margin=dict(t=30,b=20),
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📈 Battery Current Timeline (First 2000 rows)")
    n = min(2000, len(df))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=preds[:n], mode="lines", name="Battery Current (A)",
                               line=dict(color="#34d399", width=1.5)))
    fig3.add_hline(y=0, line_dash="dash", line_color="#f87171",
                   annotation_text="Regen / Discharge boundary")
    fig3.add_hrect(y0=preds[:n].min(), y1=0, fillcolor="#064e3b", opacity=0.15,
                   annotation_text="Regen Zone")
    fig3.update_layout(**DARK, height=300, xaxis_title="Row Index",
                       yaxis_title="Battery Current (A)", margin=dict(t=20,b=20))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 💡 Recommendations")
    if regen_pct < 5:
        st.warning("⚠️ Very low regen opportunity in this route. Consider routes with more elevation change.")
    elif regen_pct > 25:
        st.success(f"✅ Excellent regen opportunity ({regen_pct:.0f}% of trip). "
                   "Ensure regen intensity is set to maximum for this route.")
    else:
        st.info(f"ℹ️ Moderate regen opportunity ({regen_pct:.0f}%). "
                "Activate 1-pedal driving on downhill segments.")

    with st.expander("📄 Prediction Table (first 500 rows)"):
        show_cols = [c for c in ["Gradient","Vehicle Speed[km/h]","HV Battery SOC[%]",
                                   "Predicted_Current_A","Regen_Active"] if c in df.columns]
        st.dataframe(df[show_cols].head(500).round(2), use_container_width=True)


# =============================================================================
# PAGE: FEATURE 3 — DRIVER BEHAVIOR
# =============================================================================
elif page == "🚗 Driver Behavior":
    st.markdown("## 🚗 Driver Behavior Fingerprinting & Eco-Score")
    st.markdown("""**Goal:** Classify each driving moment as Eco / Moderate / Aggressive
    using KMeans-discovered archetypes + supervised prediction. Real-time Eco-Score (0–100).""")
    st.markdown("---")

    df_raw = need_data()
    from features.feature3_driver_behavior.preprocessing import run_preprocessing as f3_prep

    with st.spinner("Running F3 preprocessing pipeline…"):
        try:
            result = f3_prep(df_raw.copy())
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

    df     = result["df"]
    model  = joblib.load(os.path.join(MODEL_DIR, "f3_behavior_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "f3_scaler.pkl"))
    features = result["features"]

    X_all  = scaler.transform(df[features].values)
    probs  = model.predict_proba(X_all)          # shape: N x 3
    pred_classes = model.predict(X_all)
    eco_scores   = (probs[:, 0] * 100 + probs[:, 1] * 50).round(0).astype(int)

    df = df.copy()
    df["Style_Predicted"] = pred_classes
    df["Eco_Score"]       = eco_scores

    NAMES  = ["Eco", "Moderate", "Aggressive"]
    COLORS = ["#34d399", "#fbbf24", "#f87171"]

    counts = pd.Series(pred_classes).value_counts().sort_index()
    avg_eco = eco_scores.mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Avg Eco-Score", f"{avg_eco:.0f}/100", "#34d399"), unsafe_allow_html=True)
    c2.markdown(card("Eco Moments", f"{counts.get(0,0):,}", "#34d399"), unsafe_allow_html=True)
    c3.markdown(card("Moderate Moments", f"{counts.get(1,0):,}", "#fbbf24"), unsafe_allow_html=True)
    c4.markdown(card("Aggressive Moments", f"{counts.get(2,0):,}", "#f87171"), unsafe_allow_html=True)
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🥧 Driving Style Distribution")
        fig = go.Figure(go.Pie(
            labels=NAMES,
            values=[counts.get(i, 0) for i in range(3)],
            marker=dict(colors=COLORS), hole=0.45,
            textinfo="label+percent"
        ))
        fig.update_layout(paper_bgcolor="#1f2937", font_color="#e6edf3",
                          height=320, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### 📊 Eco-Score Distribution")
        fig2 = px.histogram(x=eco_scores, nbins=50,
                            color_discrete_sequence=["#60a5fa"],
                            labels={"x": "Eco-Score (0–100)"})
        fig2.update_layout(**DARK, height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📈 Eco-Score Timeline (First 2000 rows)")
    n = min(2000, len(df))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=eco_scores[:n], mode="lines", name="Eco-Score",
                               line=dict(color="#34d399", width=1.5)))
    fig3.add_hrect(y0=70, y1=100, fillcolor="#064e3b", opacity=0.15, annotation_text="Eco Zone")
    fig3.add_hrect(y0=0, y1=39,  fillcolor="#7f1d1d", opacity=0.15, annotation_text="Aggressive Zone")
    fig3.update_layout(**DARK, height=300, xaxis_title="Row Index",
                       yaxis_title="Eco-Score", yaxis_range=[0,100],
                       margin=dict(t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 💡 Coaching Recommendations")
    agg_pct = counts.get(2, 0) / max(len(df), 1) * 100
    eco_pct = counts.get(0, 0) / max(len(df), 1) * 100
    if agg_pct > 20:
        st.error(f"🔴 **{agg_pct:.0f}%** of driving is Aggressive. Reduce rapid acceleration, "
                 "avoid high-RPM bursts, and increase following distance.")
    if eco_pct > 60:
        st.success(f"✅ Excellent! **{eco_pct:.0f}%** Eco driving. Battery life and range are well-preserved.")
    else:
        st.warning(f"⚠️ Only **{eco_pct:.0f}%** Eco driving detected. "
                   "Smooth, gradual acceleration/braking can increase Eco-Score.")

    with st.expander("📄 Prediction Table (first 500 rows)"):
        show_cols = [c for c in ["Vehicle Speed[km/h]","Engine RPM[RPM]","Energy_Consumption",
                                   "Style_Predicted","Eco_Score"] if c in df.columns]
        st.dataframe(df[show_cols].head(500).round(2), use_container_width=True)


# =============================================================================
# PAGE: FEATURE 4 — ROAD HAZARD RISK
# =============================================================================
elif page == "⚠️ Road Hazard":
    st.markdown("## ⚠️ Road Hazard Risk Predictor")
    st.markdown("""**Goal:** Classify each road segment as Low / Medium / High hazard risk
    from OBD + road metadata alone — no camera or radar required.""")
    st.markdown("---")

    df_raw = need_data()
    from features.feature4_road_hazard.preprocessing import run_preprocessing as f4_prep

    with st.spinner("Running F4 preprocessing pipeline…"):
        try:
            result = f4_prep(df_raw.copy())
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

    df     = result["df"]
    model  = joblib.load(os.path.join(MODEL_DIR, "f4_hazard_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "f4_scaler.pkl"))
    features = result["features"]

    X_all  = scaler.transform(df[features].values)
    probs  = model.predict_proba(X_all)
    preds  = model.predict(X_all)

    RNAMES  = ["Low", "Medium", "High"]
    RCOLORS = ["#34d399", "#fbbf24", "#f87171"]

    df = df.copy()
    df["Hazard_Predicted"] = preds
    df["Hazard_Label"]     = [RNAMES[p] for p in preds]

    counts = pd.Series(preds).value_counts().sort_index()
    high_pct = counts.get(2, 0) / max(len(df), 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Low Risk", f"{counts.get(0,0):,}", "#34d399"), unsafe_allow_html=True)
    c2.markdown(card("Medium Risk", f"{counts.get(1,0):,}", "#fbbf24"), unsafe_allow_html=True)
    c3.markdown(card("High Risk", f"{counts.get(2,0):,}", "#f87171"), unsafe_allow_html=True)
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🥧 Hazard Risk Distribution")
        fig = go.Figure(go.Pie(
            labels=RNAMES,
            values=[counts.get(i, 0) for i in range(3)],
            marker=dict(colors=RCOLORS), hole=0.45,
            textinfo="label+percent"
        ))
        fig.update_layout(paper_bgcolor="#1f2937", font_color="#e6edf3",
                          height=320, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### 🚀 Speed vs Speed Limit (Excess)")
        if "Speed_Excess" in df.columns:
            fig2 = px.histogram(df, x="Speed_Excess", nbins=50,
                                color="Hazard_Label",
                                color_discrete_map={"Low":"#34d399","Medium":"#fbbf24","High":"#f87171"},
                                labels={"Speed_Excess":"Speed Excess (km/h)"})
            fig2.update_layout(**DARK, height=320, margin=dict(t=20, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📈 Hazard Risk Timeline (First 2000 rows)")
    n = min(2000, len(df))
    fig3 = go.Figure()
    row_colors = [RCOLORS[int(p)] for p in preds[:n]]
    fig3.add_trace(go.Bar(y=preds[:n], marker_color=row_colors, name="Risk Level",
                          hovertext=[RNAMES[int(p)] for p in preds[:n]]))
    fig3.update_layout(**DARK, height=280, xaxis_title="Row Index",
                       margin=dict(t=20, b=20))
    fig3.update_yaxes(tickvals=[0, 1, 2], ticktext=["Low", "Medium", "High"],
                      gridcolor="#374151")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 💡 Safety Recommendations")
    if high_pct > 15:
        st.error(f"🚨 **{high_pct:.0f}%** of route data is HIGH RISK. Reduce speed, "
                 "especially at intersections. Check for icy road conditions (OAT < 4°C).")
    elif high_pct > 5:
        st.warning(f"⚠️ **{high_pct:.0f}%** High-risk segments detected. "
                   "Slow down at identified intersections and steep gradients.")
    else:
        st.success(f"✅ Only **{high_pct:.0f}%** High-risk. Route conditions are generally safe.")

    if "Ice_Risk_Flag" in df.columns and df["Ice_Risk_Flag"].sum() > 0:
        ice_rows = int(df["Ice_Risk_Flag"].sum())
        st.warning(f"🧊 **{ice_rows:,} rows** detected with OAT < 4°C (ice risk). "
                   "Reduce speed and increase following distance in these segments.")

    with st.expander("📄 Prediction Table (first 500 rows)"):
        show_cols = [c for c in ["Vehicle Speed[km/h]","Speed Limit[km/h]","Speed_Excess",
                                   "Intersection","OAT[DegC]","Hazard_Label"] if c in df.columns]
        st.dataframe(df[show_cols].head(500).round(2), use_container_width=True)


# =============================================================================
# PAGE: FEATURE 5 — BATTERY HEALTH
# =============================================================================
elif page == "🔬 Battery Health":
    st.markdown("## 🔬 Battery Health & Degradation Risk Monitor")
    st.markdown("""**Goal:** Classify battery stress level (Low/Medium/High) per timestep
    from discharge patterns, SOC extremes, and thermal conditions — using XGBoost.""")
    st.markdown("---")

    df_raw = need_data()
    from features.feature5_battery_health.preprocessing import run_preprocessing as f5_prep

    with st.spinner("Running F5 preprocessing pipeline…"):
        try:
            result = f5_prep(df_raw.copy())
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

    df     = result["df"]
    model  = joblib.load(os.path.join(MODEL_DIR, "f5_battery_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "f5_scaler.pkl"))
    features = result["features"]

    X_all  = scaler.transform(df[features].values)
    probs  = model.predict_proba(X_all)
    preds  = model.predict(X_all)

    SNAMES  = ["Low", "Medium", "High"]
    SCOLORS = ["#34d399", "#fbbf24", "#f87171"]

    df = df.copy()
    df["Stress_Predicted"] = preds
    df["Stress_Label"]     = [SNAMES[p] for p in preds]

    counts   = pd.Series(preds).value_counts().sort_index()
    high_pct = counts.get(2, 0) / max(len(df), 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Low Stress", f"{counts.get(0,0):,}", "#34d399"), unsafe_allow_html=True)
    c2.markdown(card("Medium Stress", f"{counts.get(1,0):,}", "#fbbf24"), unsafe_allow_html=True)
    c3.markdown(card("High Stress", f"{counts.get(2,0):,}", "#f87171"), unsafe_allow_html=True)
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🥧 Stress Level Distribution")
        fig = go.Figure(go.Pie(
            labels=SNAMES,
            values=[counts.get(i, 0) for i in range(3)],
            marker=dict(colors=SCOLORS), hole=0.45,
            textinfo="label+percent"
        ))
        fig.update_layout(paper_bgcolor="#1f2937", font_color="#e6edf3",
                          height=320, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### ⚡ Battery Current vs SOC (by Stress)")
        if "HV Battery Current[A]" in df.columns and "HV Battery SOC[%]" in df.columns:
            s = df.sample(min(3000, len(df)))
            fig2 = px.scatter(s, x="HV Battery SOC[%]", y="HV Battery Current[A]",
                              color="Stress_Label",
                              color_discrete_map={"Low":"#34d399","Medium":"#fbbf24","High":"#f87171"},
                              labels={"HV Battery SOC[%]":"Battery SOC (%)",
                                      "HV Battery Current[A]":"Battery Current (A)"})
            fig2.update_layout(**DARK, height=320, margin=dict(t=20, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📈 Battery Stress Timeline (First 2000 rows)")
    n   = min(2000, len(df))
    row_colors = [SCOLORS[int(p)] for p in preds[:n]]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(y=preds[:n], marker_color=row_colors, name="Stress Level"))
    fig3.update_layout(**DARK, height=280, xaxis_title="Row Index",
                       margin=dict(t=20, b=20))
    fig3.update_yaxes(tickvals=[0, 1, 2], ticktext=["Low", "Medium", "High"],
                      gridcolor="#374151")
    st.plotly_chart(fig3, use_container_width=True)

    # Feature contributing most to high stress
    if "Battery_Power_Watts" in df.columns:
        avg_power = df[df["Stress_Predicted"]==2]["Battery_Power_Watts"].mean() if counts.get(2,0) else 0
        st.markdown("### 💡 Battery Health Recommendations")
        if high_pct > 20:
            st.error(f"🔴 **{high_pct:.0f}%** High battery stress detected! "
                     f"Average discharge power in stress events: **{avg_power:.0f} W**. "
                     "Avoid rapid acceleration, deep discharges (<20% SOC), and high HVAC load simultaneously.")
        elif high_pct > 5:
            st.warning(f"⚠️ **{high_pct:.0f}%** Medium-to-high stress events. "
                       "Keep SOC between 20–85% for maximum battery longevity.")
        else:
            st.success(f"✅ Battery stress is LOW ({high_pct:.0f}% high-stress events). "
                       "Current driving pattern preserves battery health well.")

    if "SOC_Extremity" in df.columns:
        soc_ext = int(df["SOC_Extremity"].sum())
        if soc_ext > 0:
            st.warning(f"⚠️ **{soc_ext:,}** timesteps with SOC < 15% or > 90% detected. "
                       "Charge before reaching 15% and avoid charging above 85–90% regularly.")

    with st.expander("📄 Prediction Table (first 500 rows)"):
        show_cols = [c for c in ["HV Battery Current[A]","HV Battery SOC[%]",
                                   "HV Battery Voltage[V]","OAT[DegC]",
                                   "Battery_Power_Watts","Stress_Label"] if c in df.columns]
        st.dataframe(df[show_cols].head(500).round(2), use_container_width=True)


# =============================================================================
# PAGE: LIVE STREAMING SIMULATION
# =============================================================================
elif page == "📡 Live Streaming":
    st.markdown("## 📡 Live Vehicle Sensor Stream Simulation")
    st.markdown("""Animates through the uploaded CSV row-by-row to simulate
    real-time OBD-II sensor data processing and prediction.""")
    st.markdown("---")

    df_raw = need_data()

    # Load all 5 models for real-time scoring
    models_ok = models_exist()
    if not models_ok:
        st.error("❌ Models not trained yet. Please restart the app.")
        st.stop()

    m1  = joblib.load(os.path.join(MODEL_DIR, "f1_hvac_model.pkl"))
    sc1 = joblib.load(os.path.join(MODEL_DIR, "f1_scaler.pkl"))
    f1_cols = joblib.load(os.path.join(MODEL_DIR, "f1_features.pkl"))

    m4  = joblib.load(os.path.join(MODEL_DIR, "f4_hazard_model.pkl"))
    sc4 = joblib.load(os.path.join(MODEL_DIR, "f4_scaler.pkl"))
    f4_cols = joblib.load(os.path.join(MODEL_DIR, "f4_features.pkl"))

    m5  = joblib.load(os.path.join(MODEL_DIR, "f5_battery_model.pkl"))
    sc5 = joblib.load(os.path.join(MODEL_DIR, "f5_scaler.pkl"))
    f5_cols = joblib.load(os.path.join(MODEL_DIR, "f5_features.pkl"))

    def safe_predict(model, scaler, cols, row_dict):
        try:
            row = np.array([[row_dict.get(c, 0.0) for c in cols]])
            return model.predict(scaler.transform(row))[0]
        except Exception:
            return None

    col_ctrl, col_settings = st.columns([2, 1])
    with col_settings:
        speed = st.slider("Playback speed (rows/sec)", 1, 30, 5)
        start_row = st.number_input("Start at row", 0, max(0, len(df_raw)-1), 0, step=1)

    with col_ctrl:
        run_stream = st.button("▶ Start Live Stream", type="primary", use_container_width=True)
        stop_stream = st.button("⏹ Stop", use_container_width=True)

    if run_stream:
        st.session_state["streaming"] = True
    if stop_stream:
        st.session_state["streaming"] = False

    if st.session_state.get("streaming", False):
        row_placeholder    = st.empty()
        gauge_placeholder  = st.empty()
        metric_placeholder = st.empty()

        HAZARD_NAMES  = ["🟢 LOW RISK", "🟡 MEDIUM RISK", "🔴 HIGH RISK"]
        STRESS_NAMES  = ["🟢 LOW STRESS", "🟡 MEDIUM STRESS", "🔴 HIGH STRESS"]

        # Feature engineering for live row
        def enrich_row(row):
            d = dict(row)
            # F1 features
            oat  = float(d.get("OAT[DegC]", 15) or 15)
            spd  = float(d.get("Vehicle Speed[km/h]", 0) or 0)
            soc  = float(d.get("HV Battery SOC[%]", 60) or 60)
            grad = float(d.get("Gradient", 0) or 0)
            elev = float(d.get("Elevation Smoothed[m]", 260) or 260)
            d["Thermal_Load_Index"]         = abs(oat - 22) * (1 + spd / 100)
            d["Battery_Gradient_Constraint"]= (100 - soc) * max(grad, 0)
            # F4 features
            sl = float(d.get("Speed Limit[km/h]", 50) or 50)
            d["Speed_Excess"]           = max(0, spd - sl)
            d["Speed_Ratio"]            = spd / (sl + 1e-6)
            d["Ice_Risk_Flag"]           = int(oat < 4)
            inter = float(d.get("Intersection", 0) or 0)
            d["Intersection_Speed_Risk"] = inter * spd / 50
            d["Steep_Gradient_Flag"]     = int(abs(grad) > 0.04)
            d["Absolute Load[%]"]        = d.get("Absolute Load[%]", 0) or 0
            d["Engine RPM[RPM]"]         = d.get("Engine RPM[RPM]", 0) or 0
            d["MAF[g/sec]"]              = d.get("MAF[g/sec]", 0) or 0
            d["Class of Speed Limit"]    = d.get("Class of Speed Limit", 1) or 1
            # F5 features
            cur = float(d.get("HV Battery Current[A]", 0) or 0)
            vol = float(d.get("HV Battery Voltage[V]", 360) or 360)
            ac  = float(d.get("Air Conditioning Power[Watts]", 0) or 0)
            ht  = float(d.get("Heater Power[Watts]", 0) or 0)
            d["HVAC_Total_Watts"]       = ac + ht
            d["Battery_Power_Watts"]    = cur * vol
            d["SOC_Extremity"]          = int(soc < 15 or soc > 90)
            d["Cold_Stress"]            = int(oat < 0)
            d["Heat_Stress"]            = int(oat > 35)
            d["High_Discharge_Flag"]    = int(abs(cur) > 60)
            return d

        delay = 1.0 / speed
        total = len(df_raw)

        for idx in range(int(start_row), total):
            if not st.session_state.get("streaming", False):
                break

            row = df_raw.iloc[idx]
            d   = enrich_row(row)

            hvac_pred   = safe_predict(m1, sc1, f1_cols, d)
            hazard_pred = safe_predict(m4, sc4, f4_cols, d)
            stress_pred = safe_predict(m5, sc5, f5_cols, d)

            hvac_val  = max(0, float(hvac_pred))   if hvac_pred   is not None else 0
            haz_idx   = int(hazard_pred)            if hazard_pred is not None else 0
            stress_idx= int(stress_pred)            if stress_pred is not None else 0

            spd_val  = float(d.get("Vehicle Speed[km/h]", 0) or 0)
            soc_val  = float(d.get("HV Battery SOC[%]", 60)  or 60)
            oat_val  = float(d.get("OAT[DegC]", 0)           or 0)
            cur_val  = float(d.get("HV Battery Current[A]",0) or 0)

            row_placeholder.markdown(f"""
            <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;
            padding:14px 20px;margin-bottom:8px'>
            <span style='color:#9ca3af;font-size:.8rem'>ROW {idx+1:,} / {total:,}</span>
            &nbsp;&nbsp;
            <strong style='color:#60a5fa'>Speed:</strong> {spd_val:.1f} km/h &nbsp;
            <strong style='color:#60a5fa'>SOC:</strong> {soc_val:.1f}% &nbsp;
            <strong style='color:#60a5fa'>OAT:</strong> {oat_val:.1f}°C &nbsp;
            <strong style='color:#60a5fa'>Current:</strong> {cur_val:.1f} A
            </div>""", unsafe_allow_html=True)

            HAZARD_COLORS = ["#34d399", "#fbbf24", "#f87171"]
            STRESS_COLORS = ["#34d399", "#fbbf24", "#f87171"]
            gauge_placeholder.markdown(f"""
            <div style='display:flex;gap:16px'>
            <div class='card' style='flex:1;border-left:4px solid #60a5fa'>
                <div class='mlbl'>HVAC Load</div>
                <div class='mval'>{hvac_val:.0f} W</div>
            </div>
            <div class='card' style='flex:1;border-left:4px solid {HAZARD_COLORS[haz_idx]}'>
                <div class='mlbl'>Road Hazard</div>
                <div class='mval' style='font-size:1.4rem;color:{HAZARD_COLORS[haz_idx]}'>{HAZARD_NAMES[haz_idx]}</div>
            </div>
            <div class='card' style='flex:1;border-left:4px solid {STRESS_COLORS[stress_idx]}'>
                <div class='mlbl'>Battery Stress</div>
                <div class='mval' style='font-size:1.4rem;color:{STRESS_COLORS[stress_idx]}'>{STRESS_NAMES[stress_idx]}</div>
            </div>
            </div>""", unsafe_allow_html=True)

            time.sleep(delay)

        st.session_state["streaming"] = False
        st.success("✅ Stream complete.")
    else:
        st.info("👆 Click **▶ Start Live Stream** to begin the real-time simulation.")
