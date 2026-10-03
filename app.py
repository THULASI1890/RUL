import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="BatteryAI • Reusable Battery Life",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODEL_CANDIDATES = [
    "rul_model(1).pkl",
    "rul_model_1_.pkl",
    "rul_model.pkl",
]

FEATURES = [
    "cycle",
    "voltage",
    "current",
    "temperature",
    "capacity",
    "resistance",
    "charge_time",
    "discharge_time",
]

# RUL is measured in cycles. Training targets range from 0 to ~627.
MAX_RUL = 627.0

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(0, 229, 180, .12), transparent 28%),
        radial-gradient(circle at 100% 10%, rgba(45, 130, 255, .12), transparent 30%),
        #071018;
    color: #edf5f7;
}

.block-container { max-width: 1450px; padding-top: 1.4rem; padding-bottom: 3rem; }

[data-testid="stSidebar"] { background: #09141d; border-right: 1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] * { color: #dbe8eb; }

.hero {
    padding: 28px 32px;
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 26px;
    background: linear-gradient(135deg, rgba(13, 34, 43, .92), rgba(8, 22, 31, .80));
    box-shadow: 0 24px 70px rgba(0,0,0,.28);
    margin-bottom: 22px;
}
.eyebrow { color: #51e0be; font-size: 12px; font-weight: 800; letter-spacing: 1.7px; text-transform: uppercase; }
.hero h1 { font-size: clamp(32px, 5vw, 54px); line-height: 1.04; margin: 8px 0 10px; letter-spacing: -1.8px; }
.hero p { color: #9fb4bd; max-width: 820px; font-size: 15px; line-height: 1.65; margin: 0; }

.card {
    background: rgba(14, 28, 37, .78);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 14px 45px rgba(0,0,0,.18);
    height: 100%;
}
.card-title { font-size: 14px; font-weight: 800; color: #eef7f8; margin-bottom: 6px; }
.card-subtitle { color: #8fa6b0; font-size: 12px; line-height: 1.5; }

.kpi {
    background: linear-gradient(145deg, rgba(18,37,47,.92), rgba(10,24,33,.92));
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 18px;
    padding: 18px;
    min-height: 112px;
}
.kpi-label { color: #8ea5ae; font-size: 12px; font-weight: 600; }
.kpi-value { color: #f4fbfc; font-size: 27px; font-weight: 800; margin-top: 9px; }
.kpi-note { color: #5eddbd; font-size: 11px; margin-top: 4px; }

.section { font-size: 23px; font-weight: 800; letter-spacing: -.5px; margin: 25px 0 13px; }
.mini-label { color: #89a1ab; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.big-rul { font-size: 68px; line-height: 1; font-weight: 900; letter-spacing: -3px; margin: 8px 0; }
.good { color: #51e0be; }
.warn { color: #ffd166; }
.bad  { color: #ff6b6b; }

.pill {
    display: inline-block; padding: 7px 11px; border-radius: 999px;
    background: rgba(81,224,190,.10); border: 1px solid rgba(81,224,190,.22);
    color: #70e8ca; font-size: 11px; font-weight: 800;
}
.warning {
    background: rgba(255, 209, 102, .08);
    border: 1px solid rgba(255, 209, 102, .20);
    color: #f4d68a; border-radius: 15px; padding: 13px 15px;
    font-size: 12px; line-height: 1.55;
}

.stButton > button, .stDownloadButton > button {
    width: 100%;
    border-radius: 13px;
    border: 1px solid rgba(81,224,190,.25);
    background: linear-gradient(135deg, #16b894, #0e8f78);
    color: white; font-weight: 800; min-height: 48px;
    box-shadow: 0 10px 30px rgba(16, 184, 148, .16);
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #67ead0; transform: translateY(-1px); }

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 16px; padding: 12px;
}
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 20px; }
[data-testid="stDataFrame"] { border-radius: 15px; overflow: hidden; }
hr { border-color: rgba(255,255,255,.07); }
footer { visibility: hidden; }
input { color-scheme: dark; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_model():
    for name in MODEL_CANDIDATES:
        path = BASE_DIR / name
        if path.exists():
            return joblib.load(path)
    # last resort: any .pkl next to app.py
    pkls = sorted(BASE_DIR.glob("*.pkl"))
    if pkls:
        return joblib.load(pkls[0])
    raise FileNotFoundError(
        "No model file found. Place your .pkl (e.g. 'rul_model(1).pkl') in the same folder as app.py."
    )


try:
    model = load_model()
    model_loaded = True
    model_error = None
    N_TREES = getattr(model, "n_estimators", "—")
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)
    N_TREES = "—"

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# HELPERS
# ============================================================
def classify_rul(rul: float):
    """Bands are based on RUL as a share of the max RUL seen in training."""
    pct = rul / MAX_RUL * 100
    if pct >= 50:
        return "Excellent", "good", "High second-life potential"
    if pct >= 20:
        return "Moderate", "warn", "Requires further evaluation"
    return "Low", "bad", "Detailed inspection recommended"


def add_history(row: dict):
    st.session_state.history.insert(0, row)
    st.session_state.history = st.session_state.history[:10]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🔋 BatteryAI")
    st.caption("Reusable Battery Intelligence Platform")
    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        ["Overview", "Battery Prediction", "Prediction History", "Model Info"],
    )

    st.markdown("---")
    if model_loaded:
        st.markdown("🟢 **Model online**")
        st.caption(f"Random Forest Regressor • {N_TREES} trees")
    else:
        st.markdown("🔴 **Model offline**")
        st.caption("Place the .pkl file beside app.py")


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">AI-POWERED BATTERY INTELLIGENCE</div>
    <h1>Predict. Evaluate. Reuse.</h1>
    <p>
        Estimate the remaining useful life (in cycles) of a used lithium-ion battery
        from electrical, thermal and cycle-level measurements, and turn the
        prediction into a clear second-life screening result.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown('<div class="section">System overview</div>', unsafe_allow_html=True)

    cards = [
        ("MODEL", "Random Forest", f"{N_TREES} decision trees"),
        ("INPUTS", f"{len(FEATURES)} features", "Electrical + thermal"),
        ("OUTPUT", "RUL (cycles)", "Regression prediction"),
        ("USE CASE", "Second-life", "Reuse screening"),
    ]
    for col, (label, value, note) in zip(st.columns(4), cards):
        with col:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section">How BatteryAI works</div>', unsafe_allow_html=True)
    a, b = st.columns([1.15, 1])

    with a:
        st.markdown("""
        <div class="card">
            <div class="card-title">01 — Capture battery condition</div>
            <div class="card-subtitle">
                Enter cycle count, voltage, current, temperature, capacity,
                resistance, charge time and discharge time.
            </div>
            <br>
            <div class="card-title">02 — Run the trained model</div>
            <div class="card-subtitle">
                The Random Forest regression model processes the eight features
                and produces the RUL estimate.
            </div>
            <br>
            <div class="card-title">03 — Interpret the result</div>
            <div class="card-subtitle">
                The dashboard converts the prediction into a simple condition
                band and second-life screening indicator.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="card">
            <div class="card-title">♻️ Why second-life batteries?</div>
            <div class="card-subtitle">
                A battery that is no longer ideal for its original application
                may still have useful capacity for less demanding stationary
                applications.
            </div>
            <br>
            <div class="warning">
                AI prediction is a screening tool, not a battery safety
                certification. Physical inspection and appropriate electrical
                testing are required before real-world reuse.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Model loading failed: {model_error}")


# ============================================================
# PREDICTION
# ============================================================
elif page == "Battery Prediction":
    st.markdown('<div class="section">Battery condition input</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model is not available. Check the model filename and dependencies.")
        st.code(model_error)
        st.stop()

    st.markdown("""
    <div class="card">
        <div class="card-title">Enter measured battery parameters</div>
        <div class="card-subtitle">
            Use values from your battery dataset or test equipment. The fields
            below match the feature order expected by the model.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    r1 = st.columns(4)
    with r1[0]:
        cycle = st.number_input("Cycle", min_value=0.0, value=100.0, step=1.0)
    with r1[1]:
        voltage = st.number_input("Voltage (V)", min_value=0.0, value=3.70, step=0.01, format="%.2f")
    with r1[2]:
        current = st.number_input("Current (A)", value=2.00, step=0.10, format="%.2f")
    with r1[3]:
        temperature = st.number_input("Temperature (°C)", value=25.0, step=0.5, format="%.1f")

    r2 = st.columns(4)
    with r2[0]:
        capacity = st.number_input("Capacity (Ah)", min_value=0.0, value=2.50, step=0.01, format="%.2f")
    with r2[1]:
        resistance = st.number_input("Resistance (Ω)", min_value=0.0, value=0.050, step=0.001, format="%.3f")
    with r2[2]:
        charge_time = st.number_input("Charge time (min)", min_value=0.0, value=120.0, step=1.0)
    with r2[3]:
        discharge_time = st.number_input("Discharge time (min)", min_value=0.0, value=100.0, step=1.0)

    st.markdown("")
    predict = st.button("⚡ ANALYZE BATTERY WITH AI")

    if predict:
        data = pd.DataFrame([{
            "cycle": cycle,
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "capacity": capacity,
            "resistance": resistance,
            "charge_time": charge_time,
            "discharge_time": discharge_time,
        }], columns=FEATURES)

        # Real input checks
        notes = []
        if cycle > MAX_RUL:
            notes.append(
                f"Cycle ({cycle:.0f}) is above the range the model was trained on "
                f"(up to ~{MAX_RUL:.0f}). Random Forests cannot extrapolate, so treat this result with caution."
            )
        if discharge_time > charge_time * 1.5:
            notes.append("Discharge time is much longer than charge time. Please double-check the values.")

        with st.spinner("Running Random Forest inference…"):
            time.sleep(0.4)
            try:
                rul = max(0.0, float(model.predict(data)[0]))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        condition, css_class, suitability = classify_rul(rul)
        timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
        rul_pct = float(np.clip(rul / MAX_RUL, 0, 1))

        add_history({
            "Time": timestamp,
            "RUL (cycles)": round(rul, 2),
            "Condition": condition,
            "Cycle": cycle,
            "Capacity (Ah)": capacity,
        })

        for n in notes:
            st.warning(n)

        st.markdown('<div class="section">AI assessment</div>', unsafe_allow_html=True)
        left, middle, right = st.columns([1.25, 1, 1])

        with left:
            st.markdown(f"""
            <div class="card">
                <div class="mini-label">PREDICTED REMAINING USEFUL LIFE</div>
                <div class="big-rul {css_class}">{rul:.0f}</div>
                <span class="pill">CYCLES • MODEL OUTPUT</span>
                <p style="color:#8fa6b0;font-size:12px;margin-top:14px;">
                    Direct prediction returned by the Random Forest model.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with middle:
            st.markdown(f"""
            <div class="card">
                <div class="mini-label">BATTERY CONDITION</div>
                <div class="big-rul {css_class}" style="font-size:32px;letter-spacing:-1px;">{condition}</div>
                <p style="color:#8fa6b0;font-size:12px;">Screening band based on predicted RUL.</p>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown(f"""
            <div class="card">
                <div class="mini-label">SECOND-LIFE SCREEN</div>
                <div class="big-rul {css_class}" style="font-size:32px;letter-spacing:-1px;">{suitability}</div>
                <p style="color:#8fa6b0;font-size:12px;">{timestamp}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section">Battery health view</div>', unsafe_allow_html=True)
        g1, g2 = st.columns([1.05, 1])

        with g1:
            with st.container(border=True):
                st.markdown("**RUL indicator**")
                st.caption(f"Shown against the maximum RUL seen in training (~{MAX_RUL:.0f} cycles).")
                st.progress(rul_pct)
                m1, m2 = st.columns(2)
                m1.metric("Predicted RUL", f"{rul:.0f} cycles")
                m2.metric("Share of max RUL", f"{rul_pct * 100:.0f}%")

        with g2:
            with st.container(border=True):
                st.markdown("**Input snapshot**")
                st.caption("Values sent to the model.")
                snapshot = pd.DataFrame({
                    "Parameter": ["Cycle", "Voltage", "Current", "Temperature",
                                  "Capacity", "Resistance", "Charge time", "Discharge time"],
                    "Value": [f"{cycle:.0f}", f"{voltage:.2f} V", f"{current:.2f} A",
                              f"{temperature:.1f} °C", f"{capacity:.2f} Ah",
                              f"{resistance:.3f} Ω", f"{charge_time:.0f} min",
                              f"{discharge_time:.0f} min"],
                })
                st.dataframe(snapshot, hide_index=True)

        st.markdown('<div class="section">Engineering note</div>', unsafe_allow_html=True)
        if condition == "Excellent":
            st.success("The model predicts a relatively high RUL. This can support further "
                       "evaluation for second-life use, subject to physical testing.")
        elif condition == "Moderate":
            st.warning("The model predicts a moderate RUL. Additional electrical and safety "
                       "testing should be performed before reuse.")
        else:
            st.error("The model predicts a low RUL. Treat this as a screening signal and "
                     "perform detailed inspection before considering reuse.")

        report = f"""BATTERYAI — PREDICTION REPORT
========================================
Generated: {timestamp}

MODEL
-----
Algorithm: Random Forest Regressor
Trees: {N_TREES}
Features: {len(FEATURES)}

INPUTS
------
Cycle: {cycle}
Voltage: {voltage} V
Current: {current} A
Temperature: {temperature} C
Capacity: {capacity} Ah
Resistance: {resistance} Ohm
Charge time: {charge_time} min
Discharge time: {discharge_time} min

RESULT
------
Predicted RUL: {rul:.2f} cycles
Condition band: {condition}
Second-life screening: {suitability}

IMPORTANT
---------
This is an AI/ML screening prediction. It is not a safety certification
and should not replace physical battery inspection, electrical testing,
thermal testing or engineering approval.
"""
        st.download_button(
            "📄 DOWNLOAD PREDICTION REPORT",
            data=report,
            file_name="batteryai_prediction_report.txt",
            mime="text/plain",
        )


# ============================================================
# HISTORY
# ============================================================
elif page == "Prediction History":
    st.markdown('<div class="section">Recent predictions</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="card">
            <div class="card-title">No predictions yet</div>
            <div class="card-subtitle">
                Run a battery analysis and your latest results will appear here.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, hide_index=True)

        st.download_button(
            "⬇️ Export history as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="batteryai_prediction_history.csv",
            mime="text/csv",
        )
        if st.button("Clear prediction history"):
            st.session_state.history = []
            st.rerun()


# ============================================================
# MODEL INFO
# ============================================================
elif page == "Model Info":
    st.markdown('<div class="section">Model information</div>', unsafe_allow_html=True)

    kpis = [
        ("ALGORITHM", "Random Forest", "Regression"),
        ("ESTIMATORS", str(N_TREES), "Decision trees"),
        ("FEATURES", str(len(FEATURES)), "Expected inputs"),
    ]
    for col, (label, value, note) in zip(st.columns(3), kpis):
        with col:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section">Expected model inputs</div>', unsafe_allow_html=True)
    info = pd.DataFrame({
        "Feature": FEATURES,
        "Role": ["Battery cycle count", "Electrical potential", "Operating current",
                 "Thermal condition", "Available capacity", "Internal resistance",
                 "Charging duration", "Discharging duration"],
    })
    st.dataframe(info, hide_index=True)

    if model_loaded and hasattr(model, "feature_importances_"):
        st.markdown('<div class="section">Feature importance</div>', unsafe_allow_html=True)
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        st.bar_chart(imp)
        top, share = imp.index[0], imp.iloc[0] * 100
        if share > 80:
            st.info(
                f"The model relies mostly on **{top}** ({share:.1f}% of importance). "
                "Changing the other inputs will have little effect on the predicted RUL."
            )

    st.markdown("""
    <div class="warning">
        <strong>Important:</strong> the app sends the eight values to the model in this exact
        feature order. If you retrain the model with different columns, update FEATURES and
        the input form accordingly.
    </div>
    """, unsafe_allow_html=True)
