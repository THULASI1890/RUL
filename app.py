
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BatteryAI | Reusable Battery Life",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0b1728 50%, #07111f 100%);
        color: white;
    }

    .main-title {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: #ffffff;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #a9bdd5;
        margin-bottom: 35px;
    }

    .hero {
        padding: 30px;
        border-radius: 25px;
        background: linear-gradient(
            135deg,
            rgba(0, 255, 170, 0.12),
            rgba(0, 140, 255, 0.10)
        );
        border: 1px solid rgba(0,255,170,0.25);
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 22px;
        border-radius: 18px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        text-align: center;
        min-height: 130px;
    }

    .metric-title {
        color: #9fb4cb;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: white;
        font-size: 30px;
        font-weight: 800;
    }

    .section-title {
        font-size: 27px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .result-box {
        padding: 30px;
        border-radius: 22px;
        text-align: center;
        margin-top: 20px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
    }

    .result-number {
        font-size: 55px;
        font-weight: 900;
        margin: 5px;
    }

    .result-label {
        color: #a9bdd5;
        font-size: 17px;
    }

    .status-good {
        font-size: 26px;
        font-weight: 800;
        color: #35f2a0;
    }

    .status-medium {
        font-size: 26px;
        font-weight: 800;
        color: #ffd166;
    }

    .status-bad {
        font-size: 26px;
        font-weight: 800;
        color: #ff6b6b;
    }

    .info-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 15px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = "rul_model(1).pkl"

    try:
        model = joblib.load(model_path)
        return model, None

    except Exception as e:
        return None, str(e)


model, model_error = load_model()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="main-title">
🔋 BatteryAI
</div>

<div class="subtitle">
AI / ML Model for Predicting Reusable Battery Life
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ERROR
# ============================================================

if model is None:

    st.error("❌ Model could not be loaded.")

    st.code(model_error)

    st.info("""
    Make sure your project folder contains:

    app.py
    rul_model(1).pkl
    requirements.txt
    """)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔋 BatteryAI")

st.sidebar.markdown("""
### Navigation

Use this application to estimate:

- 🔋 Battery Health
- ⏳ Remaining Useful Life
- ♻️ Second-Life Suitability
- 📊 Battery condition
- 🤖 AI prediction

---

### Model

**Algorithm:** Random Forest Regressor

**Trees:** 200

**Inputs:** 8 battery parameters
""")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "🔮 RUL Prediction",
        "📊 Battery Analysis",
        "ℹ️ About Project"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">⚡ Battery Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">🤖 ML Algorithm</div>
            <div class="metric-value">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">🌳 Trees</div>
            <div class="metric-value">200</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">📥 Input Features</div>
            <div class="metric-value">8</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">♻️ Application</div>
            <div class="metric-value">2nd Life</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔋 What does this system do?</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">

        ### 🔬 Battery Prediction

        The system uses machine learning to estimate the
        **Remaining Useful Life (RUL)** of a used lithium-ion battery.

        The prediction is based on:

        - Charge/discharge cycles
        - Voltage
        - Current
        - Temperature
        - Capacity
        - Internal resistance
        - Charge time
        - Discharge time

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">

        ### ♻️ Second-Life Applications

        Batteries with useful remaining life can potentially
        be considered for applications such as:

        ☀️ Solar energy storage

        💡 Emergency lighting

        🔋 Portable power systems

        🏠 Backup energy storage

        </div>
        """, unsafe_allow_html=True)

    st.info(
        "💡 Go to **RUL Prediction** from the sidebar to test your battery."
    )


# ============================================================
# RUL PREDICTION
# ============================================================

elif page == "🔮 RUL Prediction":

    st.markdown(
        '<div class="section-title">🔮 AI Battery Life Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the battery operating parameters below. "
        "The trained Random Forest model will estimate the remaining useful life."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        cycle = st.number_input(
            "🔄 Cycle",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=1.0,
            help="Number of charge/discharge cycles"
        )

    with col2:

        voltage = st.number_input(
            "⚡ Voltage (V)",
            min_value=0.0,
            max_value=100.0,
            value=3.7,
            step=0.01
        )

    with col3:

        current = st.number_input(
            "🔌 Current (A)",
            min_value=-100.0,
            max_value=100.0,
            value=2.0,
            step=0.1
        )

    with col4:

        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=-50.0,
            max_value=100.0,
            value=25.0,
            step=0.5
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:

        capacity = st.number_input(
            "🔋 Capacity (Ah)",
            min_value=0.0,
            max_value=1000.0,
            value=2.5,
            step=0.01
        )

    with col6:

        resistance = st.number_input(
            "〰️ Resistance (Ω)",
            min_value=0.0,
            max_value=100.0,
            value=0.05,
            step=0.001,
            format="%.3f"
        )

    with col7:

        charge_time = st.number_input(
            "⏱️ Charge Time (min)",
            min_value=0.0,
            max_value=10000.0,
            value=120.0,
            step=1.0
        )

    with col8:

        discharge_time = st.number_input(
            "⏱️ Discharge Time (min)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=1.0
        )

    st.markdown("")

    # --------------------------------------------------------
    # INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "cycle": [cycle],
        "voltage": [voltage],
        "current": [current],
        "temperature": [temperature],
        "capacity": [capacity],
        "resistance": [resistance],
        "charge_time": [charge_time],
        "discharge_time": [discharge_time]
    })

    st.markdown("### 📋 Battery Input Summary")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict = st.button(
        "🚀 ANALYZE BATTERY WITH AI",
        use_container_width=True,
        type="primary"
    )

    if predict:

        # ----------------------------------------------
        # AI ANALYSIS ANIMATION
        # ----------------------------------------------

        progress = st.progress(0)

        status = st.empty()

        steps = [
            "🔍 Reading battery parameters...",
            "⚙️ Processing battery characteristics...",
            "🧠 Running Random Forest model...",
            "📊 Estimating remaining useful life...",
            "♻️ Evaluating second-life potential...",
            "✅ Analysis completed!"
        ]

        for i, message in enumerate(steps):

            status.markdown(f"### {message}")

            progress.progress(
                int((i + 1) / len(steps) * 100)
            )

            time.sleep(0.35)

        # ----------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------

        try:

            prediction = model.predict(input_data)

            rul = float(prediction[0])

            # Avoid negative RUL display
            rul_display = max(0, rul)

            # ------------------------------------------
            # STATUS
            # ------------------------------------------

            if rul_display >= 80:

                battery_status = "🟢 EXCELLENT"
                status_class = "status-good"
                suitability = "HIGH"
                recommendation = (
                    "Battery shows strong remaining useful life "
                    "and may be considered for second-life applications."
                )

            elif rul_display >= 50:

                battery_status = "🟡 MODERATE"
                status_class = "status-medium"
                suitability = "MEDIUM"
                recommendation = (
                    "Battery has moderate remaining useful life. "
                    "Further testing is recommended before reuse."
                )

            else:

                battery_status = "🔴 LOW"
                status_class = "status-bad"
                suitability = "LOW"
                recommendation = (
                    "Battery has relatively low predicted remaining "
                    "useful life. Detailed safety and performance testing "
                    "should be performed before reuse."
                )

            # ------------------------------------------
            # RESULTS
            # ------------------------------------------

            st.markdown("---")

            st.markdown(
                '<div class="section-title">🤖 AI Prediction Result</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.markdown(f"""
                <div class="result-box">

                    <div class="result-label">
                    ⏳ PREDICTED RUL
                    </div>

                    <div class="result-number">
                    {rul_display:.2f}
                    </div>

                    <div class="result-label">
                    Model Output
                    </div>

                </div>
                """, unsafe_allow_html=True)

            with col2:

                st.markdown(f"""
                <div class="result-box">

                    <div class="result-label">
                    🔋 BATTERY CONDITION
                    </div>

                    <div class="{status_class}">
                    {battery_status}
                    </div>

                </div>
                """, unsafe_allow_html=True)

            with col3:

                st.markdown(f"""
                <div class="result-box">

                    <div class="result-label">
                    ♻️ SECOND-LIFE SUITABILITY
                    </div>

                    <div class="{status_class}">
                    {suitability}
                    </div>

                </div>
                """, unsafe_allow_html=True)

            # ------------------------------------------
            # GAUGE
            # ------------------------------------------

            st.markdown("### 📊 Remaining Useful Life Indicator")

            gauge_value = min(max(rul_display, 0), 100)

            st.progress(
                gauge_value / 100
            )

            st.caption(
                f"Predicted RUL: {rul_display:.2f}"
            )

            # ------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------

            st.markdown("### ♻️ AI Assessment")

            st.info(recommendation)

            # ------------------------------------------
            # BATTERY PARAMETERS CHART
            # ------------------------------------------

            st.markdown("### 📈 Battery Parameters")

            chart_data = pd.DataFrame({
                "Parameter": [
                    "Cycle",
                    "Voltage",
                    "Current",
                    "Temperature",
                    "Capacity",
                    "Resistance",
                    "Charge Time",
                    "Discharge Time"
                ],
                "Value": [
                    cycle,
                    voltage,
                    current,
                    temperature,
                    capacity,
                    resistance,
                    charge_time,
                    discharge_time
                ]
            })

            st.bar_chart(
                chart_data.set_index("Parameter")
            )

            # ------------------------------------------
            # DOWNLOAD REPORT
            # ------------------------------------------

            report = f"""
BATTERYAI - AI BATTERY LIFE PREDICTION REPORT
==============================================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

BATTERY INPUTS
--------------

Cycle: {cycle}
Voltage: {voltage} V
Current: {current} A
Temperature: {temperature} °C
Capacity: {capacity} Ah
Resistance: {resistance} Ohm
Charge Time: {charge_time} minutes
Discharge Time: {discharge_time} minutes

AI PREDICTION
-------------

Predicted RUL: {rul_display:.2f}

Battery Condition:
{battery_status}

Second-Life Suitability:
{suitability}

AI Assessment:
{recommendation}

MODEL
-----

Algorithm: Random Forest Regressor
Number of Trees: 200
Input Features: 8

NOTE
----

This prediction is an AI/ML estimate and should not be
used as the sole basis for battery safety or engineering decisions.
Physical inspection and appropriate battery testing are required.
"""

            st.download_button(
                label="📥 Download Prediction Report",
                data=report,
                file_name="battery_prediction_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as e:

            st.error("❌ Prediction failed.")

            st.code(str(e))


# ============================================================
# BATTERY ANALYSIS
# ============================================================

elif page == "📊 Battery Analysis":

    st.markdown(
        '<div class="section-title">📊 Battery Parameter Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The trained model uses eight battery characteristics "
        "to estimate Remaining Useful Life."
    )

    feature_info = pd.DataFrame({
        "Feature": [
            "cycle",
            "voltage",
            "current",
            "temperature",
            "capacity",
            "resistance",
            "charge_time",
            "discharge_time"
        ],
        "Meaning": [
            "Charge/discharge cycle count",
            "Battery voltage",
            "Operating current",
            "Battery temperature",
            "Battery capacity",
            "Internal resistance",
            "Time required for charging",
            "Time during discharge"
        ]
    })

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🧠 Model Information")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Algorithm",
            "Random Forest Regressor"
        )

        st.metric(
            "Number of Trees",
            "200"
        )

    with col2:

        st.metric(
            "Input Features",
            "8"
        )

        st.metric(
            "Prediction Type",
            "Regression"
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ## 🔋 AI/ML Model for Predicting Reusable Battery Life

    This project uses **Artificial Intelligence and Machine Learning**
    to estimate the remaining useful life of used lithium-ion batteries.

    ### 🎯 Main Objective

    The objective is to determine whether a battery may still have
    useful remaining life for potential **second-life applications**.

    ### 🤖 Machine Learning Model

    The uploaded trained model is a:

    **Random Forest Regressor**

    with:

    - 200 decision trees
    - 8 input features
    - Regression output

    ### 📥 Input Parameters

    The model analyzes:

    1. Cycle
    2. Voltage
    3. Current
    4. Temperature
    5. Capacity
    6. Resistance
    7. Charge Time
    8. Discharge Time

    ### ♻️ Possible Applications

    Used batteries with sufficient performance may potentially be
    evaluated for:

    - Solar energy storage
    - Emergency lighting
    - Portable power systems
    - Backup energy storage
    - Stationary energy storage

    ### ⚠️ Important

    The application provides an ML-based prediction.

    It does **not** replace professional battery safety testing,
    physical inspection, electrical characterization, or certification.
    """)

    st.success(
        "🔋 BatteryAI — Turning battery data into intelligent reuse insights."
    )
