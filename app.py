import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Battery RUL AI",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS - DARK MODERN UI
# =========================================================

st.markdown("""
<style>

    /* ---------- Main Background ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(0, 220, 180, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(50, 120, 255, 0.08),
                transparent 30%
            ),
            #080b12;
        color: #f5f7fa;
    }


    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #0d111a;
        border-right: 1px solid #202938;
    }

    section[data-testid="stSidebar"] * {
        color: #dce3ed;
    }


    /* ---------- Header ---------- */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #8d99aa;
        font-size: 16px;
        margin-bottom: 25px;
    }


    /* ---------- Cards ---------- */

    .card {
        background: linear-gradient(
            145deg,
            #111722,
            #0d121b
        );

        border: 1px solid #202938;
        border-radius: 18px;
        padding: 24px;

        box-shadow:
            0 10px 30px rgba(0,0,0,0.25);

        margin-bottom: 18px;
    }


    .prediction-card {
        background:
            linear-gradient(
                135deg,
                rgba(0, 214, 170, 0.15),
                rgba(30, 80, 150, 0.12)
            );

        border: 1px solid rgba(0, 214, 170, 0.35);
        border-radius: 22px;
        padding: 30px;

        text-align: center;

        box-shadow:
            0 15px 45px rgba(0,0,0,0.3);
    }


    .prediction-label {
        color: #8d99aa;
        font-size: 15px;
        margin-bottom: 8px;
    }


    .prediction-value {
        color: #00d6aa;
        font-size: 48px;
        font-weight: 800;
    }


    .prediction-unit {
        color: #aab4c3;
        font-size: 16px;
    }


    /* ---------- Section Titles ---------- */

    .section-title {
        font-size: 21px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }


    /* ---------- Status ---------- */

    .status-good {
        background: rgba(0, 214, 170, 0.10);
        border: 1px solid rgba(0, 214, 170, 0.3);
        color: #00d6aa;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }


    .status-medium {
        background: rgba(255, 190, 70, 0.10);
        border: 1px solid rgba(255, 190, 70, 0.3);
        color: #ffbe46;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }


    .status-low {
        background: rgba(255, 80, 90, 0.10);
        border: 1px solid rgba(255, 80, 90, 0.3);
        color: #ff6670;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }


    /* ---------- Buttons ---------- */

    .stButton > button {
        width: 100%;
        height: 52px;

        border-radius: 12px;
        border: 1px solid #00d6aa;

        background: linear-gradient(
            90deg,
            #00b894,
            #00d6aa
        );

        color: #06110e;
        font-size: 16px;
        font-weight: 700;

        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 8px 25px rgba(0,214,170,0.25);
    }


    /* ---------- Inputs ---------- */

    div[data-baseweb="input"] {
        background: #111722 !important;
        border-radius: 10px;
    }

    div[data-baseweb="input"] input {
        color: #ffffff !important;
    }


    /* ---------- Divider ---------- */

    hr {
        border-color: #202938;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #667085;
        font-size: 13px;
        margin-top: 40px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load("rul_model.pkl")


try:

    model = load_model()

except Exception:

    st.error(
        "❌ Unable to load rul_model.pkl. "
        "Place the model file in the same folder as app.py."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:25px;
            font-weight:800;
            margin-bottom:25px;
        ">
        🔋 Battery AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    page = st.radio(
        "",
        [
            "🏠 Dashboard",
            "🔮 RUL Prediction",
            "ℹ️ About Model"
        ]
    )

    st.divider()

    st.markdown(
        """
        **AI Engine**

        Random Forest Regressor

        **Prediction**

        Remaining Useful Life

        **Unit**

        Battery Cycles
        """
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">Battery Health Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered Remaining Useful Life prediction system'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="card">
                <div style="color:#8d99aa;">
                    MODEL
                </div>

                <div style="
                    font-size:24px;
                    font-weight:700;
                    margin-top:8px;
                ">
                    Random Forest
                </div>

                <div style="
                    color:#00d6aa;
                    margin-top:6px;
                ">
                    ● Active
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">
                <div style="color:#8d99aa;">
                    INPUT FEATURES
                </div>

                <div style="
                    font-size:24px;
                    font-weight:700;
                    margin-top:8px;
                ">
                    8
                </div>

                <div style="
                    color:#8d99aa;
                    margin-top:6px;
                ">
                    Battery parameters
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="card">
                <div style="color:#8d99aa;">
                    OUTPUT
                </div>

                <div style="
                    font-size:24px;
                    font-weight:700;
                    margin-top:8px;
                ">
                    RUL
                </div>

                <div style="
                    color:#8d99aa;
                    margin-top:6px;
                ">
                    Remaining cycles
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <div class="card">

        <div class="section-title">
        🔋 How the system works
        </div>

        <p style="color:#9aa5b5;line-height:1.8;">

        The system uses battery operating parameters such as
        cycle count, voltage, current, temperature, capacity,
        resistance, charge time and discharge time.

        These parameters are passed to a trained
        <b>Random Forest Regression model</b> to estimate
        the battery's Remaining Useful Life (RUL).

        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# RUL PREDICTION PAGE
# =========================================================

elif page == "🔮 RUL Prediction":

    st.markdown(
        '<div class="main-title">RUL Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter the current battery operating parameters'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # INPUT CARD
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🔧 Battery Parameters</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    with col1:

        cycle = st.number_input(
            "Cycle",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        voltage = st.number_input(
            "Voltage (V)",
            min_value=0.0,
            value=3.70,
            step=0.01
        )

        current = st.number_input(
            "Current (A)",
            value=1.00,
            step=0.01
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=25.0,
            step=0.1
        )


    with col2:

        capacity = st.number_input(
            "Capacity (Ah)",
            min_value=0.0,
            value=2.00,
            step=0.01
        )

        resistance = st.number_input(
            "Resistance (Ohm)",
            min_value=0.0,
            value=0.050,
            step=0.001,
            format="%.3f"
        )

        charge_time = st.number_input(
            "Charge Time (hours)",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

        discharge_time = st.number_input(
            "Discharge Time (hours)",
            min_value=0.0,
            value=2.0,
            step=0.1
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    predict = st.button(
        "🔮  Predict Remaining Useful Life",
        use_container_width=True
    )


    if predict:

        # Validation

        if voltage <= 0:

            st.error("Voltage must be greater than 0.")
            st.stop()

        if capacity <= 0:

            st.error("Capacity must be greater than 0.")
            st.stop()

        if resistance < 0:

            st.error("Resistance cannot be negative.")
            st.stop()

        if charge_time <= 0:

            st.error("Charge time must be greater than 0.")
            st.stop()

        if discharge_time <= 0:

            st.error("Discharge time must be greater than 0.")
            st.stop()


        # -------------------------------------------------
        # MODEL INPUT
        # -------------------------------------------------

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


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        with st.spinner("Analyzing battery condition..."):

            prediction = model.predict(input_data)[0]


        prediction = max(0, prediction)


        st.markdown("<br>", unsafe_allow_html=True)


        # -------------------------------------------------
        # MAIN RESULT
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    ESTIMATED REMAINING USEFUL LIFE
                </div>

                <div class="prediction-value">
                    {prediction:.2f}
                </div>

                <div class="prediction-unit">
                    battery cycles
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)


        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        if prediction <= 50:

            st.markdown(
                """
                <div class="status-low">
                    🔴 LOW ESTIMATED RUL — The model estimates relatively few
                    remaining cycles.
                </div>
                """,
                unsafe_allow_html=True
            )

        elif prediction <= 200:

            st.markdown(
                """
                <div class="status-medium">
                    🟡 MODERATE ESTIMATED RUL — The model estimates a
                    moderate number of remaining cycles.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="status-good">
                    🟢 HIGHER ESTIMATED RUL — The model estimates a
                    relatively higher number of remaining cycles.
                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("<br>", unsafe_allow_html=True)


        # -------------------------------------------------
        # TWO COLUMN RESULT
        # -------------------------------------------------

        result1, result2 = st.columns(2)


        with result1:

            st.markdown(
                """
                <div class="card">

                <div class="section-title">
                📋 Input Summary
                </div>

                """,
                unsafe_allow_html=True
            )

            summary = pd.DataFrame({

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
                    f"{cycle:.0f}",
                    f"{voltage:.2f} V",
                    f"{current:.2f} A",
                    f"{temperature:.1f} °C",
                    f"{capacity:.2f} Ah",
                    f"{resistance:.3f} Ω",
                    f"{charge_time:.2f} h",
                    f"{discharge_time:.2f} h"
                ]

            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("</div>", unsafe_allow_html=True)


        with result2:

            st.markdown(
                f"""
                <div class="card">

                <div class="section-title">
                🤖 AI Prediction
                </div>

                <p style="color:#8d99aa;">
                Model
                </p>

                <p style="font-size:20px;font-weight:700;">
                Random Forest Regressor
                </p>

                <p style="color:#8d99aa;">
                Estimated RUL
                </p>

                <p style="
                    font-size:32px;
                    font-weight:800;
                    color:#00d6aa;
                ">
                {prediction:.2f} cycles
                </p>

                <p style="color:#8d99aa;line-height:1.6;">
                This value represents the model's estimated
                remaining useful life based on the supplied
                battery parameters.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# ABOUT MODEL
# =========================================================

elif page == "ℹ️ About Model":

    st.markdown(
        '<div class="main-title">About the AI Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Technical information about the RUL prediction system'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">

        <div class="section-title">
        🤖 Machine Learning Model
        </div>

        <p style="color:#9aa5b5;line-height:1.8;">

        This application uses a <b>Random Forest Regressor</b>
        trained to predict the Remaining Useful Life (RUL)
        of a battery.

        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.subheader("📥 Input Features")

    features = pd.DataFrame({

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

        "Description": [
            "Battery operating cycle",
            "Battery voltage",
            "Battery current",
            "Operating temperature",
            "Battery capacity",
            "Internal resistance",
            "Charging duration",
            "Discharging duration"
        ]

    })

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "The prediction should be interpreted within the range "
        "and characteristics of the data used to train the model."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🔋 Battery AI • Remaining Useful Life Prediction

    <br>

    Powered by Machine Learning

    </div>
    """,
    unsafe_allow_html=True
)
