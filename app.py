import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import hashlib
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BatteryAI • Reusable Battery Life",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    "rul_model(1).pkl",
    "rul_model_1_.pkl",
    "rul_model.pkl"
]

FEATURES = [
    "cycle",
    "voltage",
    "current",
    "temperature",
    "capacity",
    "resistance",
    "charge_time",
    "discharge_time"
]

MAX_RUL = 627.0


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #f5f7fb;
}

.login-box {
    max-width: 450px;
    margin: 70px auto;
    padding: 35px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.08);
}

.login-title {
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 5px;
}

.login-subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 25px;
}

.header {
    padding: 20px;
    border-radius: 18px;
    background: white;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
}

.success-box {
    padding: 25px;
    border-radius: 18px;
    background: #ecfdf5;
    border: 1px solid #10b981;
    margin-top: 20px;
}

.warning-box {
    padding: 25px;
    border-radius: 18px;
    background: #fff7ed;
    border: 1px solid #f97316;
    margin-top: 20px;
}

.danger-box {
    padding: 25px;
    border-radius: 18px;
    background: #fef2f2;
    border: 1px solid #ef4444;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# PASSWORD HASHING
# =========================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# =========================================================
# USER CREDENTIALS
# =========================================================
# DEMO ONLY
#
# For production, move these values to st.secrets.
# =========================================================

USERS = {
    "admin@batteryai.com":
        hash_password("BatteryAI@123"),

    "thulasi@batteryai.com":
        hash_password("Thulasi@123")
}


# =========================================================
# AUTHENTICATION
# =========================================================

def verify_login(email, password):

    email = email.strip().lower()

    if email not in USERS:
        return False

    password_hash = hash_password(password)

    return USERS[email] == password_hash


# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    st.markdown("""
    <div class="login-box">

        <div class="login-title">
            🔋 BatteryAI
        </div>

        <div class="login-subtitle">
            Reusable Battery Life Prediction System
        </div>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown("### 🔐 Login")

        email = st.text_input(
            "Email",
            placeholder="Enter your email"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )

        login = st.button(
            "🚀 Login",
            use_container_width=True
        )

        if login:

            if not email or not password:

                st.error("Please enter email and password.")

            elif verify_login(email, password):

                st.session_state.authenticated = True
                st.session_state.user_email = email

                st.success("Login successful!")

                st.rerun()

            else:

                st.error("Invalid email or password.")


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():

    for name in MODEL_CANDIDATES:

        path = BASE_DIR / name

        if path.exists():

            return joblib.load(path)

    pkls = list(BASE_DIR.glob("*.pkl"))

    if pkls:

        return joblib.load(pkls[0])

    raise FileNotFoundError(
        "No .pkl model file found."
    )


# =========================================================
# DASHBOARD
# =========================================================

def dashboard():

    st.markdown("""
    <div class="header">

        <h1>🔋 BatteryAI Dashboard</h1>

        <p>
        AI-powered reusable battery life prediction
        </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown("""
        <div class="metric-card">

        <h2>🤖</h2>

        <b>AI Model</b>

        <p>Random Forest</p>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">

        <h2>🔋</h2>

        <b>Battery</b>

        <p>Li-ion</p>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="metric-card">

        <h2>📊</h2>

        <b>Predictions</b>

        <p>{}</p>

        </div>
        """.format(len(st.session_state.history)),
                    unsafe_allow_html=True)

    with col4:

        st.markdown("""
        <div class="metric-card">

        <h2>⚡</h2>

        <b>Status</b>

        <p>Online</p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🎯 System Purpose")

    st.info(
        """
        BatteryAI predicts the Remaining Useful Life (RUL) of
        used lithium-ion batteries and helps identify whether
        a battery may be suitable for second-life applications.
        """
    )


# =========================================================
# BATTERY PREDICTION
# =========================================================

def prediction_page():

    st.title("🔋 Battery RUL Prediction")

    st.write(
        "Enter the battery operating parameters below."
    )

    try:

        model = load_model()

        st.success("AI Model Loaded Successfully")

    except Exception as e:

        st.error(f"Model loading failed: {e}")

        return

    st.markdown("### ⚙️ Battery Parameters")

    col1, col2 = st.columns(2)

    with col1:

        cycle = st.number_input(
            "Cycle",
            min_value=0.0,
            value=100.0
        )

        voltage = st.number_input(
            "Voltage (V)",
            min_value=0.0,
            value=3.7
        )

        current = st.number_input(
            "Current (A)",
            value=1.5
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=25.0
        )

    with col2:

        capacity = st.number_input(
            "Capacity (Ah)",
            min_value=0.0,
            value=2.5
        )

        resistance = st.number_input(
            "Resistance (Ω)",
            min_value=0.0,
            value=0.05,
            format="%.4f"
        )

        charge_time = st.number_input(
            "Charge Time (minutes)",
            min_value=0.0,
            value=120.0
        )

        discharge_time = st.number_input(
            "Discharge Time (minutes)",
            min_value=0.0,
            value=100.0
        )

    st.markdown("")

    predict = st.button(
        "🚀 Predict Battery Life",
        use_container_width=True
    )

    if predict:

        data = pd.DataFrame(
            [{
                "cycle": cycle,
                "voltage": voltage,
                "current": current,
                "temperature": temperature,
                "capacity": capacity,
                "resistance": resistance,
                "charge_time": charge_time,
                "discharge_time": discharge_time
            }],
            columns=FEATURES
        )

        try:

            rul = float(model.predict(data)[0])

            rul = max(0.0, rul)

            # ---------------------------------------------
            # CONDITION CLASSIFICATION
            # ---------------------------------------------

            percentage = min(
                100,
                (rul / MAX_RUL) * 100
            )

            if percentage >= 70:

                condition = "Excellent"
                message = "Battery appears suitable for second-life use."
                box = "success-box"

            elif percentage >= 40:

                condition = "Moderate"
                message = "Battery may be usable with monitoring."
                box = "warning-box"

            else:

                condition = "Poor"
                message = "Battery may require replacement or further testing."
                box = "danger-box"

            # ---------------------------------------------
            # RESULT
            # ---------------------------------------------

            st.markdown(
                f"""
                <div class="{box}">

                    <h2>🔋 Prediction Result</h2>

                    <h1>{rul:.2f} Cycles</h1>

                    <h3>Battery Condition: {condition}</h3>

                    <p>{message}</p>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ---------------------------------------------
            # HISTORY
            # ---------------------------------------------

            record = {

                "Time":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Cycle": cycle,

                "Voltage":
                    voltage,

                "Capacity":
                    capacity,

                "Predicted RUL":
                    round(rul, 2),

                "Condition":
                    condition
            }

            st.session_state.history.append(record)

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# =========================================================
# HISTORY
# =========================================================

def history_page():

    st.title("📜 Prediction History")

    if not st.session_state.history:

        st.info(
            "No predictions have been made yet."
        )

        return

    df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    csv = df.to_csv(index=False)

    st.download_button(
        "⬇️ Download Prediction History",
        csv,
        "battery_prediction_history.csv",
        "text/csv",
        use_container_width=True
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

def model_info_page():

    st.title("🤖 Model Information")

    st.markdown("""
    ### AI/ML Model

    **Algorithm:** Random Forest Regressor

    **Task:** Regression

    **Prediction:** Remaining Useful Life (RUL)

    ### Input Features

    - Cycle
    - Voltage
    - Current
    - Temperature
    - Capacity
    - Resistance
    - Charge Time
    - Discharge Time

    ### Output

    The model predicts the estimated number of
    remaining battery cycles.

    ### Second-Life Screening

    The predicted RUL is converted into a simple
    battery-condition indicator:

    - 🟢 Excellent
    - 🟠 Moderate
    - 🔴 Poor

    > This application is a screening aid and should
    > not be treated as a battery safety certification.
    """)


# =========================================================
# MAIN APPLICATION
# =========================================================

def main_app():

    # ---------------------------------------------
    # SIDEBAR
    # ---------------------------------------------

    with st.sidebar:

        st.markdown("## 🔋 BatteryAI")

        st.caption(
            f"Logged in as\n{st.session_state.user_email}"
        )

        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "🏠 Dashboard",
                "🔋 Battery Prediction",
                "📜 Prediction History",
                "🤖 Model Info"
            ]
        )

        st.markdown("---")

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.authenticated = False
            st.session_state.user_email = ""

            st.rerun()

    # ---------------------------------------------
    # PAGE ROUTING
    # ---------------------------------------------

    if page == "🏠 Dashboard":

        dashboard()

    elif page == "🔋 Battery Prediction":

        prediction_page()

    elif page == "📜 Prediction History":

        history_page()

    elif page == "🤖 Model Info":

        model_info_page()


# =========================================================
# APPLICATION ENTRY
# =========================================================

if not st.session_state.authenticated:

    login_page()

else:

    main_app()
