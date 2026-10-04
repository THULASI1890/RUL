import hashlib
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="BatteryAI • Reusable Battery Life",
    page_icon="🔋",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "cycle", "voltage", "current", "temperature", "capacity",
    "resistance", "charge_time", "discharge_time", "soh"
]

RUL_MODEL_FILES = ["rul_model.pkl", "rul_model.pkl.b64", "rul_model(1).pkl"]
REUSABLE_MODEL_FILES = ["reusable_model.pkl", "reusable_model.pkl.b64"]

USERS = {
    "admin@batteryai.com": hashlib.sha256("BatteryAI@123".encode()).hexdigest(),
    "thulasi@batteryai.com": hashlib.sha256("Thulasi@123".encode()).hexdigest(),
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "history" not in st.session_state:
    st.session_state.history = []


def verify_login(email, password):
    email = email.strip().lower()
    digest = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(email) == digest


def load_packaged_model(filename):
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} was not found.")

    if filename.endswith(".b64"):\n        raw = base64.b64decode(path.read_text(encoding="utf-8"))\n        obj = joblib.load(__import__("io").BytesIO(raw))\n    else:\n        obj = joblib.load(path)

    # New deployment-safe format:
    # {"model": estimator, "features": [...], "target": "...", ...}
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        features = obj.get("features", FEATURES)
        if list(features) != FEATURES:
            raise ValueError(
                f"{filename} expects a different feature order: {features}"
            )
        return model

    # Backward compatibility for a plain sklearn estimator.
    if hasattr(obj, "predict"):
        expected = getattr(obj, "n_features_in_", len(FEATURES))
        if expected != len(FEATURES):
            raise ValueError(
                f"{filename} expects {expected} features, but the app supplies {len(FEATURES)}."
            )
        return obj

    raise ValueError(f"{filename} does not contain a prediction model.")


@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    rul_errors = []
    rul_model = None

    for filename in RUL_MODEL_FILES:
        try:
            rul_model = load_packaged_model(filename)
            break
        except Exception as exc:
            rul_errors.append(f"{filename}: {type(exc).__name__}: {exc}")

    if rul_model is None:
        raise RuntimeError(
            "No usable RUL model was found.\n" + "\n".join(rul_errors)
        )

    reusable_model = None
    reusable_errors = []

    for filename in REUSABLE_MODEL_FILES:
        try:
            reusable_model = load_packaged_model(filename)
            break
        except Exception as exc:
            reusable_errors.append(f"{filename}: {type(exc).__name__}: {exc}")

    return rul_model, reusable_model, reusable_errors


def login_page():
    st.markdown(
        """
        <style>
        .login-card {
            max-width: 560px;
            margin: 60px auto 20px;
            padding: 35px;
            background: white;
            border-radius: 24px;
            box-shadow: 0 10px 35px rgba(0,0,0,.08);
        }
        .hero { text-align:center; }
        .hero h1 { font-size:42px; margin-bottom:5px; }
        .hero p { color:#6b7280; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="login-card hero"><h1>🔋 BatteryAI</h1>'
        '<p>Reusable Battery Life Prediction System</p></div>',
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 2, 1])
    with center:
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input(
            "Password", type="password", placeholder="Enter your password"
        )

        if st.button("🚀 Login", use_container_width=True):
            if not email or not password:
                st.warning("Please enter both email and password.")
            elif verify_login(email, password):
                st.session_state.authenticated = True
                st.session_state.user_email = email.strip().lower()
                st.rerun()
            else:
                st.error("Invalid email or password.")


def dashboard():
    st.title("🔋 BatteryAI Dashboard")
    st.caption("AI-powered Remaining Useful Life and second-life screening")

    try:
        rul_model, reusable_model, reusable_errors = load_models()
        st.success("✅ AI models loaded successfully.")
    except Exception as exc:
        st.error("❌ AI model is not deployment-ready.")
        st.code(str(exc))
        st.info(
            "Upload the valid rul_model.pkl and reusable_model.pkl generated "
            "from the battery dataset, then redeploy."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RUL Model", type(rul_model).__name__)
    c2.metric("Battery Type", "Li-ion")
    c3.metric("Predictions", len(st.session_state.history))
    c4.metric("Screening Model", "Ready" if reusable_model else "RUL only")

    st.markdown("---")
    st.info(
        "BatteryAI predicts Remaining Useful Life (RUL) and screens a used "
        "lithium-ion battery for possible second-life use."
    )

    if reusable_errors:
        st.warning("Second-life classifier unavailable; RUL prediction remains available.")


def prediction_page():
    st.title("🔋 Battery Prediction")
    st.write("Enter the measured battery parameters.")

    try:
        rul_model, reusable_model, _ = load_models()
    except Exception as exc:
        st.error("Model loading failed.")
        st.code(str(exc))
        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:
        cycle = st.number_input("Cycle", min_value=0.0, value=100.0, step=1.0)
        voltage = st.number_input("Voltage (V)", min_value=0.0, value=3.70)
        current = st.number_input("Current (A)", value=1.50)
    with c2:
        temperature = st.number_input("Temperature (°C)", value=25.0)
        capacity = st.number_input("Capacity (Ah)", min_value=0.0, value=2.50)
        resistance = st.number_input(
            "Resistance (Ω)", min_value=0.0, value=0.0500, format="%.4f"
        )
    with c3:
        charge_time = st.number_input(
            "Charge Time (min)", min_value=0.0, value=120.0
        )
        discharge_time = st.number_input(
            "Discharge Time (min)", min_value=0.0, value=100.0
        )
        soh = st.number_input(
            "State of Health (SOH %)", min_value=0.0, max_value=100.0, value=90.0
        )

    if st.button("🚀 Predict Battery Life", use_container_width=True):
        data = pd.DataFrame(
            [[
                cycle, voltage, current, temperature, capacity,
                resistance, charge_time, discharge_time, soh
            ]],
            columns=FEATURES,
        )

        try:
            rul = max(0.0, float(rul_model.predict(data)[0]))
            st.success(f"### 🔋 Estimated Remaining Useful Life: {rul:.2f} cycles")

            reusable_label = None
            confidence = None

            if reusable_model is not None:
                pred = reusable_model.predict(data)[0]
                reusable_label = "Suitable" if int(pred) == 1 else "Not Suitable"

                if hasattr(reusable_model, "predict_proba"):
                    confidence = float(np.max(reusable_model.predict_proba(data)[0])) * 100

                if reusable_label == "Suitable":
                    st.success("🟢 Second-life screening: **Suitable for further evaluation**")
                else:
                    st.warning("🟠 Second-life screening: **Not recommended by the model**")

                if confidence is not None:
                    st.metric("Model confidence", f"{confidence:.1f}%")

            st.session_state.history.append(
                {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Cycle": cycle,
                    "Voltage": voltage,
                    "SOH": soh,
                    "Predicted RUL": round(rul, 2),
                    "Second-life": reusable_label or "N/A",
                }
            )

            st.info(
                "This is an AI screening result, not a certified electrical or "
                "battery-safety assessment. Physical testing is required before reuse."
            )

        except Exception as exc:
            st.error("Prediction failed.")
            st.code(f"{type(exc).__name__}: {exc}")


def history_page():
    st.title("📜 Prediction History")

    if not st.session_state.history:
        st.info("No predictions have been made in this session.")
        return

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download History",
        df.to_csv(index=False),
        "battery_prediction_history.csv",
        "text/csv",
        use_container_width=True,
    )


def model_info_page():
    st.title("🤖 Model Information")
    st.markdown(
        """
        **RUL task:** Remaining Useful Life regression

        **Second-life task:** Battery reusability classification

        **Algorithm:** Random Forest

        **Input features:**
        cycle, voltage, current, temperature, capacity, resistance,
        charge_time, discharge_time, soh

        The application validates the model feature order before prediction.
        """
    )

    try:
        rul_model, reusable_model, _ = load_models()
        st.success("RUL model is ready.")
        st.code(
            f"RUL estimator: {type(rul_model).__name__}\n"
            f"RUL features: {getattr(rul_model, 'n_features_in_', 'packaged')}\n"
            f"Reusable model: {type(reusable_model).__name__ if reusable_model else 'Not installed'}"
        )
    except Exception as exc:
        st.error("Model is not ready.")
        st.code(str(exc))


def main_app():
    with st.sidebar:
        st.markdown("## 🔋 BatteryAI")
        st.caption(st.session_state.user_email)
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🔋 Battery Prediction", "📜 History", "🤖 Model Info"],
        )

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_email = ""
            st.rerun()

    if page == "🏠 Dashboard":
        dashboard()
    elif page == "🔋 Battery Prediction":
        prediction_page()
    elif page == "📜 History":
        history_page()
    else:
        model_info_page()


if not st.session_state.authenticated:
    login_page()
else:
    main_app()
