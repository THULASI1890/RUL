import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import hashlib
from datetime import datetime
import sklearn

st.set_page_config(page_title="BatteryAI • Reusable Battery Life", page_icon="🔋", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = ["rul_model(1).pkl", "rul_model_1_.pkl", "rul_model.pkl"]
FEATURES = ["cycle", "voltage", "current", "temperature", "capacity", "resistance", "charge_time", "discharge_time"]
MAX_RUL = 627.0

st.markdown("""<style>
.stApp{background:#f5f7fb}.login-box{max-width:450px;margin:70px auto;padding:35px;background:white;border-radius:20px;box-shadow:0 10px 35px rgba(0,0,0,.08)}
.login-title{text-align:center;font-size:32px;font-weight:800}.login-subtitle{text-align:center;color:#6b7280;margin-bottom:25px}
.header,.card,.metric-card{background:white;border-radius:18px;box-shadow:0 5px 20px rgba(0,0,0,.05)}.header{padding:20px;margin-bottom:25px}.card{padding:22px;margin-bottom:20px}.metric-card{padding:25px;text-align:center}
.success-box{padding:25px;border-radius:18px;background:#ecfdf5;border:1px solid #10b981;margin-top:20px}.warning-box{padding:25px;border-radius:18px;background:#fff7ed;border:1px solid #f97316;margin-top:20px}.danger-box{padding:25px;border-radius:18px;background:#fef2f2;border:1px solid #ef4444;margin-top:20px}
</style>""", unsafe_allow_html=True)

if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "user_email" not in st.session_state: st.session_state.user_email = ""
if "history" not in st.session_state: st.session_state.history = []

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "admin@batteryai.com": hash_password("BatteryAI@123"),
    "thulasi@batteryai.com": hash_password("Thulasi@123"),
}

def verify_login(email, password):
    email = email.strip().lower()
    return email in USERS and USERS[email] == hash_password(password)

def login_page():
    st.markdown('<div class="login-box"><div class="login-title">🔋 BatteryAI</div><div class="login-subtitle">Reusable Battery Life Prediction System</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("### 🔐 Login")
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        if st.button("🚀 Login", use_container_width=True):
            if not email or not password: st.error("Please enter email and password.")
            elif verify_login(email, password):
                st.session_state.authenticated = True
                st.session_state.user_email = email.strip().lower()
                st.rerun()
            else: st.error("Invalid email or password.")

def model_files():
    found = []
    for name in MODEL_CANDIDATES:
        p = BASE_DIR / name
        if p.is_file(): found.append(p)
    for pattern in ("*.pkl", "*.joblib"):
        for p in sorted(BASE_DIR.glob(pattern)):
            if p not in found: found.append(p)
    return found

def validate_model(model):
    if model is None: return False, "The model file contains None."
    if not hasattr(model, "predict"): return False, "The loaded object has no predict() method and is not a usable estimator."
    if type(model).__name__ == "RandomForestRegressor":
        if not hasattr(model, "estimators_"): return False, "RandomForestRegressor was loaded but has no fitted estimators_. The model is untrained."
        if len(model.estimators_) == 0: return False, "RandomForestRegressor contains zero fitted estimators."
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(FEATURES):
        return False, f"Model expects {model.n_features_in_} features, but this app supplies {len(FEATURES)}."
    return True, "Model is fitted and prediction-ready."

def test_input():
    return pd.DataFrame([{
        "cycle":100.0,"voltage":3.7,"current":1.5,"temperature":25.0,
        "capacity":2.5,"resistance":0.05,"charge_time":120.0,"discharge_time":100.0
    }], columns=FEATURES)

@st.cache_resource(show_spinner=False)
def load_model():
    files = model_files()
    if not files:
        raise RuntimeError("No .pkl or .joblib model file was found. Expected: " + ", ".join(MODEL_CANDIDATES))
    errors = []
    for path in files:
        try:
            model = joblib.load(path)
            valid, message = validate_model(model)
            if not valid:
                errors.append(f"{path.name}: {message}")
                continue
            try:
                prediction = model.predict(test_input())
                if len(prediction) != 1: raise ValueError("predict() did not return one value")
                float(prediction[0])
            except Exception as e:
                errors.append(f"{path.name}: loaded, but test prediction failed: {type(e).__name__}: {e}")
                continue
            return model, path.name
        except Exception as e:
            errors.append(f"{path.name}: {type(e).__name__}: {e}")
    raise RuntimeError("No usable trained model could be loaded.\n\n" + "\n".join("- " + e for e in errors) + "\n\nRuntime scikit-learn: " + sklearn.__version__)

def model_status():
    try:
        model, filename = load_model()
        st.success(f"✅ Trained AI model online: {filename}")
        return model
    except Exception as e:
        st.error("❌ AI model is not deployment-ready.")
        st.code(str(e))
        st.info("If this reports a scikit-learn/joblib compatibility error, retrain the model with the versions in requirements.txt and commit the new model file.")
        return None

def dashboard():
    st.markdown('<div class="header"><h1>🔋 BatteryAI Dashboard</h1><p>AI-powered reusable battery life prediction</p></div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown('<div class="metric-card"><h2>🤖</h2><b>AI Model</b><p>Random Forest</p></div>',unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><h2>🔋</h2><b>Battery</b><p>Li-ion</p></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><h2>📊</h2><b>Predictions</b><p>{len(st.session_state.history)}</p></div>',unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h2>⚡</h2><b>Status</b><p>Online</p></div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🎯 System Purpose")
    st.info("BatteryAI predicts Remaining Useful Life (RUL) of used lithium-ion batteries and helps screen batteries for possible second-life applications.")
    model_status()

def prediction_page():
    st.title("🔋 Battery RUL Prediction")
    st.write("Enter the battery operating parameters below.")
    model = model_status()
    if model is None: st.stop()
    c1,c2=st.columns(2)
    with c1:
        cycle=st.number_input("Cycle",min_value=0.0,value=100.0)
        voltage=st.number_input("Voltage (V)",min_value=0.0,value=3.7)
        current=st.number_input("Current (A)",value=1.5)
        temperature=st.number_input("Temperature (°C)",value=25.0)
    with c2:
        capacity=st.number_input("Capacity (Ah)",min_value=0.0,value=2.5)
        resistance=st.number_input("Resistance (Ω)",min_value=0.0,value=0.05,format="%.4f")
        charge_time=st.number_input("Charge Time (minutes)",min_value=0.0,value=120.0)
        discharge_time=st.number_input("Discharge Time (minutes)",min_value=0.0,value=100.0)
    if st.button("🚀 Predict Battery Life",use_container_width=True):
        data=pd.DataFrame([{"cycle":cycle,"voltage":voltage,"current":current,"temperature":temperature,"capacity":capacity,"resistance":resistance,"charge_time":charge_time,"discharge_time":discharge_time}],columns=FEATURES)
        try:
            rul=float(model.predict(data)[0])
            if not pd.api.types.is_number(rul): raise ValueError("Model returned a non-numeric prediction.")
            rul=max(0.0,rul)
            percentage=min(100,(rul/MAX_RUL)*100)
            if percentage>=70: condition,message,box="Excellent","Battery appears suitable for second-life use.","success-box"
            elif percentage>=40: condition,message,box="Moderate","Battery may be usable with monitoring.","warning-box"
            else: condition,message,box="Poor","Battery may require replacement or further testing.","danger-box"
            st.markdown(f'<div class="{box}"><h2>🔋 Prediction Result</h2><h1>{rul:.2f} Cycles</h1><h3>Battery Condition: {condition}</h3><p>{message}</p></div>',unsafe_allow_html=True)
            st.session_state.history.append({"Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Cycle":cycle,"Voltage":voltage,"Capacity":capacity,"Predicted RUL":round(rul,2),"Condition":condition})
        except Exception as e:
            st.error("❌ Prediction failed.")
            st.code(f"{type(e).__name__}: {e}")

def history_page():
    st.title("📜 Prediction History")
    if not st.session_state.history: st.info("No predictions have been made yet."); return
    df=pd.DataFrame(st.session_state.history)
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download Prediction History",df.to_csv(index=False),"battery_prediction_history.csv","text/csv",use_container_width=True)

def model_info_page():
    st.title("🤖 Model Information")
    st.markdown("""### AI/ML Model
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

### Deployment Diagnostics
The app checks that the model file exists, can be deserialized, has predict(), is fitted, expects eight features, and can produce a test prediction.

> This application is a screening aid and should not be treated as battery safety certification.""")
    try:
        model,filename=load_model()
        st.success(f"Model ready: {filename}")
        st.code(f"Object: {type(model).__module__}.{type(model).__name__}\nRuntime scikit-learn: {sklearn.__version__}\nExpected features: {getattr(model,'n_features_in_','not reported')}")
    except Exception as e:
        st.error("Model is not deployment-ready.")
        st.code(str(e))

def main_app():
    with st.sidebar:
        st.markdown("## 🔋 BatteryAI")
        st.caption(f"Logged in as\n{st.session_state.user_email}")
        st.markdown("---")
        page=st.radio("Navigation",["🏠 Dashboard","🔋 Battery Prediction","📜 Prediction History","🤖 Model Info"])
        st.markdown("---")
        if st.button("🚪 Logout",use_container_width=True):
            st.session_state.authenticated=False
            st.session_state.user_email=""
            st.rerun()
    if page=="🏠 Dashboard": dashboard()
    elif page=="🔋 Battery Prediction": prediction_page()
    elif page=="📜 Prediction History": history_page()
    elif page=="🤖 Model Info": model_info_page()

if not st.session_state.authenticated: login_page()
else: main_app()