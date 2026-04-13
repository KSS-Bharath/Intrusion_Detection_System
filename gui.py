import streamlit as st
import pickle
import numpy as np
from alert import trigger_alert

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Intrusion Detection System",
    layout="centered"
)

st.title("🚨 Intrusion Detection System")
st.write("Binary IDS using XGBoost (Demo GUI)")

# ===============================
# Load model & preprocessing
# ===============================
@st.cache_resource
def load_resources():
    with open("models/xgboost_binary_ids.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/feature_names.pkl", "rb") as f:
        feature_names = list(pickle.load(f))

    return model, scaler, feature_names


model, scaler, feature_names = load_resources()

# ===============================
# Initialize session state
# ===============================
if "inputs" not in st.session_state:
    st.session_state.inputs = [0.0] * len(feature_names)

# ===============================
# Predefined demo vectors
# ===============================
def load_normal_sample():
    x = [0.0] * len(feature_names)

    for f in ["logged_in", "same_srv_rate", "dst_host_same_srv_rate"]:
        if f in feature_names:
            x[feature_names.index(f)] = 1.0

    return x


def load_attack_sample():
    x = [0.0] * len(feature_names)

    for f in ["serror_rate", "srv_serror_rate", "dst_host_serror_rate"]:
        if f in feature_names:
            x[feature_names.index(f)] = 1.0

    if "count" in feature_names:
        x[feature_names.index("count")] = 100

    return x


# ===============================
# Quick demo buttons
# ===============================
st.subheader("⚡ Quick Demo Inputs")

col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Load NORMAL Traffic"):
        st.session_state.inputs = load_normal_sample()
        st.success("Normal traffic sample loaded")

with col2:
    if st.button("🚨 Load ATTACK Traffic"):
        st.session_state.inputs = load_attack_sample()
        st.error("Attack traffic sample loaded")

# ===============================
# Detection section
# ===============================
st.divider()
st.subheader("🧪 Detection Result")

if st.button("🔍 Detect Intrusion"):
    st.info("⏳ Running intrusion detection...")

    X = np.array(st.session_state.inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)

    probability = float(model.predict_proba(X_scaled)[0][1])

    # Custom IDS threshold
    THRESHOLD = 0.7

    prediction = 0 if probability >= THRESHOLD else 1


    if prediction == 1:
        st.error("🚨 INTRUSION DETECTED")
        st.write(f"**Attack Probability:** {probability:.2f}")
        trigger_alert(1)
    else:
        st.success("✅ NORMAL TRAFFIC")
        st.write(f"**Attack Probability:** {probability:.2f}")

# ===============================
# Footer
# ===============================
st.divider()
st.caption("Intrusion Detection System | XGBoost Binary IDS | Demo GUI")
