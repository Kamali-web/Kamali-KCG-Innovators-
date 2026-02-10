import streamlit as st
import librosa
import numpy as np
import pickle
import time
import pandas as pd
from datetime import datetime


model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(
    page_title="AI Voice Call Security",
    layout="centered"
)

st.title("📞 AI Voice Call Security System")
st.caption("Real-time Deepfake Voice Detection for Banking Fraud Prevention")


if "fraud_logs" not in st.session_state:
    st.session_state.fraud_logs = []


st.markdown("### 🎧 Incoming Call Simulation")

audio_file = st.file_uploader(
    "Upload caller voice (WAV file only)",
    type=["wav"]
)


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y = librosa.effects.trim(y)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    return features


def banking_api_action(trust_score):
    if trust_score < 50:
        return "🚫 Transaction BLOCKED"
    return "✅ Transaction ALLOWED"


def risk_explanation(trust_score):
    if trust_score < 30:
        return "Extremely high synthetic patterns detected"
    elif trust_score < 50:
        return "Voice shows unnatural frequency variations"
    elif trust_score < 70:
        return "Minor anomalies detected"
    else:
        return "Natural human voice patterns confirmed"


if audio_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    st.audio("temp.wav")

    if st.button("🔍 Analyze Call"):
        with st.spinner("Analyzing voice for fraud patterns..."):
            time.sleep(2)

            features = extract_features("temp.wav")
            probs = model.predict_proba(features)[0]
            trust_score = int(probs[0] * 100)

       
        st.subheader("📊 Call Verification Result")
        st.metric("Trust Score", f"{trust_score}%")

        
        st.progress(trust_score)

        explanation = risk_explanation(trust_score)
        st.info(f"🧠 AI Insight: {explanation}")

        # -------------------- Decision --------------------
        if trust_score < 50:
            st.error(
                "⚠ Possible AI-generated voice detected\n"
                "Do NOT proceed with financial actions"
            )
            scam_status = "Scam Alert Triggered"
        else:
            st.success(
                "✔ Human voice verified\n"
                "Safe to proceed"
            )
            scam_status = "Safe Call"

        # -------------------- Banking Action --------------------
        bank_action = banking_api_action(trust_score)
        st.subheader("🏦 Banking System Response")
        st.write(bank_action)

        # -------------------- Save to Fraud Logs --------------------
        log_entry = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Trust Score": trust_score,
            "Risk Level": explanation,
            "Bank Action": bank_action,
            "Status": scam_status
        }

        st.session_state.fraud_logs.append(log_entry)

# -------------------- Fraud Reporting Dashboard --------------------
st.markdown("---")
st.subheader("📁 Fraud Monitoring Dashboard")

if st.session_state.fraud_logs:
    df = pd.DataFrame(st.session_state.fraud_logs)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No calls analyzed yet.")

