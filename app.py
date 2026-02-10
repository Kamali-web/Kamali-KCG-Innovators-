import streamlit as st
import numpy as np
import librosa
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🔊 Deepfake Voice Detector")
st.write("Upload a voice sample to check if it is real or AI-generated.")

# File upload
audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

def extract_features(file):
    y, sr = librosa.load(file, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

if audio_file is not None:
    st.audio(audio_file)

    # Feature extraction
    features = extract_features(audio_file)
    features = features.reshape(1, -1)

    # Prediction
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]

    if prediction == 1:
        st.error("⚠️ Deepfake Voice Detected")
        st.write(f"Confidence: {max(confidence)*100:.2f}%")
    else:
        st.success("✅ Real Human Voice")
        st.write(f"Confidence: {max(confidence)*100:.2f}%")

