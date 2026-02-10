import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import pickle
import scipy.io.wavfile as wav

st.title("📞 Live Voice Trust Check")

model = pickle.load(open("model.pkl", "rb"))

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=20)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

if st.button("Start Voice Check"):
    st.write("Speak now...")

    duration = 3  # seconds
    fs = 16000

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    audio = recording.flatten()

    features = extract_features(audio)
    probs = model.predict_proba(features)[0]
    trust_score = int(probs[0] * 100)

    st.metric("Trust Score", f"{trust_score}%")

    if trust_score < 50:
        st.error("⚠ Possible AI-generated voice detected")
    else:
        st.success("✔ Human voice verified")
