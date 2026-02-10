from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
import pickle
import librosa
import numpy as np
import requests
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime

app = FastAPI()

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# In-memory fraud logs
fraud_logs = []


# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y = librosa.effects.trim(y)[0]

    # MUST match training (13 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)

    return features


# -----------------------------
# Banking API Simulation
# -----------------------------
def banking_decision(trust_score):
    if trust_score < 50:
        return "BLOCKED"
    return "APPROVED"


# -----------------------------
# Step 1: Incoming Call Handler
# -----------------------------
@app.post("/voice")
async def voice_call(request: Request):
    response = VoiceResponse()

    response.say("This call is being verified for security purposes.")

    response.record(
        timeout=3,
        action="/analyze",
        method="POST"
    )

    return Response(content=str(response), media_type="application/xml")


# -----------------------------
# Step 2: Analyze Recorded Audio
# -----------------------------
@app.post("/analyze")
async def analyze_call(request: Request):
    form = await request.form()
    recording_url = form.get("RecordingUrl")

    # Download audio from Twilio
    audio_url = recording_url + ".wav"
    audio = requests.get(audio_url)

    with open("call.wav", "wb") as f:
        f.write(audio.content)

    # Extract features
    features = extract_features("call.wav")

    # Predict
    probs = model.predict_proba(features)[0]
    trust_score = int(probs[0] * 100)

    # Banking decision
    bank_status = banking_decision(trust_score)

    # Log fraud event
    fraud_logs.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trust_score": trust_score,
        "bank_status": bank_status
    })

    response = VoiceResponse()

    # Decision logic
    if trust_score < 50:
        response.say(
            "Warning. Possible AI generated voice detected. "
            "Transaction has been blocked."
        )
    else:
        response.say(
            "Voice verified. Transaction approved."
        )

    return Response(content=str(response), media_type="application/xml")



@app.post("/bank/verify")
async def verify_transaction(request: Request):
    data = await request.json()
    trust_score = data.get("trust_score")

    if trust_score < 50:
        return JSONResponse({
            "status": "blocked",
            "message": "Suspicious voice detected. Transaction stopped."
        })
    else:
        return JSONResponse({
            "status": "approved",
            "message": "Voice verified. Transaction allowed."
        })


# -----------------------------
# Fraud Logs API (for dashboard)
# -----------------------------
@app.get("/fraud-logs")
def get_fraud_logs():
    return fraud_logs

