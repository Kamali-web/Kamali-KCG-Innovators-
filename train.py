import numpy as np
import librosa
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def extract_features(file):
    y, sr = librosa.load(file, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

X = []
y = []

# real voices
for file in os.listdir("real"):
    path = os.path.join("real", file)
    X.append(extract_features(path))
    y.append(0)

# fake voices
for file in os.listdir("fake"):
    path = os.path.join("fake", file)
    X.append(extract_features(path))
    y.append(1)

X = np.array(X)
y = np.array(y)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved as model.pkl")
