
import glob
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew

DATA_FILES = {
    "Normal": "Normal_0.mat",
    "Ball Fault": "B007_0.mat",
    "Inner Race Fault": "IR007_0.mat",
    "Outer Race Fault": "OR007_0.mat",
}

WINDOW_SIZE = 2048
FEATURES = [
    "rms",
    "std",
    "kurtosis",
    "skewness",
    "peak_to_peak",
    "crest_factor",
]


def extract_features(signal):
    signal = np.asarray(signal, dtype=np.float64).ravel()
    signal = signal[np.isfinite(signal)]

    if len(signal) < WINDOW_SIZE:
        return None

    rms = np.sqrt(np.mean(signal**2))
    std = np.std(signal)
    peak = np.max(np.abs(signal))
    return {
        "rms": rms,
        "std": std,
        "kurtosis": float(kurtosis(signal, fisher=False)),
        "skewness": float(skew(signal)),
        "peak_to_peak": float(np.ptp(signal)),
        "crest_factor": float(peak / rms) if rms > 0 else 0.0,
    }


rows = []

for label, filename in DATA_FILES.items():
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing dataset file: {filename}")

    mat = loadmat(filename)
    signal_keys = [
        key for key in mat
        if key.endswith("_DE_time")
    ]

    if not signal_keys:
        raise ValueError(f"No drive-end vibration signal found in {filename}")

    signal = np.asarray(mat[signal_keys[0]]).ravel()
    signal = signal[np.isfinite(signal)]

    print(f"{filename}: {len(signal):,} vibration readings")

    for start in range(0, len(signal) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = signal[start:start + WINDOW_SIZE]
        features = extract_features(window)

        if features is not None:
            rows.append({
                **features,
                "fault_type": label,
                "source_file": filename,
                "window_start": start,
            })

df = pd.DataFrame(rows)

if df.empty:
    raise ValueError("No feature rows were extracted.")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=FEATURES, inplace=True)
df.to_csv("bearing_features.csv", index=False)

print("\nDataset extraction complete!")
print("Total samples:", len(df))
print("\nSamples per class:")
print(df["fault_type"].value_counts())
print("\nSaved to bearing_features.csv")
