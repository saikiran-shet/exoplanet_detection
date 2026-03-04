import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

RANDOM_SEED = 42
RAW_PATH = "data/exoTrain.csv"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

df = pd.read_csv(RAW_PATH)

df["LABEL"] = df["LABEL"].map({1: 0, 2: 1})

X = df.drop("LABEL", axis=1).values
y = df["LABEL"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=RANDOM_SEED
)

mean = np.mean(X_train)
std = np.std(X_train)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

np.savez(os.path.join(PROCESSED_DIR, "train.npz"), X=X_train, y=y_train)
np.savez(os.path.join(PROCESSED_DIR, "val.npz"), X=X_val, y=y_val)
np.savez(os.path.join(PROCESSED_DIR, "test.npz"), X=X_test, y=y_test)

print("Normalization complete.\n")

print("Train mean:", np.mean(X_train))
print("Train std:", np.std(X_train))

print("\nPlanet samples per split:")
print("Train:", np.sum(y_train))
print("Validation:", np.sum(y_val))
print("Test:", np.sum(y_test))