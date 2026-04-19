import os
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

FEATURES = [
    "anxiety_level",
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
]


def categorize_burnout(score: float) -> str:
    if score < 40:
        return "Low"
    if score < 70:
        return "Moderate"
    return "High"


def compute_burnout_score(values: dict, predicted_stress: float) -> float:
    raw = (
        0.20 * values["anxiety_level"]
        + 0.15 * values["depression"]
        + 0.15 * values["study_load"]
        + 0.10 * values["future_career_concerns"]
        + 0.10 * values["peer_pressure"]
        + 0.20 * predicted_stress
        + 0.10 * (10 - values["sleep_quality"])
        + 0.05 * (10 - values["social_support"])
        + 0.05 * (10 - values["self_esteem"])
        + 0.05 * (10 - values["academic_performance"])
        + 0.05 * (10 - values["extracurricular_activities"])
    )
    score_0_100 = (raw / 11.5) * 100
    return float(np.clip(score_0_100, 0, 100))


# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR
data_path = os.path.join(PROJECT_ROOT, "data", "StressLevelDataset.csv")
models_dir = os.path.join(PROJECT_ROOT, "models")

os.makedirs(models_dir, exist_ok=True)

# load data
df = pd.read_csv(data_path)

# features / target
X = df[FEATURES]
y = df["stress_level"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train
model = LinearRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 10)

# STEP 3: calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Model Performance:")
print(f"R^2 Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# STEP 4: save metrics
metrics = {
    "r2": float(r2),
    "mse": float(mse)
}

with open(os.path.join(models_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# burnout results preview
results = X_test.copy()
results["predicted_stress_level"] = y_pred
results["burnout_score"] = results.apply(
    lambda row: compute_burnout_score(row.to_dict(), row["predicted_stress_level"]),
    axis=1
)
results["burnout_risk_category"] = results["burnout_score"].apply(categorize_burnout)

print("\nBurnout Risk Results")
print(results[[
    "predicted_stress_level",
    "burnout_score",
    "burnout_risk_category"
]].head(10))

# save model
with open(os.path.join(models_dir, "burnout_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("\nSaved:")
print("- models/burnout_model.pkl")
print("- models/metrics.json")