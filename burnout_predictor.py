import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# load dataset
data_path = os.path.join("data", "StressLevelDataset.csv")
df = pd.read_csv(data_path)

# choose features for regression model
X = df[
    [
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
        "extracurricular_activities"
    ]
]

# target for regression model
y = df["stress_level"]

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create and train multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predict stress level
y_pred = model.predict(X_test)

# evaluate regression model
print("Stress Prediction Results")
print("R^2 score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# -----------------------------------------
# HYBRID APPROACH: Create burnout risk score
# -----------------------------------------

# make a copy of test data so we can attach predictions
results = X_test.copy()

# add actual and predicted stress
results["actual_stress_level"] = y_test.values
results["predicted_stress_level"] = y_pred

# burnout score formula
# higher anxiety, depression, study load, peer pressure, and predicted stress increase burnout risk
# higher sleep quality, self-esteem, academic performance, social support, and extracurriculars reduce burnout risk

results["burnout_score_raw"] = (
    0.20 * results["anxiety_level"] +
    0.15 * results["depression"] +
    0.15 * results["study_load"] +
    0.10 * results["future_career_concerns"] +
    0.10 * results["peer_pressure"] +
    0.20 * results["predicted_stress_level"] +
    0.10 * (10 - results["sleep_quality"]) +
    0.05 * (10 - results["social_support"]) +
    0.05 * (10 - results["self_esteem"]) +
    0.05 * (10 - results["academic_performance"]) +
    0.05 * (10 - results["extracurricular_activities"])
)

# normalize burnout score to 0-100
min_score = results["burnout_score_raw"].min()
max_score = results["burnout_score_raw"].max()

results["burnout_score"] = (
    (results["burnout_score_raw"] - min_score) / (max_score - min_score)
) * 100

# classify burnout risk
def categorize_burnout(score):
    if score < 40:
        return "Low"
    elif score < 70:
        return "Moderate"
    else:
        return "High"

results["burnout_risk_category"] = results["burnout_score"].apply(categorize_burnout)

# display sample output
print("\nBurnout Risk Results")
print(results[[
    "predicted_stress_level",
    "burnout_score",
    "burnout_risk_category"
]].head(10))

import pickle
import os

# create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# save the trained model
with open("models/burnout_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved!")