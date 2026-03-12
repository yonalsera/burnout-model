import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# load dataset
data_path = os.path.join("data", "StressLevelDataset.csv")

df = pd.read_csv(data_path)

# choose features
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

# target
y = df["stress_level"]

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# run python burnout_predictor.py
