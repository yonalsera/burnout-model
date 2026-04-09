import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

st.set_page_config(page_title="Student Burnout Risk Prediction", page_icon="🧠", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
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

FRIENDLY_NAMES = {
    "anxiety_level": "Anxiety Level",
    "self_esteem": "Self-Esteem",
    "mental_health_history": "Mental Health History",
    "depression": "Depression",
    "headache": "Headache",
    "blood_pressure": "Blood Pressure",
    "sleep_quality": "Sleep Quality",
    "breathing_problem": "Breathing Problems",
    "noise_level": "Noise Level",
    "living_conditions": "Living Conditions",
    "safety": "Safety",
    "basic_needs": "Basic Needs",
    "academic_performance": "Academic Performance",
    "study_load": "Study Load",
    "future_career_concerns": "Future Career Concerns",
    "social_support": "Social Support",
    "peer_pressure": "Peer Pressure",
    "extracurricular_activities": "Extracurricular Activities",
}

DEFAULTS = {
    "anxiety_level": 5,
    "self_esteem": 5,
    "mental_health_history": 1,
    "depression": 5,
    "headache": 4,
    "blood_pressure": 3,
    "sleep_quality": 5,
    "breathing_problem": 3,
    "noise_level": 4,
    "living_conditions": 6,
    "safety": 6,
    "basic_needs": 7,
    "academic_performance": 6,
    "study_load": 5,
    "future_career_concerns": 5,
    "social_support": 6,
    "peer_pressure": 4,
    "extracurricular_activities": 5,
}


def categorize_burnout(score: float) -> str:
    if score < 40:
        return "Low"
    if score < 70:
        return "Moderate"
    return "High"


def risk_color(category: str) -> str:
    return {
        "Low": "#16a34a",
        "Moderate": "#d97706",
        "High": "#dc2626",
    }[category]


def approximate_stress(values: dict[str, float]) -> float:
    """
    Fallback estimate if no trained model is available.
    Returns a value roughly on a 0-10 scale.
    """
    score = (
        0.16 * values["anxiety_level"]
        + 0.14 * values["depression"]
        + 0.13 * values["study_load"]
        + 0.10 * values["future_career_concerns"]
        + 0.08 * values["peer_pressure"]
        + 0.09 * values["mental_health_history"] * 5
        + 0.07 * values["headache"]
        + 0.06 * values["breathing_problem"]
        + 0.05 * values["noise_level"]
        + 0.05 * values["blood_pressure"]
        - 0.10 * values["sleep_quality"]
        - 0.08 * values["social_support"]
        - 0.07 * values["self_esteem"]
        - 0.05 * values["academic_performance"]
        - 0.04 * values["living_conditions"]
        - 0.04 * values["safety"]
        - 0.03 * values["basic_needs"]
        - 0.03 * values["extracurricular_activities"]
    )
    # Shift and clip to a 0-10 range
    score = (score + 2.5)
    return float(np.clip(score, 0, 10))


def compute_burnout_score(values: dict[str, float], predicted_stress: float) -> float:
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
    # Normalize approximate raw range to 0-100 for display
    score_0_100 = (raw / 11.5) * 100
    return float(np.clip(score_0_100, 0, 100))


@st.cache_resource
def try_load_model():
    model_path = os.path.join("models", "burnout_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


# -----------------------------
# Header
# -----------------------------
st.title("Student Burnout Risk Prediction")
st.write(
    "Estimate burnout risk using psychological, physiological, academic, environmental, and social factors."
)

model = try_load_model()
if model is None:
    st.info("No trained model file found in `models/burnout_model.pkl`. The app is using an approximate stress estimator for now.")
else:
    st.success("Loaded trained regression model from `models/burnout_model.pkl`.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Student Inputs")
st.sidebar.caption("Use the sliders below to enter values on a 0-10 scale unless noted otherwise.")

values: dict[str, float] = {}
for feature in FEATURES:
    if feature == "mental_health_history":
        values[feature] = st.sidebar.selectbox(
            FRIENDLY_NAMES[feature],
            options=[0, 1],
            index=DEFAULTS[feature],
            help="0 = No history, 1 = Has history",
        )
    else:
        values[feature] = st.sidebar.slider(
            FRIENDLY_NAMES[feature],
            min_value=0,
            max_value=10,
            value=DEFAULTS[feature],
        )

input_df = pd.DataFrame([values])[FEATURES]

# -----------------------------
# Prediction Logic
# -----------------------------
if model is not None:
    predicted_stress = float(model.predict(input_df)[0])
    predicted_stress = float(np.clip(predicted_stress, 0, 10))
else:
    predicted_stress = approximate_stress(values)

burnout_score = compute_burnout_score(values, predicted_stress)
category = categorize_burnout(burnout_score)

# -----------------------------
# Main Layout
# -----------------------------
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("Predicted Stress", f"{predicted_stress:.2f} / 10")
with col2:
    st.metric("Burnout Score", f"{burnout_score:.1f} / 100")
with col3:
    st.markdown(
        f"""
        <div style="padding: 0.6rem 1rem; border-radius: 0.8rem; background-color: {risk_color(category)}22; border: 1px solid {risk_color(category)}; text-align:center;">
            <div style="font-size:0.9rem; color:#666;">Risk Category</div>
            <div style="font-size:1.5rem; font-weight:700; color:{risk_color(category)};">{category}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Factor Summary")
    display_df = pd.DataFrame(
        {
            "Factor": [FRIENDLY_NAMES[f] for f in FEATURES],
            "Value": [values[f] for f in FEATURES],
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Key Burnout Drivers")
    driver_values = {
        "Anxiety": values["anxiety_level"],
        "Depression": values["depression"],
        "Study Load": values["study_load"],
        "Career Concerns": values["future_career_concerns"],
        "Peer Pressure": values["peer_pressure"],
        "Poor Sleep": 10 - values["sleep_quality"],
        "Low Social Support": 10 - values["social_support"],
    }
    chart_df = pd.DataFrame(
        {"Driver": list(driver_values.keys()), "Level": list(driver_values.values())}
    ).set_index("Driver")
    st.bar_chart(chart_df)

st.divider()

with st.expander("How the model works"):
    st.write(
        "This app uses a hybrid approach. A regression model estimates stress from student factors, then a weighted burnout formula combines predicted stress with major contributors such as anxiety, depression, study load, sleep quality, and social support."
    )

with st.expander("Suggested project sections for your poster"):
    st.markdown(
        """
        - **Problem:** Student burnout is influenced by multiple interacting factors.
        - **Method:** Multiple linear regression + burnout score formula.
        - **Results:** Report your model R², MSE, and burnout risk categories.
        - **Future Work:** Compare against Random Forest and refine weights.
        """
    )

st.caption("Built with Streamlit for the website portion of the burnout risk project.")
