import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Student Burnout Risk Prediction", page_icon="🧠", layout="wide")

QUESTION_TEXT = {
    "anxiety_level": "How anxious have you felt recently?",
    "self_esteem": "How confident have you felt in yourself recently?",
    "mental_health_history": "Do you have a history of mental health challenges?",
    "depression": "How often have you felt down or emotionally drained recently?",
    "headache": "How often have you experienced headaches recently?",
    "blood_pressure": "How often have you felt physically tense or pressured?",
    "sleep_quality": "How would you rate your sleep quality recently?",
    "breathing_problem": "How often have you had trouble breathing or felt shortness of breath during stress?",
    "noise_level": "How distracting or noisy is your usual environment?",
    "living_conditions": "How comfortable are your current living conditions?",
    "safety": "How safe do you feel in your daily environment?",
    "basic_needs": "How well are your basic needs being met?",
    "academic_performance": "How would you rate your recent academic performance?",
    "study_load": "How overwhelming does your workload feel?",
    "future_career_concerns": "How worried are you about your future career?",
    "social_support": "How supported do you feel by friends, family, or others?",
    "peer_pressure": "How much pressure do you feel from peers?",
    "extracurricular_activities": "How involved are you in activities outside of class?",
}

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
    score = score + 2.5
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
    score_0_100 = (raw / 11.5) * 100
    return float(np.clip(score_0_100, 0, 100))


def burnout_contributions(values: dict[str, float], predicted_stress: float):
    contributions = {
        "Anxiety": 0.20 * values["anxiety_level"],
        "Depression": 0.15 * values["depression"],
        "Study Load": 0.15 * values["study_load"],
        "Career Concerns": 0.10 * values["future_career_concerns"],
        "Peer Pressure": 0.10 * values["peer_pressure"],
        "Predicted Stress": 0.20 * predicted_stress,
        "Poor Sleep": 0.10 * (10 - values["sleep_quality"]),
        "Low Social Support": 0.05 * (10 - values["social_support"]),
        "Low Self-Esteem": 0.05 * (10 - values["self_esteem"]),
        "Lower Academic Performance": 0.05 * (10 - values["academic_performance"]),
        "Low Extracurricular Activity": 0.05 * (10 - values["extracurricular_activities"]),
    }

    protective = {
        "Sleep Quality": values["sleep_quality"],
        "Social Support": values["social_support"],
        "Self-Esteem": values["self_esteem"],
        "Academic Performance": values["academic_performance"],
        "Extracurricular Activity": values["extracurricular_activities"],
        "Safety": values["safety"],
        "Basic Needs": values["basic_needs"],
    }

    top_risks = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
    top_protective = sorted(protective.items(), key=lambda x: x[1], reverse=True)[:3]
    return top_risks, top_protective


@st.cache_resource
def try_load_model():
    model_path = os.path.join(BASE_DIR, "models", "burnout_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


def load_metrics():
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def suggested_actions(top_risks):
    actions = []
    risk_names = [name for name, _ in top_risks]

    if "Anxiety" in risk_names or "Depression" in risk_names:
        actions.append("Consider counseling, mindfulness, or short weekly check-ins to manage emotional strain.")
    if "Study Load" in risk_names or "Career Concerns" in risk_names:
        actions.append("Use a weekly study plan and break large assignments into smaller tasks to reduce overload.")
    if "Poor Sleep" in risk_names:
        actions.append("Improve sleep habits by keeping a regular sleep schedule and limiting late-night screen time.")
    if "Peer Pressure" in risk_names:
        actions.append("Set boundaries and reduce comparison-driven stress by focusing on personal goals.")
    if "Predicted Stress" in risk_names and not actions:
        actions.append("Schedule recovery time during the week and monitor stress patterns before they escalate.")

    if not actions:
        actions.append("Maintain current habits and continue monitoring stress, sleep, and workload balance.")

    return actions[:3]


st.title("Student Burnout Risk Prediction")
st.write("Answer the survey below to estimate burnout risk.")

model = try_load_model()
metrics = load_metrics()

if model is None:
    st.info("No trained model file found. The app is using an approximate stress estimator for now.")
else:
    st.success("Loaded trained regression model.")

st.subheader("Burnout Risk Survey")
st.caption("For most questions, use 0 for very low/not at all and 10 for very high/very often.")

with st.form("burnout_survey_form"):
    values = {}

    survey_left, survey_right = st.columns(2)
    left_features = FEATURES[:9]
    right_features = FEATURES[9:]

    with survey_left:
        for feature in left_features:
            if feature == "mental_health_history":
                values[feature] = st.selectbox(
                    QUESTION_TEXT[feature],
                    options=[0, 1],
                    index=DEFAULTS[feature],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                )
            else:
                values[feature] = st.slider(
                    QUESTION_TEXT[feature],
                    min_value=0,
                    max_value=10,
                    value=DEFAULTS[feature],
                )

    with survey_right:
        for feature in right_features:
            values[feature] = st.slider(
                QUESTION_TEXT[feature],
                min_value=0,
                max_value=10,
                value=DEFAULTS[feature],
            )

    submitted = st.form_submit_button("See My Results")

if "submitted_values" not in st.session_state:
    st.session_state.submitted_values = None

if submitted:
    st.session_state.submitted_values = values.copy()

if st.session_state.submitted_values is None:
    st.stop()

values = st.session_state.submitted_values.copy()
input_df = pd.DataFrame([values])[FEATURES]

if model is not None:
    predicted_stress = float(model.predict(input_df)[0])
    predicted_stress = float(np.clip(predicted_stress, 0, 10))
else:
    predicted_stress = approximate_stress(values)

burnout_score = compute_burnout_score(values, predicted_stress)
category = categorize_burnout(burnout_score)
top_risks, top_protective = burnout_contributions(values, predicted_stress)

if metrics:
    st.subheader("Model Performance")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("R² Score", f"{metrics['r2']:.2f}")
    with m2:
        st.metric("MSE", f"{metrics['mse']:.3f}")

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

st.markdown("### Burnout Gauge")
st.progress(min(max(int(round(burnout_score)), 0), 100))
st.caption(f"Burnout score: {burnout_score:.1f}/100")

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
    ).sort_values("Level", ascending=False).set_index("Driver")
    st.bar_chart(chart_df)

st.divider()

e1, e2 = st.columns(2)
with e1:
    st.subheader("Top Risk Drivers")
    for name, value in top_risks:
        st.write(f"- **{name}**: {value:.2f}")

with e2:
    st.subheader("Top Protective Factors")
    for name, value in top_protective:
        st.write(f"- **{name}**: {value:.1f}/10")

st.divider()

st.subheader("Suggested Actions")
for action in suggested_actions(top_risks):
    st.write(f"- {action}")

st.divider()

st.subheader("What-If Simulator")
what_if_factor = st.selectbox(
    "Choose a factor to adjust",
    options=FEATURES,
    format_func=lambda x: FRIENDLY_NAMES[x],
)

current_value = values[what_if_factor]
if what_if_factor == "mental_health_history":
    simulated_value = st.selectbox(
        "Simulated value",
        options=[0, 1],
        index=int(current_value),
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="what_if_value_mhh",
    )
else:
    simulated_value = st.slider(
        "Simulated value",
        min_value=0,
        max_value=10,
        value=int(current_value),
        key="what_if_value_slider",
    )

simulated_values = values.copy()
simulated_values[what_if_factor] = simulated_value
simulated_df = pd.DataFrame([simulated_values])[FEATURES]

if model is not None:
    simulated_stress = float(model.predict(simulated_df)[0])
    simulated_stress = float(np.clip(simulated_stress, 0, 10))
else:
    simulated_stress = approximate_stress(simulated_values)

simulated_burnout = compute_burnout_score(simulated_values, simulated_stress)
delta = simulated_burnout - burnout_score

s1, s2, s3 = st.columns(3)
with s1:
    st.metric("Current Burnout Score", f"{burnout_score:.1f}")
with s2:
    st.metric("Simulated Burnout Score", f"{simulated_burnout:.1f}")
with s3:
    st.metric("Change", f"{delta:+.1f}")

st.caption(
    f"Changing {FRIENDLY_NAMES[what_if_factor]} from {current_value} to {simulated_value} changes the burnout score by {delta:+.1f} points."
)

st.divider()

st.subheader("Download Results")
report_df = pd.DataFrame([{
    "predicted_stress": round(predicted_stress, 2),
    "burnout_score": round(burnout_score, 2),
    "risk_category": category,
    "top_risk_1": top_risks[0][0] if len(top_risks) > 0 else "",
    "top_risk_2": top_risks[1][0] if len(top_risks) > 1 else "",
    "top_risk_3": top_risks[2][0] if len(top_risks) > 2 else "",
    "top_protective_1": top_protective[0][0] if len(top_protective) > 0 else "",
    "top_protective_2": top_protective[1][0] if len(top_protective) > 1 else "",
    "top_protective_3": top_protective[2][0] if len(top_protective) > 2 else "",
}])

for feature in FEATURES:
    report_df[feature] = values[feature]

csv_data = report_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download burnout summary (CSV)",
    data=csv_data,
    file_name="burnout_summary.csv",
    mime="text/csv",
)

st.divider()

with st.expander("How the model works"):
    st.write(
        "This app uses a hybrid approach. A regression model estimates stress from student factors, then a weighted burnout formula combines predicted stress with major contributors such as anxiety, depression, study load, sleep quality, and social support."
    )

st.caption("Built with Streamlit for the website portion of the burnout risk project.")