"""
Intelligent Fitness Recommendation System — Streamlit App
Loads pre-trained artifacts (ANN model, scaler, encoders) — never retrains.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
from tensorflow import keras
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ----------------------------------------------------------------------
# Load saved artifacts (cached so they load once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model('models/fitness_ann_model.keras')
    scaler = joblib.load('models/fitness_scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    gender_encoder = joblib.load('models/gender_encoder.pkl')
    return model, scaler, feature_columns, gender_encoder

model, scaler, FEATURES, gender_encoder = load_artifacts()

# ----------------------------------------------------------------------
# Fuzzy Logic system (rebuilt once — lightweight, deterministic)
# ----------------------------------------------------------------------
@st.cache_resource
def build_fuzzy_system():
    bmi_in = ctrl.Antecedent(np.arange(10, 45, 0.1), 'bmi')
    freq_in = ctrl.Antecedent(np.arange(0, 8, 1), 'freq')
    exp_in = ctrl.Antecedent(np.arange(1, 4, 1), 'exp')
    intensity_out = ctrl.Consequent(np.arange(0, 101, 1), 'intensity')

    bmi_in['low'] = fuzz.trimf(bmi_in.universe, [10, 10, 18.5])
    bmi_in['normal'] = fuzz.trimf(bmi_in.universe, [17, 22, 27])
    bmi_in['high'] = fuzz.trimf(bmi_in.universe, [25, 45, 45])

    freq_in['low'] = fuzz.trimf(freq_in.universe, [0, 0, 3])
    freq_in['medium'] = fuzz.trimf(freq_in.universe, [2, 4, 6])
    freq_in['high'] = fuzz.trimf(freq_in.universe, [4, 7, 7])

    exp_in['beginner'] = fuzz.trimf(exp_in.universe, [1, 1, 2])
    exp_in['intermediate'] = fuzz.trimf(exp_in.universe, [1, 2, 3])
    exp_in['advanced'] = fuzz.trimf(exp_in.universe, [2, 3, 3])

    intensity_out['light'] = fuzz.trimf(intensity_out.universe, [0, 0, 50])
    intensity_out['moderate'] = fuzz.trimf(intensity_out.universe, [25, 50, 75])
    intensity_out['high'] = fuzz.trimf(intensity_out.universe, [50, 100, 100])

    rules = [
        # Full Experience x Frequency coverage (avoids zero-firing gaps)
        ctrl.Rule(exp_in['beginner'] & freq_in['low'], intensity_out['light']),
        ctrl.Rule(exp_in['beginner'] & freq_in['medium'], intensity_out['light']),
        ctrl.Rule(exp_in['beginner'] & freq_in['high'], intensity_out['moderate']),
        ctrl.Rule(exp_in['intermediate'] & freq_in['low'], intensity_out['light']),
        ctrl.Rule(exp_in['intermediate'] & freq_in['medium'], intensity_out['moderate']),
        ctrl.Rule(exp_in['intermediate'] & freq_in['high'], intensity_out['moderate']),
        ctrl.Rule(exp_in['advanced'] & freq_in['low'], intensity_out['moderate']),
        ctrl.Rule(exp_in['advanced'] & freq_in['medium'], intensity_out['moderate']),
        ctrl.Rule(exp_in['advanced'] & freq_in['high'], intensity_out['high']),
        # BMI-based refinements (supplementary — reinforce, never the only path)
        ctrl.Rule(exp_in['beginner'] & bmi_in['high'], intensity_out['light']),
        ctrl.Rule(exp_in['intermediate'] & bmi_in['high'], intensity_out['moderate']),
        ctrl.Rule(exp_in['advanced'] & bmi_in['normal'] & freq_in['high'], intensity_out['high']),
        ctrl.Rule(bmi_in['low'] & exp_in['beginner'], intensity_out['light']),
    ]
    return ctrl.ControlSystem(rules)

intensity_ctrl = build_fuzzy_system()

def get_fuzzy_intensity(bmi_val, freq_val, exp_val):
    sim = ctrl.ControlSystemSimulation(intensity_ctrl)
    sim.input['bmi'] = float(np.clip(bmi_val, 10, 44.9))
    sim.input['freq'] = float(np.clip(freq_val, 0, 7))
    sim.input['exp'] = float(np.clip(exp_val, 1, 3))
    try:
        sim.compute()
        score = sim.output['intensity']
    except (KeyError, ValueError):
        # Safety fallback: no rule fired (shouldn't happen with full coverage above,
        # but kept so the app never crashes on an edge-case input combination)
        score = 35 if exp_val <= 1 else (50 if exp_val == 2 else 70)
    label = 'Light' if score < 40 else ('Moderate' if score < 65 else 'High')
    return score, label

# ----------------------------------------------------------------------
# Approximation Mode helpers
# ----------------------------------------------------------------------
BODY_BUILD_BMI_RANGE = {
    'Slim': (16.0, 20.0), 'Average': (20.0, 25.0),
    'Muscular': (23.0, 28.0), 'Heavy': (28.0, 35.0),
}
ACTIVITY_FREQ_MAP = {'Sedentary': 1, 'Lightly Active': 2, 'Moderately Active': 4, 'Highly Active': 6}
EXP_MAP = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}

def approximate_profile(height_m, body_build, activity_level, workout_experience):
    bmi_low, bmi_high = BODY_BUILD_BMI_RANGE[body_build]
    bmi_mid = (bmi_low + bmi_high) / 2
    weight_mid = bmi_mid * (height_m ** 2)
    return {
        'is_estimate': True,
        'bmi_estimate_range': (round(bmi_low, 1), round(bmi_high, 1)),
        'bmi_representative': round(bmi_mid, 1),
        'weight_representative_kg': round(weight_mid, 1),
        'workout_frequency_estimate': ACTIVITY_FREQ_MAP[activity_level],
        'experience_level_estimate': EXP_MAP[workout_experience],
    }

# ----------------------------------------------------------------------
# Fitness Goal rules
# ----------------------------------------------------------------------
GOAL_RULES = {
    'Weight Loss': {'workout_focus': 'Cardio + light-moderate resistance training',
                     'duration': '30-45 minutes/session', 'frequency': '4-5 sessions/week',
                     'notes': 'Prioritize consistent moderate-intensity cardio over very high intensity for sustainability.'},
    'Muscle Gain': {'workout_focus': 'Resistance/strength training with progressive overload',
                     'duration': '45-60 minutes/session', 'frequency': '3-5 sessions/week',
                     'notes': 'Allow ~48 hours of recovery per muscle group; general protein-aware nutrition supports recovery.'},
    'General Fitness': {'workout_focus': 'Balanced mix of cardio and strength training',
                         'duration': '30-45 minutes/session', 'frequency': '3-4 sessions/week',
                         'notes': 'Variety across workout types supports overall conditioning and adherence.'},
    'Endurance': {'workout_focus': 'Cardio-focused training (running, cycling, swimming)',
                  'duration': '40-60 minutes/session', 'frequency': '4-6 sessions/week',
                  'notes': 'Increase session duration gradually rather than increasing intensity abruptly.'},
    'Strength': {'workout_focus': 'Resistance/strength-focused routine',
                 'duration': '45-60 minutes/session', 'frequency': '3-4 sessions/week',
                 'notes': 'Prioritize recovery between sessions targeting the same muscle groups.'},
}

# Specific workout type, aligned to the dataset's own Workout_Type categories
WORKOUT_TYPE_MAP = {
    'Weight Loss':     {'Light': 'Yoga',     'Moderate': 'Cardio',   'High': 'HIIT'},
    'Muscle Gain':     {'Light': 'Strength', 'Moderate': 'Strength', 'High': 'Strength'},
    'General Fitness': {'Light': 'Yoga',     'Moderate': 'Cardio',   'High': 'HIIT'},
    'Endurance':       {'Light': 'Cardio',   'Moderate': 'Cardio',   'High': 'HIIT'},
    'Strength':        {'Light': 'Strength', 'Moderate': 'Strength', 'High': 'Strength'},
}

def get_recovery_days(fuzzy_label, workout_frequency):
    base = {'Light': 1, 'Moderate': 2, 'High': 3}[fuzzy_label]
    if workout_frequency >= 6:
        base += 1  # very high training frequency needs an extra recovery day
    return min(base, 4)

# General sports-nutrition ranges (g protein / kg body weight / day) — not medical advice
PROTEIN_RANGE_G_PER_KG = {
    'Weight Loss':     (1.6, 2.2),
    'Muscle Gain':     (1.6, 2.2),
    'General Fitness': (1.2, 1.6),
    'Endurance':       (1.2, 1.6),
    'Strength':        (1.6, 2.0),
}

def get_protein_guidance(goal, weight_kg):
    lo, hi = PROTEIN_RANGE_G_PER_KG[goal]
    return {
        'range_g_per_kg': f'{lo}-{hi} g/kg body weight/day',
        'estimated_daily_grams': f'{round(lo * weight_kg)}-{round(hi * weight_kg)} g/day' if weight_kg else 'N/A',
    }

def generate_recommendation(is_estimate, predicted_calories, fuzzy_score, fuzzy_label, goal,
                             workout_frequency, weight_kg):
    goal_info = GOAL_RULES[goal]
    return {
        'fitness_goal': goal,
        'estimated_calories_burned_kcal': round(float(predicted_calories), 1),
        'fuzzy_intensity_score': round(float(fuzzy_score), 1),
        'recommended_workout_intensity': fuzzy_label,
        'recommended_workout_type': WORKOUT_TYPE_MAP[goal][fuzzy_label],
        'workout_focus': goal_info['workout_focus'],
        'suggested_session_duration': goal_info['duration'],
        'suggested_weekly_frequency': goal_info['frequency'],
        'hydration_guidance': 'Aim for at least 2-3 liters of water/day; more on high-intensity or long-duration days.',
        'recovery_days_per_week': get_recovery_days(fuzzy_label, workout_frequency),
        'protein_guidance': get_protein_guidance(goal, weight_kg),
        'goal_specific_guidance': goal_info['notes'],
        'data_basis': 'Estimated (Approximation Mode)' if is_estimate else 'User-provided measurements (Standard Mode)',
        'explanation': (
            f"Predicted Calories Burned ({predicted_calories:.0f} kcal) comes from the ANN model trained on session and "
            f"physiological features. Workout Intensity ('{fuzzy_label}') comes from the Fuzzy Logic system evaluating "
            f"BMI, workout frequency and experience level together, which handles vagueness in these inputs better than "
            f"hard thresholds. Workout type, recovery days and protein range are then adjusted for the '{goal}' goal "
            f"and the computed intensity using explicit rules."
        )
    }

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Intelligent Fitness Recommendation System", layout="centered")
st.title("Intelligent Fitness Recommendation System")
st.caption("Soft Computing Techniques Mini-Project — ANN + Fuzzy Logic")

st.sidebar.header("Input Mode")
input_mode = st.sidebar.radio("Select mode", ["Standard", "Approximation"])

st.header("User Information")

age = st.number_input("Age", min_value=10, max_value=90, value=25)
gender = st.selectbox("Gender", options=list(gender_encoder.classes_))
goal = st.selectbox("Fitness Goal", options=list(GOAL_RULES.keys()))

is_estimate = False

if input_mode == "Standard":
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input("Height (m)", min_value=1.2, max_value=2.3, value=1.70, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        bmi = weight / (height ** 2)
        st.metric("Calculated BMI", f"{bmi:.1f}")
        fat_pct = st.slider("Fat Percentage (%)", 5.0, 50.0, 20.0)
        water = st.number_input("Water Intake (liters)", min_value=0.5, max_value=6.0, value=2.5, step=0.1)
    with col2:
        max_bpm = st.number_input("Max BPM", min_value=120, max_value=220, value=180)
        avg_bpm = st.number_input("Avg BPM", min_value=60, max_value=200, value=140)
        resting_bpm = st.number_input("Resting BPM", min_value=40, max_value=100, value=65)
        session_duration = st.slider("Session Duration (hours)", 0.2, 3.0, 1.0, step=0.1)
        freq = st.slider("Workout Frequency (days/week)", 0, 7, 4)
    exp_level = st.selectbox("Experience Level", options=[1, 2, 3],
                              format_func=lambda x: {1: "1 - Beginner", 2: "2 - Intermediate", 3: "3 - Advanced"}[x])

else:  # Approximation Mode
    st.info("Approximation Mode: values below are ESTIMATES, not exact measurements.")
    height = st.number_input("Approximate Height (m)", min_value=1.2, max_value=2.3, value=1.70, step=0.01)
    body_build = st.selectbox("Body Build", options=list(BODY_BUILD_BMI_RANGE.keys()))
    activity_level = st.selectbox("Activity Level", options=list(ACTIVITY_FREQ_MAP.keys()))
    workout_experience = st.selectbox("Workout Experience", options=list(EXP_MAP.keys()))

    approx = approximate_profile(height, body_build, activity_level, workout_experience)
    is_estimate = True

    st.write(f"Estimated BMI range: **{approx['bmi_estimate_range'][0]} - {approx['bmi_estimate_range'][1]}** "
             f"(representative value used: {approx['bmi_representative']})")
    st.write(f"Estimated Weight (representative): **{approx['weight_representative_kg']} kg**")

    weight = approx['weight_representative_kg']
    bmi = approx['bmi_representative']
    freq = approx['workout_frequency_estimate']
    exp_level = approx['experience_level_estimate']

    # Reasonable representative defaults for fields not collected in Approximation Mode
    fat_pct = 25.0
    water = 2.5
    max_bpm = 180
    avg_bpm = 130
    resting_bpm = 68
    session_duration = 0.75

st.divider()

if st.button("Generate Fitness Recommendation", type="primary"):
    gender_encoded = gender_encoder.transform([gender])[0]

    input_row = np.array([[age, gender_encoded, weight, height, max_bpm, avg_bpm,
                            resting_bpm, session_duration, fat_pct, water, freq, exp_level, bmi]])
    input_scaled = scaler.transform(input_row)
    predicted_calories = model.predict(input_scaled, verbose=0).flatten()[0]

    fuzzy_score, fuzzy_label = get_fuzzy_intensity(bmi, freq, exp_level)

    rec = generate_recommendation(is_estimate, predicted_calories, fuzzy_score, fuzzy_label, goal,
                                   workout_frequency=freq, weight_kg=weight)

    st.header("Recommendation")

    st.subheader("1. User Profile")
    st.write(f"Age: {age} | Gender: {gender} | Mode: {rec['data_basis']}")

    st.subheader("2. Estimated / Entered BMI")
    st.metric("BMI", f"{bmi:.1f}")

    st.subheader("3. Predicted Calories Burned")
    st.metric("Calories Burned (per session)", f"{rec['estimated_calories_burned_kcal']} kcal")

    st.subheader("4. Fuzzy Workout Intensity")
    st.write(f"**{rec['recommended_workout_intensity']}**  (defuzzified score: {rec['fuzzy_intensity_score']}/100)")
    st.progress(min(int(rec['fuzzy_intensity_score']), 100))

    st.subheader("5. Recommended Workout")
    st.write(f"**Workout Type: {rec['recommended_workout_type']}**")
    st.write(rec['workout_focus'])

    st.subheader("6. Weekly Plan")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Session Duration", rec['suggested_session_duration'])
    col_b.metric("Weekly Frequency", rec['suggested_weekly_frequency'])
    col_c.metric("Recovery Days/Week", rec['recovery_days_per_week'])

    st.subheader("7. General Fitness Guidance")
    st.write(f"Hydration: {rec['hydration_guidance']}")
    st.write(f"Recovery: At least **{rec['recovery_days_per_week']} rest/active-recovery day(s)** per week, "
             f"scaled to workout intensity and training frequency.")
    st.write(f"Protein: **{rec['protein_guidance']['range_g_per_kg']}** "
             f"(≈ {rec['protein_guidance']['estimated_daily_grams']} for this body weight). "
             f"General sports-nutrition guidance, not a medical/dietetic prescription.")
    st.write(f"Goal-specific: {rec['goal_specific_guidance']}")

    with st.expander("8. Explanation"):
        st.write(rec['explanation'])

    st.caption("This is an academic recommendation system, not medical advice. Consult a professional before starting a new fitness program.")