import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================
# Load model artifacts
# =========================
BASE_DIR = Path(__file__).parent

try:
    loaded_model = joblib.load(BASE_DIR / "fire_model.pkl")
    loaded_scaler = joblib.load(BASE_DIR / "scaler.pkl")
    loaded_feature_order = joblib.load(BASE_DIR / "feature_order.pkl")

    if not isinstance(loaded_feature_order, list):
        st.error("feature_order.pkl is invalid.")
        st.stop()

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# =========================
# App UI
# =========================
st.title("🔥 Forest Fire Occurrence Prediction")
st.write("Adjust the inputs to predict whether a forest fire is likely to occur.")

# =========================
# User inputs
# =========================
user_input = {}

month_map = {
    'jan': 'month_jan', 'feb': 'month_feb', 'mar': 'month_mar',
    'apr': 'month_apr', 'may': 'month_may', 'jun': 'month_jun',
    'jul': 'month_jul', 'aug': 'month_aug', 'sep': 'month_sep',
    'oct': 'month_oct', 'nov': 'month_nov', 'dec': 'month_dec'
}

day_map = {
    'mon': 'day_mon', 'tue': 'day_tue', 'wed': 'day_wed',
    'thu': 'day_thu', 'fri': 'day_fri', 'sat': 'day_sat', 'sun': 'day_sun'
}

selected_month = st.selectbox("Month", list(month_map.keys()))
selected_day = st.selectbox("Day", list(day_map.keys()))

# Initialize categorical features
for col in loaded_feature_order:
    if col.startswith("month_") or col.startswith("day_"):
        user_input[col] = 0

user_input[month_map[selected_month]] = 1
user_input[day_map[selected_day]] = 1

# =========================
# Numeric inputs
# =========================
user_input['X'] = st.slider('X (1–9)', 1, 9, 5)
user_input['Y'] = st.slider('Y (2–9)', 2, 9, 5)
user_input['FFMC'] = st.slider('FFMC', 18.0, 97.0, 90.0)
user_input['DMC'] = st.slider('DMC', 1.0, 292.0, 110.0)
user_input['DC'] = st.slider('DC', 7.0, 861.0, 500.0)
user_input['ISI'] = st.slider('ISI', 0.0, 57.0, 9.0)
user_input['temp'] = st.slider('Temperature (°C)', 2.0, 34.0, 18.0)
user_input['RH'] = st.slider('Relative Humidity (%)', 15, 100, 45)
user_input['wind'] = st.slider('Wind (km/h)', 0.0, 10.0, 4.0)
user_input['rain'] = st.slider('Rain (mm)', 0.0, 7.0, 0.0)

# =========================
# Prepare input (FIXED)
# =========================
input_df = pd.DataFrame([user_input])

final_input_df = input_df.reindex(
    columns=loaded_feature_order,
    fill_value=0
).astype(float)

# 🔑 CRITICAL FIX: use NumPy array
scaled_input = loaded_scaler.transform(final_input_df.values)

# =========================
# Prediction
# =========================
if st.button("Predict Fire Occurrence"):
    prediction = loaded_model.predict(scaled_input)
    prediction_proba = loaded_model.predict_proba(scaled_input)

    if prediction[0] == 1:
        st.error(
            f"🔥 Fire likely to occur "
            f"(Probability: {prediction_proba[0][1]:.2f})"
        )
    else:
        st.success(
            f"✅ Fire unlikely to occur "
            f"(Probability: {prediction_proba[0][0]:.2f})"
        )
