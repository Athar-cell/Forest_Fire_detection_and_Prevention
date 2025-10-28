import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="AI Forest Fire Risk Predictor", layout="wide")

# ------------------------------
# LOAD MODEL AND DATA
# ------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("best_model.pkl")
    df = pd.read_csv("merged_dataset.csv")
    return model, df

model, df = load_resources()

st.title("🌲🔥 AI-Based Forest Fire Detection & Early Warning System")
st.markdown("This dashboard predicts forest fire risk using AI and visualizes recent fire-prone regions.")

# ------------------------------
# SIDEBAR — USER INPUT
# ------------------------------
st.sidebar.header("🧾 Input Environmental Conditions")

temp = st.sidebar.slider("🌡️ Avg Temperature (°C)", 5.0, 50.0, 30.0)
temp_max = st.sidebar.slider("🌞 Max Temperature (°C)", 10.0, 55.0, temp + 2)
temp_min = st.sidebar.slider("🌙 Min Temperature (°C)", 0.0, 45.0, temp - 2)
humidity = st.sidebar.slider("💧 Relative Humidity (%)", 0.0, 100.0, 45.0)
wind = st.sidebar.slider("💨 Wind Speed (m/s)", 0.0, 25.0, 5.0)
precip = st.sidebar.slider("🌧️ Precipitation (mm)", 0.0, 50.0, 0.0)

dryness_idx = temp_max - 0.05 * humidity
temp_range = temp_max - temp_min

features = pd.DataFrame({
    "temperature_2m_mean": [temp],
    "temperature_2m_max": [temp_max],
    "temperature_2m_min": [temp_min],
    "relative_humidity_2m_mean": [humidity],
    "windspeed_10m_mean": [wind],
    "precipitation_sum": [precip],
    "dryness_idx": [dryness_idx],
    "temp_range": [temp_range]
})

# ------------------------------
# PREDICTION
# ------------------------------
prob = model.predict_proba(features)[0][1]
risk = "🔥 High Risk" if prob > 0.6 else ("⚠️ Moderate Risk" if prob > 0.3 else "✅ Low Risk")

st.markdown(f"## 🔮 Predicted Fire Risk: **{risk}**")
st.progress(float(prob))
st.markdown(f"**Probability of Fire:** `{prob:.2%}`")

# ------------------------------
# FEATURE IMPORTANCE (if available)
# ------------------------------
if hasattr(model, "feature_importances_"):
    st.subheader("📊 Feature Importance")
    imp = pd.Series(model.feature_importances_, index=features.columns).sort_values()
    st.bar_chart(imp)

# ------------------------------
# MAP VISUALIZATION
# ------------------------------
st.markdown("---")
st.subheader("🗺️ Historical Fire-Prone Zones (from dataset)")

if {"lat", "lon", "risk_next7d"}.issubset(df.columns):
    df_sample = df.sample(min(5000, len(df)))
    fig = px.scatter_mapbox(
        df_sample,
        lat="lat",
        lon="lon",
        color="risk_next7d",
        color_continuous_scale="hot",
        zoom=6,
        height=500,
        title="Fire-Prone Zones"
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(r=0, t=30, l=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No lat/lon columns available for map plotting.")

st.caption("© AI-Based Forest Fire Detection System | Streamlit + Machine Learning")
