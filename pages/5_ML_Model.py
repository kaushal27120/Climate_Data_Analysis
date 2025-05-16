import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_models():
    rf = joblib.load("saved_models/random_forest.pkl")
    gb = joblib.load("saved_models/gradient_boosting.pkl")
    et = joblib.load("saved_models/extra_trees.pkl")
    lr = joblib.load("saved_models/logistic_regression.pkl")
    dt = joblib.load("saved_models/decision_tree.pkl")
    ensemble = joblib.load("saved_models/ensemble.pkl")
    le = joblib.load("saved_models/label_encoder.pkl")
    scaler = joblib.load("saved_models/scaler.pkl")
    features = ["temp_c", "humidity", "pressure", "wind_cat_enc"]
    return rf, gb, et, lr, dt, ensemble, le, scaler, features

rf, gb, et, lr, dt, ensemble, le, scaler, features = load_models()

st.title("🤖 Weather Prediction - Pretrained Ensemble")

st.subheader("📊 Model Accuracy Overview")
accs = {}
for model_name, model in zip(
    ["Random Forest", "Gradient Boosting", "Extra Trees", "Logistic Regression", "Decision Tree", "Voting Ensemble"],
    [rf, gb, et, lr, dt, ensemble],
):
    # This assumes you saved test data or evaluate on your own offline
    # For demo, we'll just show placeholders (replace with real accuracies if you have)
    accs[model_name] = "Load/test accuracy offline"

acc_df = pd.DataFrame(list(accs.items()), columns=["Model", "Accuracy"])
st.dataframe(acc_df)

st.subheader("🔍 Predict Weather with Voting Ensemble")

with st.form("live_form"):
    col1, col2 = st.columns(2)
    with col1:
        temp_c = st.number_input("Temperature (°C)", -40.0, 60.0, 25.0)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
    with col2:
        pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
        wind_cat = st.selectbox("Wind Category", ["calm", "breeze", "windy", "storm"])
    submit = st.form_submit_button("Predict")

if submit:
    wind_mapping = {"calm": 0, "breeze": 1, "windy": 2, "storm": 3}
    wind_encoded = wind_mapping[wind_cat]
    input_df = pd.DataFrame([[temp_c, humidity, pressure, wind_encoded]], columns=features)
    input_scaled = scaler.transform(input_df)
    prediction = ensemble.predict(input_scaled)
    label = le.inverse_transform(prediction)[0]
    st.success(f"🌤️ Predicted Weather Condition: **{label}**")
