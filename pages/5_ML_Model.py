import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

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
    model_scores = joblib.load("saved_models/model_scores.pkl")
    features = ["temp_c", "humidity", "pressure", "wind_cat_enc"]
    return rf, gb, et, lr, dt, ensemble, le, scaler, model_scores, features

# Load everything
rf, gb, et, lr, dt, ensemble, le, scaler, model_scores, features = load_models()

# Sidebar chart switch
chart_type = st.sidebar.radio(
    "📈 Choose Accuracy Visualization",
    ["Funnel Chart", "Radar Chart", "Bar Race Chart"]
)

st.title("🤖 Weather Prediction - Pretrained Ensemble")

# Prepare accuracy data
model_names = list(model_scores.keys())
accuracies = [round(acc * 100, 2) for acc in model_scores.values()]
df_scores = pd.DataFrame({"Model": model_names, "Accuracy": accuracies}).sort_values("Accuracy", ascending=False)

# Charts
st.subheader("📊 Model Accuracy Comparison")

if chart_type == "Funnel Chart":
    fig = px.funnel(df_scores, x="Accuracy", y="Model", color="Model", title="Funnel Chart: Accuracy Comparison")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Radar Chart":
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df_scores["Accuracy"],
        theta=df_scores["Model"],
        fill='toself',
        name='Accuracy'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      showlegend=False, title="Radar Chart: Accuracy Comparison")
    st.plotly_chart(fig, use_container_width=True)

# elif chart_type == "Bar Race Chart":
#     fig = px.bar(df_scores, x="Accuracy", y="Model", orientation='h',
#                  title="Bar Chart Race: Accuracy by Model", color="Model", animation_frame="Model")
#     st.plotly_chart(fig, use_container_width=True)

# Optional: Confusion Matrix
st.subheader("🧪 Confusion Matrix")
selected_model = st.selectbox("Choose model to view confusion matrix", model_names)
model_map = {
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Extra Trees": et,
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Voting Ensemble": ensemble
}

if st.button("Show Confusion Matrix"):
    # Load and prepare data again
    df = pd.read_csv("processed_climate_data.csv", parse_dates=["datetime"], index_col="datetime")
    df["temp_c"] = df["temperature"] - 273.15
    df["wind_cat"] = pd.cut(df["wind_speed"], bins=[0,3,7,15,50], labels=["calm","breeze","windy","storm"])
    df["wind_cat_enc"] = joblib.load("saved_models/label_encoder.pkl").fit_transform(df["wind_cat"])
    df.dropna(inplace=True)
    df_year = df[df["year"] == df["year"].max()].copy()
    X = df_year[features]
    y = joblib.load("saved_models/label_encoder.pkl").transform(df_year["weather"])
    X_scaled = scaler.transform(X)

    model = model_map[selected_model]
    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

# Prediction UI
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
