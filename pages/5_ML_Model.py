import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

from main import load_and_prepare_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    if os.path.exists("processed_climate_data.csv"):
        df = pd.read_csv("processed_climate_data.csv", parse_dates=["datetime"], index_col="datetime")
    else:
        df = load_and_prepare_data()
        df.to_csv("processed_climate_data.csv")
    return df

@st.cache_resource
def train_fast_ensemble():
    df = load_data()
    df_year = df[df["year"] == df["year"].max()].copy()
    df_sample = df_year.sample(frac=0.3, random_state=42)

    le = LabelEncoder()
    df_sample["weather_encoded"] = le.fit_transform(df_sample["weather"])
    X = df_sample[["temperature", "humidity", "pressure", "wind_speed"]]
    y = df_sample["weather_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42)
    }

    model_results = []
    trained_models = {}

    for name, model in models.items():
        start = time.time()
        try:
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            duration = round(time.time() - start, 2)
            trained_models[name] = model
            model_results.append({"Model": name, "Accuracy": acc, "Train Time (s)": duration})
        except Exception:
            model_results.append({"Model": name, "Accuracy": 0.0, "Train Time (s)": 0})

    voting_clf = VotingClassifier(estimators=[(k, v) for k, v in trained_models.items()], voting='soft')
    voting_clf.fit(X_train, y_train)
    ensemble_acc = accuracy_score(y_test, voting_clf.predict(X_test))

    return trained_models, voting_clf, model_results, le, X.columns.tolist(), X_test, y_test

# Train models in background
with st.spinner("⏳ Training optimized models..."):
    trained_models, voting_model, model_stats, label_encoder, feature_names, X_test, y_test = train_fast_ensemble()

# Dashboard title
st.title("🤖 Optimized Voting Ensemble Dashboard")

# Model comparison
st.subheader("📊 Model Performance Comparison")
df_stats = pd.DataFrame(model_stats).sort_values(by="Accuracy", ascending=False)
fig, ax = plt.subplots()
sns.barplot(data=df_stats, x="Accuracy", y="Model", palette="crest", ax=ax)
ax.set_title("Accuracy by Model")
st.pyplot(fig)
plt.close(fig)

# Ensemble accuracy
ensemble_acc = accuracy_score(y_test, voting_model.predict(X_test))
st.metric("Voting Ensemble Accuracy", f"{ensemble_acc:.2%}")

# Live prediction UI
st.subheader("🔍 Predict Weather Using Ensemble")
with st.form("live_prediction"):
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("Temperature (K)", 200.0, 330.0, 290.0)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
    with col2:
        pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
        wind = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    user_df = pd.DataFrame([[temp, humidity, pressure, wind]], columns=feature_names)
    prediction = voting_model.predict(user_df)
    label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🌤️ Predicted Weather: **{label}**")
