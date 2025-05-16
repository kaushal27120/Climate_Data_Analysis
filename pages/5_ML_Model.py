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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@st.cache_data
def load_data():
    if os.path.exists("processed_climate_data.csv"):
        df = pd.read_csv("processed_climate_data.csv", parse_dates=["datetime"], index_col="datetime")
    else:
        df = load_and_prepare_data()
        df.to_csv("processed_climate_data.csv")
    return df

@st.cache_resource
def train_models():
    df = load_data()
    df_year = df[df["year"] == df["year"].max()].copy()
    df_sample = df_year.sample(frac=0.5, random_state=42)

    le = LabelEncoder()
    df_sample["weather_encoded"] = le.fit_transform(df_sample["weather"])
    X = df_sample[["temperature", "humidity", "pressure", "wind_speed"]]
    y = df_sample["weather_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, kernel='linear', random_state=42),
    }

    model_results = []
    fitted_models = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        duration = round(time.time() - start, 2)
        acc = accuracy_score(y_test, model.predict(X_test))
        fitted_models[name] = model
        model_results.append({"Model": name, "Accuracy": acc, "Train Time": duration})

    voting_clf = VotingClassifier(estimators=[(k, v) for k, v in fitted_models.items()], voting='soft')
    voting_clf.fit(X_train, y_train)
    ensemble_acc = accuracy_score(y_test, voting_clf.predict(X_test))

    return fitted_models, voting_clf, model_results, le, X.columns.tolist(), X_test, y_test

# Run training once, cache result
with st.spinner("⏳ Training models in background..."):
    fitted_models, voting_model, model_stats, label_encoder, feature_names, X_test, y_test = train_models()

# Title
st.title("🤖 Auto-Trained Voting Ensemble")

# Accuracy comparison chart
st.subheader("📊 Model Accuracy Comparison")
df_stats = pd.DataFrame(model_stats).sort_values(by="Accuracy", ascending=False)
fig, ax = plt.subplots()
sns.barplot(data=df_stats, x="Accuracy", y="Model", palette="coolwarm", ax=ax)
st.pyplot(fig)
plt.close(fig)

# Ensemble accuracy
ensemble_acc = accuracy_score(y_test, voting_model.predict(X_test))
st.metric("Voting Ensemble Accuracy", f"{ensemble_acc:.2%}")

# Prediction interface
st.subheader("🔍 Live Weather Prediction with Ensemble")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("Temperature (K)", 200.0, 330.0, 290.0)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
    with col2:
        pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0)
    predict_button = st.form_submit_button("Predict")

if predict_button:
    user_input = pd.DataFrame([[temp, humidity, pressure, wind_speed]], columns=feature_names)
    prediction = voting_model.predict(user_input)
    label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🌤️ Predicted Weather: **{label}**")
