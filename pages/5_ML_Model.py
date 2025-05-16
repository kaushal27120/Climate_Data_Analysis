import streamlit as st
import pandas as pd
import os
import time
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from main import load_and_prepare_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
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

def feature_engineering(df):
    df = df.copy()
    df["temp_c"] = df["temperature"] - 273.15
    df["wind_cat"] = pd.cut(df["wind_speed"], bins=[0,3,7,15,50], labels=["calm","breeze","windy","storm"])
    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_and_cache_models():
    df = load_data()
    df = feature_engineering(df)
    df_year = df[df["year"] == df["year"].max()].copy()

    # Stratified sample to keep class balance
    df_sample = df_year.groupby("weather").apply(lambda x: x.sample(frac=0.3, random_state=42)).reset_index(drop=True)

    le = LabelEncoder()
    df_sample["weather_encoded"] = le.fit_transform(df_sample["weather"])
    df_sample["wind_cat_enc"] = LabelEncoder().fit_transform(df_sample["wind_cat"])

    features = ["temp_c", "humidity", "pressure", "wind_cat_enc"]
    X = df_sample[features]
    y = df_sample["weather_encoded"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)

    for model, name in zip([rf, gb, lr], ["rf", "gb", "lr"]):
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name}.pkl")

    ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb), ("lr", lr)], voting='soft')
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, "models/voting.pkl")

    return {
        "rf": rf, "gb": gb, "lr": lr, "voting": ensemble,
        "le": le, "scaler": scaler, "features": features,
        "X_test": X_test, "y_test": y_test
    }

os.makedirs("models", exist_ok=True)

with st.spinner("Training models with feature engineering and caching..."):
    result = train_and_cache_models()

st.title("📊 Final Optimized ML Ensemble")

accs = {
    name: accuracy_score(result["y_test"], result[name].predict(result["X_test"]))
    for name in ["rf", "gb", "lr"]
}
accs["Voting Ensemble"] = accuracy_score(result["y_test"], result["voting"].predict(result["X_test"]))

st.subheader("Model Accuracies")
df_acc = pd.DataFrame(list(accs.items()), columns=["Model", "Accuracy"]).sort_values("Accuracy", ascending=False)
fig, ax = plt.subplots()
sns.barplot(data=df_acc, x="Accuracy", y="Model", ax=ax, palette="crest")
ax.set_title("Accuracy Comparison")
st.pyplot(fig)

st.subheader("🔍 Predict Weather with Final Ensemble")
with st.form("live_form"):
    c1, c2 = st.columns(2)
    with c1:
        temp_c = st.number_input("Temperature (°C)", -40.0, 60.0, 25.0)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
    with c2:
        pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
        wind_cat = st.selectbox("Wind Category", ["calm", "breeze", "windy", "storm"])
    predict_btn = st.form_submit_button("Predict")

if predict_btn:
    wind_encoded = {"calm": 0, "breeze": 1, "windy": 2, "storm": 3}[wind_cat]
    input_df = pd.DataFrame([[temp_c, humidity, pressure, wind_encoded]], columns=result["features"])
    input_scaled = result["scaler"].transform(input_df)
    pred = result["voting"].predict(input_scaled)
    label = result["le"].inverse_transform(pred)[0]
    st.success(f"🌤️ Predicted Weather: **{label}**")
