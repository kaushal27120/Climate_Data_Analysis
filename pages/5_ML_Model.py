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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@st.cache_data
def load_data():
    if os.path.exists("processed_climate_data.csv"):
        df = pd.read_csv("processed_climate_data.csv", parse_dates=["datetime"], index_col="datetime")
    else:
        df = load_and_prepare_data()
        df.to_csv("processed_climate_data.csv")
    return df

# Load and filter data
df = load_data()
st.sidebar.header("🔧 Global Filters")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()), key="year")
selected_theme = st.sidebar.selectbox("Select Theme", ["Blues", "Viridis", "Plasma", "Inferno"], key="theme")
selected_city = st.sidebar.selectbox("Select City", sorted(df["city"].unique()), key="city")

df_year = df[df["year"] == selected_year].copy()
df_sample = df_year.sample(frac=0.5, random_state=42)  # Reduce for speed

# Encode target
le = LabelEncoder()
df_sample["weather_encoded"] = le.fit_transform(df_sample["weather"])
numeric_cols = ["temperature", "humidity", "pressure", "wind_speed"]
X = df_sample[numeric_cols]
y = df_sample["weather_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
}

# UI selections
st.title("🤖 Weather Classification Dashboard")
model_choice = st.selectbox("Select an ML Model", list(models.keys()))
run_compare = st.checkbox("Compare All Models")

# Train and evaluate
if run_compare:
    results = []
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            start = time.time()
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            duration = round(time.time() - start, 2)
            results.append({"Model": name, "Accuracy": acc, "Train Time (s)": duration})
    
    df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    st.subheader("📊 Model Comparison")
    st.dataframe(df_results.reset_index(drop=True))
    
    fig, ax = plt.subplots()
    sns.barplot(data=df_results, x="Accuracy", y="Model", palette="viridis", ax=ax)
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)
    plt.close(fig)

else:
    clf = models[model_choice]
    with st.spinner(f"Training {model_choice} model..."):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{acc:.2%}")

    # Feature importance
    if hasattr(clf, "feature_importances_"):
        feat_importances = pd.Series(clf.feature_importances_, index=numeric_cols).sort_values(ascending=False)
        st.subheader("Feature Importance")
        fig, ax = plt.subplots()
        sns.barplot(x=feat_importances.values, y=feat_importances.index, color='mediumseagreen', ax=ax)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
        plt.close(fig)

    # Classification report (summary)
    if st.checkbox("Show Classification Report (Summary Only)"):
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_summary = df_report.loc[["macro avg", "weighted avg"]]
        st.dataframe(df_summary)

        csv = df_summary.to_csv(index=True)
        st.download_button("📥 Download CSV Report", data=csv, file_name="classification_summary.csv")

    # Confusion Matrix
    if st.checkbox("Show Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # Live Prediction
    st.subheader("🔍 Live Weather Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp_input = st.number_input("Temperature (K)", min_value=200.0, max_value=330.0, value=290.0)
            humidity_input = st.slider("Humidity (%)", 0, 100, 50)
        with col2:
            pressure_input = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)
            wind_speed_input = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0)
        submitted = st.form_submit_button("Predict Weather")

    if submitted:
        input_df = pd.DataFrame([[temp_input, humidity_input, pressure_input, wind_speed_input]], columns=numeric_cols)
        try:
            prediction = clf.predict(input_df)
            predicted_label = le.inverse_transform([prediction[0]])[0]
            st.success(f"🌤️ Predicted Weather Condition: **{predicted_label}**")
        except Exception as e:
            st.error("Model not ready or failed to predict. Please train a model first.")
