import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from main import load_and_prepare_data

@st.cache_data
def load_data():
    if os.path.exists("processed_climate_data.csv"):
        df = pd.read_csv("processed_climate_data.csv", parse_dates=["datetime"], index_col="datetime")
    else:
        df = load_and_prepare_data()
        df.to_csv("processed_climate_data.csv")
    return df

df = load_data()

st.sidebar.header("🔧 Global Filters")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()), key="year")
selected_theme = st.sidebar.selectbox("Select Theme", ["Blues", "Viridis", "Plasma", "Inferno"], key="theme")
selected_city = st.sidebar.selectbox("Select City", sorted(df["city"].unique()), key="city")

df_year = df[df["year"] == selected_year].copy()  # .copy() to avoid SettingWithCopyWarning

st.title(f"🤖 Weather Classification Model - {selected_year}")

# Prepare data
numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
le = LabelEncoder()
df_year["weather_encoded"] = le.fit_transform(df_year["weather"])
X = df_year[numeric_cols]
y = df_year["weather_encoded"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.metric("Model Accuracy", f"{acc:.2%}")

# Feature importance plot
feat_importances = pd.Series(model.feature_importances_, index=numeric_cols).sort_values(ascending=False)

st.subheader("Feature Importance")
fig, ax = plt.subplots()
sns.barplot(x=feat_importances.values, y=feat_importances.index, color='mediumseagreen', ax=ax)
ax.set_xlabel("Importance")
ax.set_title("Feature Importance in Random Forest")
st.pyplot(fig)
plt.close(fig)

# Optional: show classification report
if st.checkbox("Show Classification Report"):
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)
