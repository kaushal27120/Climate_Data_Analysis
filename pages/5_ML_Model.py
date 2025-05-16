import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Load data
df = load_data()

st.sidebar.header("🔧 Global Filters")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()), key="year")
selected_theme = st.sidebar.selectbox("Select Theme", ["Blues", "Viridis", "Plasma", "Inferno"], key="theme")
selected_city = st.sidebar.selectbox("Select City", sorted(df["city"].unique()), key="city")
selected_model = st.sidebar.selectbox("Choose Model", [
    "Random Forest", "Decision Tree", "Logistic Regression", "Gradient Boosting", "Voting Ensemble"
])

df_year = df[df["year"] == selected_year].copy()

st.title(f"🤖 Weather Classification - {selected_model}")

# Prepare features
numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
le = LabelEncoder()
df_year["weather_encoded"] = le.fit_transform(df_year["weather"])
X = df_year[numeric_cols]
y = df_year["weather_encoded"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# Use voting ensemble if selected
if selected_model == "Voting Ensemble":
    clf = VotingClassifier(estimators=[
        ('rf', models["Random Forest"]),
        ('dt', models["Decision Tree"]),
        ('lr', models["Logistic Regression"]),
        ('gb', models["Gradient Boosting"])
    ], voting='soft')
else:
    clf = models[selected_model]

# Train model
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.metric("Model Accuracy", f"{acc:.2%}")

# Feature importance only for tree-based models
if hasattr(clf, "feature_importances_"):
    feat_importances = pd.Series(clf.feature_importances_, index=numeric_cols).sort_values(ascending=False)
    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    sns.barplot(x=feat_importances.values, y=feat_importances.index, color='mediumseagreen', ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    st.pyplot(fig)
    plt.close(fig)

# Confusion matrix (safe for all classifiers)
if st.checkbox("Show Confusion Matrix"):
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

# Classification report - trimmed
if st.checkbox("Show Classification Report (Summary Only)"):
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_summary = df_report.loc[["macro avg", "weighted avg"]]
    st.dataframe(df_summary)
