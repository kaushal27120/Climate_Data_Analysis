import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

df_year = df[df["year"] == selected_year]
df_city = df_year[df_year["city"] == selected_city]

st.title(f"📦 Distributions & Weather Patterns - {selected_city} ({selected_year})")

# Numeric columns for analysis
numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']

# Box plots
st.subheader("Box Plots for Climate Variables")
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
axs = axs.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(x=df_city[col], ax=axs[i], color='skyblue')
    axs[i].set_title(f'{col.title()} Distribution')
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Weather condition frequency
st.subheader("Weather Condition Frequency")
fig2, ax2 = plt.subplots(figsize=(8,4))
df_city["weather"].value_counts().plot(kind="bar", color="orange", ax=ax2)
ax2.set_ylabel("Count")
ax2.set_title("Weather Description Frequency")
st.pyplot(fig2)
plt.close(fig2)

# Wind speed distribution
st.subheader("Wind Speed Distribution")
fig3, ax3 = plt.subplots(figsize=(8,4))
sns.histplot(df_city["wind_speed"], bins=20, kde=True, ax=ax3)
ax3.set_title("Wind Speed Histogram")
st.pyplot(fig3)
plt.close(fig3)
