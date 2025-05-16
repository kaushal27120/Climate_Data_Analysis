# pages/2_📈_City_Trends.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from main import load_and_prepare_data

# Load data
@st.cache_data
def load_data():
    if os.path.exists("processed_climate_data.csv"):
        df = pd.read_csv("processed_climate_data.csv", parse_dates=["datetime"], index_col="datetime")
    else:
        df = load_and_prepare_data()
        df.to_csv("processed_climate_data.csv")
    return df

df = load_data()

# Sidebar selections
st.sidebar.header("🔧 Global Filters")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()), key="year")
selected_theme = st.sidebar.selectbox("Select Theme", ["Blues", "Viridis", "Plasma", "Inferno"], key="theme")
selected_city = st.sidebar.selectbox("Select City", sorted(df["city"].unique()), key="city")

# Filtered data
df_year = df[(df["year"] == selected_year) & (df["city"] == selected_city)]

st.title(f"📈 Trends for {selected_city} - {selected_year}")

# Temperature trend
st.subheader("🌡️ Temperature Over Time")
fig_temp = px.line(df_year.reset_index(), x="datetime", y="temperature", title="Temperature (K)", color_discrete_sequence=["red"])
st.plotly_chart(fig_temp, use_container_width=True)

# Humidity trend
st.subheader("💧 Humidity Over Time")
fig_humidity = px.line(df_year.reset_index(), x="datetime", y="humidity", title="Humidity (%)", color_discrete_sequence=["blue"])
st.plotly_chart(fig_humidity, use_container_width=True)

# Wind speed trend
st.subheader("🌬️ Wind Speed Over Time")
fig_wind = px.line(df_year.reset_index(), x="datetime", y="wind_speed", title="Wind Speed", color_discrete_sequence=["green"])
st.plotly_chart(fig_wind, use_container_width=True)

# Pressure trend
st.subheader("📊 Pressure Over Time")
fig_pressure = px.line(df_year.reset_index(), x="datetime", y="pressure", title="Pressure", color_discrete_sequence=["orange"])
st.plotly_chart(fig_pressure, use_container_width=True)
