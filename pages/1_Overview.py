# pages/1_🌍_Overview.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
years = sorted(df["year"].unique())
cities = sorted(df["city"].unique())
themes = ["Blues", "Viridis", "Plasma", "Inferno"]

selected_year = st.sidebar.selectbox("Select Year", years, key="year")
selected_theme = st.sidebar.selectbox("Select Theme", themes, key="theme")
selected_city = st.sidebar.selectbox("Select City", cities, key="city")

# Filtered data
df_year = df[df["year"] == selected_year]
df_city = df_year[df_year["city"] == selected_city]

# Top KPIs
st.title("🌍 Overview - Climate Summary")
col1, col2, col3 = st.columns(3)
avg_temp = df_year.groupby("city")["temperature"].mean()
hottest_city = avg_temp.idxmax()
coldest_city = avg_temp.idxmin()

col1.metric("🔥 Hottest City", hottest_city, f"{avg_temp.max():.2f} K")
col2.metric("❄️ Coldest City", coldest_city, f"{avg_temp.min():.2f} K")
col3.metric("🌆 Cities Tracked", df_year["city"].nunique())

# Corrected Temp dial charts (filtered by selected city)
high_temp_pct = (df_city[df_city["temperature"] > df_city["temperature"].quantile(0.75)].shape[0] / df_city.shape[0]) * 100
low_temp_pct = (df_city[df_city["temperature"] < df_city["temperature"].quantile(0.25)].shape[0] / df_city.shape[0]) * 100

col4, col5 = st.columns(2)
with col4:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=high_temp_pct,
        title={'text': f"{selected_city} - High Temp Days %"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
    ))
    st.plotly_chart(fig, use_container_width=True)

with col5:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=low_temp_pct,
        title={'text': f"{selected_city} - Low Temp Days %"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}
    ))
    st.plotly_chart(fig, use_container_width=True)

# City Coordinates
city_coords = {
    "New York": {"lat": 40.7128, "lon": -74.0060},
    "Boston": {"lat": 42.3601, "lon": -71.0589},
    "Chicago": {"lat": 41.8781, "lon": -87.6298},
    "San Francisco": {"lat": 37.7749, "lon": -122.4194},
    "Seattle": {"lat": 47.6062, "lon": -122.3321},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437},
    "Houston": {"lat": 29.7604, "lon": -95.3698},
    "Phoenix": {"lat": 33.4484, "lon": -112.0740},
    "Miami": {"lat": 25.7617, "lon": -80.1918},
    "Denver": {"lat": 39.7392, "lon": -104.9903},
}

# Merge coords
df_coords = pd.DataFrame.from_dict(city_coords, orient='index')
df_coords.index.name = "city"
df_coords.reset_index(inplace=True)
df_map = df_year.merge(df_coords, on="city", how="inner")
df_avg = df_map.groupby(["city", "lat", "lon"])["temperature"].mean().reset_index()

# Map chart
st.subheader("🗺️ Average Temperature Map")
fig_map = px.scatter_geo(
    df_avg,
    lat="lat",
    lon="lon",
    text="city",
    color="temperature",
    color_continuous_scale=selected_theme.lower(),
    size="temperature",
    projection="natural earth",
    title=f"Avg Temperature per City - {selected_year}"
)
fig_map.update_traces(marker=dict(line=dict(width=0.5, color='gray')))
st.plotly_chart(fig_map, use_container_width=True)
