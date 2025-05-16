# pages/3_🗺️_Map_View.py
import streamlit as st
import pandas as pd
import plotly.express as px
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

df_coords = pd.DataFrame.from_dict(city_coords, orient='index')
df_coords.index.name = "city"
df_coords.reset_index(inplace=True)

df_year = df[df["year"] == selected_year]

df_map = df_year.merge(df_coords, on="city", how="inner")

df_avg = df_map.groupby(["city", "lat", "lon"])["temperature"].mean().reset_index()

st.title(f"🗺️ Temperature Map - {selected_year}")

fig_map = px.scatter_geo(
    df_avg,
    lat="lat",
    lon="lon",
    text="city",
    color="temperature",
    color_continuous_scale=selected_theme.lower(),
    size="temperature",
    projection="natural earth",
    title=f"Average Temperature per City - {selected_year}"
)
fig_map.update_traces(marker=dict(line=dict(width=0.5, color='gray')))
st.plotly_chart(fig_map, use_container_width=True)
