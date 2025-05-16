# app.py (entry point)
import streamlit as st

st.set_page_config(page_title="🌦️ Climate Dashboard", layout="wide")
st.title("🌦️ Climate Data Dashboard")

st.markdown("""
Welcome to the **Climate Data Dashboard**! 🌍

Navigate using the sidebar to explore:
- 🔍 Overall trends and KPIs
- 📈 City-specific analysis
- 🗺️ Interactive map of climate data
- 📊 Distributions and weather patterns
- 🤖 Weather classification using machine learning

Use the sidebar to select year, city, and theme across all pages.
""")