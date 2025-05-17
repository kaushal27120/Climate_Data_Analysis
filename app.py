# app.py (entry point)
import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🌦️ Climate Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default Streamlit style elements for a cleaner UI
hide_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🌦️ Climate Data Dashboard</h1>", unsafe_allow_html=True)

# Subtitle
st.markdown(
    "<h4 style='text-align: center; color: gray;'>Explore global and city-specific weather trends, patterns, and ML-powered insights</h4>",
    unsafe_allow_html=True
)

# Divider
st.markdown("---")

# Sidebar setup
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1116/1116453.png", width=80)
    st.title("🔧 Dashboard Controls")
    
    st.subheader("📅 Select Year")
    selected_year = st.slider("Year", 2000, datetime.now().year, 2022)
    
    st.subheader("🏙️ Select City")
    city = st.selectbox("City", ["New York", "London", "Tokyo", "Sydney", "Mumbai"])
    
    st.subheader("🎨 Theme")
    theme = st.radio("Choose Theme", ["Light", "Dark", "Auto"], horizontal=True)

    st.markdown("---")
    st.markdown("Made with ❤️ by [Your Name]")

# Main content
st.markdown("""
Welcome to the **Climate Data Dashboard**! 🌍  
Dive into powerful analytics and insightful visualizations:

- 🔍 **Overview**: Key performance indicators and summary stats  
- 📈 **City Analysis**: Deep dive into weather patterns for selected cities  
- 🗺️ **Map Explorer**: Interactive map view of climate data  
- 📊 **Distributions**: Visual trends and statistical insights  
- 🤖 **ML Models**: Weather classification powered by machine learning  

Use the sidebar to customize the experience. Happy exploring! 🌤️
""")

# Optional footer or credits
st.markdown("---")
st.caption("© 2025 Climate Insights • Built with Streamlit | Manish Rai")
