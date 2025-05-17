# app.py
import streamlit as st
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="🌦️ Climate Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1116/1116453.png", width=80)
    st.title("🔧 Dashboard Controls")

    # # Year Selector
    # st.subheader("📅 Select Year")
    # selected_year = st.slider("Year", 2000, datetime.now().year, 2022)

    # # City Selector
    # st.subheader("🏙️ Select City")
    # city = st.selectbox("City", ["New York", "London", "Tokyo", "Sydney", "Mumbai"])

    # Theme Selector
    st.subheader("🎨 Theme")
    selected_theme = st.radio("Choose Theme", ["Light", "Dark"], horizontal=True)

    # Credits
    st.markdown("---")
    st.markdown("Made with ❤️ by Manish Rai")

# Apply custom CSS based on theme
def set_theme(theme):
    if theme == "Dark":
        dark_css = """
            <style>
                body, .stApp {
                    background-color: #0e1117;
                    color: #fafafa;
                }
                .block-container {
                    background-color: #0e1117;
                }
            </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)
    else:
        light_css = """
            <style>
                body, .stApp {
                    background-color: #ffffff;
                    color: #000000;
                }
                .block-container {
                    background-color: #ffffff;
                }
            </style>
        """
        st.markdown(light_css, unsafe_allow_html=True)

set_theme(selected_theme)

# Hide default Streamlit system UI
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🌦️ Climate Data Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Explore global and city-specific weather trends, patterns, and ML-powered insights</h4>", unsafe_allow_html=True)

st.markdown("---")

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

st.markdown("---")
st.caption("© 2025 Climate Insights • Built with Streamlit")




