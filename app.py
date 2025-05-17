import streamlit as st

# Page settings
st.set_page_config(page_title="🌦️ Climate Dashboard", layout="wide")

# Inject custom CSS
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom right, #e0f7fa, #ffffff);
        }
        .main {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            animation: fadeIn 1.2s ease-in;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        h1 {
            color: #00796b;
            font-weight: 700;
        }
        .footer {
            margin-top: 100px;
            padding-top: 20px;
            border-top: 1px solid #ccc;
            text-align: center;
            color: #666;
            font-size: 0.95rem;
        }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1>🌦️ Climate Data Dashboard</h1>", unsafe_allow_html=True)

# Welcome content
st.markdown("""
<div class="main">
    <p>Welcome to the <strong>Climate Data Dashboard</strong>! 🌍</p>

    <p>Navigate using the sidebar to explore:</p>
    <ul>
        <li>🔍 Overall trends and KPIs</li>
        <li>📈 City-specific analysis</li>
        <li>🗺️ Interactive map of climate data</li>
        <li>📊 Distributions and weather patterns</li>
        <li>🤖 Weather classification using machine learning</li>
    </ul>

    <p>Use the sidebar to select year, city, and theme across all pages.</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>This project is made by:</strong></p>
    <p>1. Manish Rai &nbsp;&nbsp;&nbsp; 2. Kaushal Jashani &nbsp;&nbsp;&nbsp; 3. Darpan Madhvi</p>
</div>
""", unsafe_allow_html=True)
