# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

# Page setup
st.set_page_config(layout="wide", page_title="🌦️ Climate Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("processed_climate_data.csv", parse_dates=["datetime"], index_col="datetime")
    return df

df = load_data()

# Sidebar
st.sidebar.title("🔧 Settings")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()))
theme = st.sidebar.selectbox("Select Theme", ["Blues", "Viridis", "Plasma", "Inferno"])

df_year = df[df["year"] == selected_year]
cities = df_year["city"].unique()

# Title
st.title(f"🌍 Climate Data Dashboard - {selected_year}")

# KPIs
col1, col2, col3 = st.columns(3)
avg_temp = df_year.groupby("city")["temperature"].mean()
hottest_city = avg_temp.idxmax()
coldest_city = avg_temp.idxmin()

col1.metric("🔥 Hottest City", hottest_city, f"{avg_temp.max():.2f} K")
col2.metric("❄️ Coldest City", coldest_city, f"{avg_temp.min():.2f} K")
col3.metric("📊 Cities Tracked", len(cities))

# Toggle for Temperature Line Chart
if st.checkbox("📈 Show Temperature Trend by City"):
    city_sel = st.selectbox("Select a city", cities)
    df_city = df_year[df_year["city"] == city_sel]
    fig1 = px.line(df_city.reset_index(), x="datetime", y="temperature", title=f"Temperature Over Time - {city_sel}")
    st.plotly_chart(fig1, use_container_width=True)

# Toggle for Avg Temperature Bar Chart
if st.checkbox("🌡️ Show Average Temperature by City"):
    fig2 = px.bar(avg_temp.sort_values(), orientation='h', color=avg_temp.sort_values(), color_continuous_scale=theme.lower())
    st.plotly_chart(fig2, use_container_width=True)

# Dial Charts for extreme temperature days
if st.checkbox("📊 Show Temperature Distribution Gauges"):
    high_temp_pct = (df_year[df_year["temperature"] > df_year["temperature"].quantile(0.75)].shape[0] / df_year.shape[0]) * 100
    low_temp_pct = (df_year[df_year["temperature"] < df_year["temperature"].quantile(0.25)].shape[0] / df_year.shape[0]) * 100

    col4, col5 = st.columns(2)
    with col4:
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=high_temp_pct,
            title={'text': "High Temp Days"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
        ))
        st.plotly_chart(fig3, use_container_width=True)
    with col5:
        fig4 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=low_temp_pct,
            title={'text': "Low Temp Days"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}
        ))
        st.plotly_chart(fig4, use_container_width=True)

# Heatmap of temperature over years
if st.checkbox("🌡️ Show Yearly Temperature Heatmap Across Cities"):
    pivot_table = df.groupby([df.index.year, "city"])["temperature"].mean().unstack()
    fig5, ax5 = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot_table, cmap=theme.lower(), ax=ax5)
    st.pyplot(fig5)
    plt.close(fig5)

# Boxplots
if st.checkbox("📦 Show Climate Variable Box Plots"):
    numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
    fig6, axs = plt.subplots(2, 2, figsize=(14, 8))
    axs = axs.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axs[i], color='skyblue')
        axs[i].set_title(f'{col.title()}')
    st.pyplot(fig6)
    plt.close(fig6)

# Weather frequency
if st.checkbox("🌤️ Show Weather Condition Frequency"):
    fig7, ax7 = plt.subplots()
    df["weather"].value_counts().plot(kind="bar", ax=ax7, color="orange")
    ax7.set_title("Weather Descriptions")
    st.pyplot(fig7)
    plt.close(fig7)

# Wind speed distribution
if st.checkbox("🌬️ Show Wind Speed Distribution"):
    fig8, ax8 = plt.subplots()
    sns.histplot(df["wind_speed"], bins=20, kde=True, ax=ax8)
    st.pyplot(fig8)
    plt.close(fig8)

# Correlation heatmap
if st.checkbox("🔗 Show Correlation Heatmap"):
    fig9, ax9 = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax9)
    st.pyplot(fig9)
    plt.close(fig9)

# Weather classification model
if st.checkbox("🤖 Show Weather Classification Model Accuracy"):
    le = LabelEncoder()
    df["weather_encoded"] = le.fit_transform(df["weather"])
    X = df[numeric_cols]
    y = df["weather_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.metric("🎯 Classification Accuracy", f"{acc:.2%}")

# Footer
st.markdown("---")
st.markdown("📊 **Data Source**: Historical Hourly Weather Dataset")
st.markdown("🧠 **Developed by Manish Rai | Inspired by US Census Dashboard**")
