import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from functools import reduce

st.title("🌦️ Climate Data Analysis and Weather Classification")

# Load data
@st.cache_data
def load_data():
    folder = "historical-hourly-weather-dataset/"
    df_humidity = pd.read_csv(folder + "humidity.csv")
    df_pressure = pd.read_csv(folder + "pressure.csv")
    df_temperature = pd.read_csv(folder + "temperature.csv")
    df_weather = pd.read_csv(folder + "weather_description.csv")
    df_wind_direction = pd.read_csv(folder + "wind_direction.csv")
    df_wind_speed = pd.read_csv(folder + "wind_speed.csv")

    df_humidity_long = df_humidity.melt(id_vars=["datetime"], var_name="city", value_name="humidity")
    df_pressure_long = df_pressure.melt(id_vars=["datetime"], var_name="city", value_name="pressure")
    df_temperature_long = df_temperature.melt(id_vars=["datetime"], var_name="city", value_name="temperature")
    df_weather_long = df_weather.melt(id_vars=["datetime"], var_name="city", value_name="weather")
    df_wind_direction_long = df_wind_direction.melt(id_vars=["datetime"], var_name="city", value_name="wind_direction")
    df_wind_speed_long = df_wind_speed.melt(id_vars=["datetime"], var_name="city", value_name="wind_speed")

    dfs = [df_weather_long, df_temperature_long, df_humidity_long, df_pressure_long, df_wind_direction_long, df_wind_speed_long]
    df_final = reduce(lambda left, right: pd.merge(left, right, on=["datetime", "city"], how="outer"), dfs)

    df_final.dropna(inplace=True)
    df_final["datetime"] = pd.to_datetime(df_final["datetime"])
    df_final.set_index("datetime", inplace=True)
    df_final.sort_index(inplace=True)

    return df_final

df_final = load_data()

# City selector
city = st.selectbox("Select a city", df_final["city"].unique())
df_city = df_final[df_final["city"] == city]

# Temperature plot
st.subheader(f"📈 Temperature Trend in {city}")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_city.index, df_city["temperature"], color='red')
ax.set_title(f"Temperature Over Time - {city}")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (K)")
st.pyplot(fig)

# Humidity vs Temp
st.subheader("☁️ Humidity vs Temperature")
fig2, ax2 = plt.subplots()
sns.scatterplot(x="temperature", y="humidity", data=df_final, ax=ax2)
st.pyplot(fig2)

# Box Plots
st.subheader("📊 Box Plots for Outlier Detection")
numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
fig3, axs = plt.subplots(2, 2, figsize=(14, 8))
axs = axs.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(x=df_final[col], ax=axs[i], color='skyblue')
    axs[i].set_title(f'Box Plot of {col}')
st.pyplot(fig3)

# Avg temp by city
st.subheader("🌡️ Average Temperature by City")
fig4, ax4 = plt.subplots()
df_final.groupby("city")["temperature"].mean().sort_values().plot(kind='barh', ax=ax4)
st.pyplot(fig4)

# Weather condition frequency
st.subheader("🌤️ Weather Condition Frequency")
fig5, ax5 = plt.subplots()
df_final["weather"].value_counts().plot(kind="bar", ax=ax5)
st.pyplot(fig5)

# Wind speed distribution
st.subheader("🍃 Wind Speed Distribution")
fig6, ax6 = plt.subplots()
sns.histplot(df_final["wind_speed"], bins=10, kde=True, ax=ax6)
st.pyplot(fig6)

# Correlation heatmap
st.subheader("📉 Correlation Heatmap")
fig7, ax7 = plt.subplots()
sns.heatmap(df_final[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax7)
st.pyplot(fig7)

# Random Forest classifier
st.subheader("🤖 Weather Classification Accuracy")
le = LabelEncoder()
df_final["weather_encoded"] = le.fit_transform(df_final["weather"])
X = df_final[numeric_cols]
y = df_final["weather_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: **{acc:.2%}**")
