# 🌦️ Climate Data Dashboard

> A modern, interactive dashboard built with Streamlit to explore global and city-level climate data — powered by data visualization, machine learning, and clean UI/UX.

<!-- ![Dashboard Preview](https://your-image-link-or-screenshot.png) -->

---

## 🚀 Features

- 📊 **Visual Analytics**: Year-wise climate data (temperature, humidity, wind, rainfall)
- 🏙️ **City-Level Insights**: Select city & year dynamically for detailed trends
- 🗺️ **Interactive Map**: Global climate overview with geospatial visualization
- 🤖 **ML Weather Classification**: Predict weather categories using real data
- 🎨 **Light & Dark Theme**: Toggle UI theme for personalized experience
- ⚡ **Fast & Responsive**: Designed with performance and smooth UX in mind

---

## 🧠 Tech Stack

| Layer          | Technologies |
|----------------|--------------|
| **Frontend**   | [Streamlit](https://streamlit.io), HTML/CSS (custom styles) |
| **Data**       | Pandas, NumPy, GeoJSON |
| **Visualization** | Plotly, Altair, Folium, Matplotlib |
| **Machine Learning** | Scikit-learn (Logistic Regression, Random Forest, etc.) |
| **Deployment** | Streamlit Cloud / Docker-ready |

---

## 📂 Project Structure

```bash
.
├── app.py               # Main dashboard entry point
├── pages/               # Sub-pages (Overview, Map, City Analysis, etc.)
├── data/                # Climate data files
├── models/              # Trained ML models
├── utils/               # Helper scripts (e.g., preprocessing, visual utils)
├── .streamlit/          # Theme config and settings
└── README.md            # Project documentation
```

---

## 📸 Screenshots

| Light Mode | Dark Mode |
|------------|-----------|
| ![Light](https://your-light-mode-screenshot.png) | ![Dark](https://your-dark-mode-screenshot.png) |

---

## 📈 How to Run

### 🔧 Setup

```bash
git clone https://github.com/your-username/climate-dashboard.git
cd climate-dashboard
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧠 Machine Learning Overview

- Cleaned and preprocessed historical weather data
- Feature selection and scaling applied
- Multiple models trained and compared (Logistic Regression, Random Forest, XGBoost)
- Best-performing model used for prediction on unseen data
- Interactive SHAP plots to explain model output (optional)

---

## 🛠️ Future Improvements

- 🌍 Add real-time API data (e.g., OpenWeatherMap)
- 🧭 Time-series forecasting for temperature & rainfall
- 🔐 User authentication and personalized dashboard
- 📱 Mobile-friendly responsive layout

---

## 👨‍💻 Author

**Manish Rai**  
📧 [hire.manishrai@gmail.com] • 🌐 [LinkedIn](https://www.linkedin.com/in/manishkumarrai98/) • 🐍 Python | 📊 Data | ☁️ Azure

---

## ⭐ Show Your Support

If you like this project, please ⭐ the repo and share it with others!

---

## 📜 License

MIT License - feel free to use, modify, and build upon it.
