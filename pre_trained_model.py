import pandas as pd
import os
import joblib
from main import load_and_prepare_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def feature_engineering(df):
    df = df.copy()
    df["temp_c"] = df["temperature"] - 273.15
    df["wind_cat"] = pd.cut(df["wind_speed"], bins=[0,3,7,15,50], labels=["calm","breeze","windy","storm"])
    df.dropna(inplace=True)
    return df

def train_and_save():
    print("🔍 Loading and preparing data...")
    df = load_and_prepare_data()
    df = feature_engineering(df)
    df_year = df[df["year"] == df["year"].max()].copy()

    df_sample = (
        df_year.groupby("weather", group_keys=False)
        .apply(lambda x: x.sample(frac=0.7, random_state=42))
        .reset_index(drop=True)
    )

    print(f"📊 Training sample size: {len(df_sample)}")

    le = LabelEncoder()
    df_sample["weather_encoded"] = le.fit_transform(df_sample["weather"])
    df_sample["wind_cat_enc"] = LabelEncoder().fit_transform(df_sample["wind_cat"])

    features = ["temp_c", "humidity", "pressure", "wind_cat_enc"]
    X = df_sample[features]
    y = df_sample["weather_encoded"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model parameters
    param_grids = {
        "Random Forest": {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'max_depth': [10],
                'min_samples_split': [2]
            }
        },
        "Gradient Boosting": {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'learning_rate': [0.1],
                'max_depth': [3]
            }
        },
        "Extra Trees": {
            'model': ExtraTreesClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'max_depth': [10]
            }
        },
        "Logistic Regression": {
            'model': LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto'),
            'params': {
                'C': [1.0],
                'max_iter': [1000]
            }
        },
        "Decision Tree": {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [10],
                'min_samples_split': [2]
            }
        }
    }

    os.makedirs("saved_models", exist_ok=True)
    trained_models = {}
    model_scores = {}

    print("🚀 Training models...")
    for name in tqdm(param_grids, desc="🔄 Training models"):
        base_model = param_grids[name]["model"]
        params = param_grids[name]["params"]
        grid = GridSearchCV(base_model, params, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ {name} - Accuracy: {acc:.4f}, Best Params: {grid.best_params_}")

        # Save model with compression
        filename = f"saved_models/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(best_model, filename, compress=9)

        trained_models[name] = best_model
        model_scores[name] = acc

    # Train ensemble
    print("🧠 Training Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    ensemble_acc = accuracy_score(y_test, ensemble.predict(X_test))
    model_scores["Voting Ensemble"] = ensemble_acc
    joblib.dump(ensemble, "saved_models/ensemble.pkl", compress=9)

    # Save encoders/scaler and scores with compression
    joblib.dump(le, "saved_models/label_encoder.pkl", compress=9)
    joblib.dump(scaler, "saved_models/scaler.pkl", compress=9)
    joblib.dump(model_scores, "saved_models/model_scores.pkl", compress=9)

    print("\n🎉 All models trained and saved successfully!")
    print("📁 Models saved in 'saved_models/' directory.")
    print("📈 Accuracy scores saved to 'model_scores.pkl'")

if __name__ == "__main__":
    train_and_save()
