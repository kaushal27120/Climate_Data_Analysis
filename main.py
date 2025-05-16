# main.py
import pandas as pd
from functools import reduce
import os

def load_and_prepare_data(folder_path="historical-hourly-weather-dataset"):
    # Load CSVs
    files = {
        "humidity": "humidity.csv",
        "pressure": "pressure.csv",
        "temperature": "temperature.csv",
        "weather": "weather_description.csv",
        "wind_direction": "wind_direction.csv",
        "wind_speed": "wind_speed.csv"
    }

    dfs = {}
    for key, filename in files.items():
        df = pd.read_csv(os.path.join(folder_path, filename))
        dfs[key] = df.melt(id_vars=["datetime"], var_name="city", value_name=key)

    # Merge all dataframes
    df_final = reduce(lambda left, right: pd.merge(left, right, on=["datetime", "city"], how="outer"), dfs.values())
    
    # Clean and transform
    df_final.dropna(inplace=True)
    df_final["datetime"] = pd.to_datetime(df_final["datetime"])
    df_final["year"] = df_final["datetime"].dt.year
    df_final.set_index("datetime", inplace=True)
    df_final.sort_index(inplace=True)

    return df_final

# Save for Streamlit use
if __name__ == "__main__":
    df = load_and_prepare_data()
    df.to_csv("processed_climate_data.csv")
