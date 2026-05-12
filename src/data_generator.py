import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_cgm_data(days=90, seed=42):
    np.random.seed(seed)

    total_minutes = days * 24 * 60
    n_readings = total_minutes // 5
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n_readings)]

    glucose = np.zeros(n_readings)
    basal = 100.0

    for i in range(n_readings):
        t = timestamps[i]
        hour = t.hour + t.minute / 60.0

        # Circadian rhythm
        circadian = 10 * np.sin(2 * np.pi * (hour - 6) / 24)

        # Meal spikes
        meal_effect = 0.0
        meal_effect += 55 * np.exp(-0.5 * ((hour - 7.5) / 1.5) ** 2)   # breakfast
        meal_effect += 45 * np.exp(-0.5 * ((hour - 12.5) / 1.5) ** 2)  # lunch
        meal_effect += 60 * np.exp(-0.5 * ((hour - 18.5) / 1.8) ** 2)  # dinner

        # Overnight dip
        night_dip = 0.0
        if 1.0 <= hour <= 4.0:
            night_dip = -8 * np.sin(np.pi * (hour - 1) / 3)

        # Sensor noise
        noise = np.random.normal(0, 3)

        glucose[i] = basal + circadian + meal_effect + night_dip + noise

    glucose = np.clip(glucose, 40, 400)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "glucose_mg_dl": glucose
    })

    # Inject 2% missing values (sensor dropouts)
    dropout_mask = np.random.rand(n_readings) < 0.02
    df.loc[dropout_mask, "glucose_mg_dl"] = np.nan

    df["patient_id"] = "P001"
    df["device"] = "Dexcom_G6_sim"

    return df


if __name__ == "__main__":
    df = generate_cgm_data(days=90)
    df.to_csv("data/raw/cgm_raw.csv", index=False)
    print(f"Generated {len(df):,} readings")
    print(f"Missing values: {df['glucose_mg_dl'].isna().sum()}")
    print(f"Glucose range: {df['glucose_mg_dl'].min():.1f} – {df['glucose_mg_dl'].max():.1f} mg/dL")
    print("Saved to data/raw/cgm_raw.csv")