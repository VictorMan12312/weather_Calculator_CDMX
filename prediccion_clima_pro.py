import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib
import warnings
import requests

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

MODEL_FILE = "lgb_clima_pro.pkl"
LAT, LON = 19.5047, -99.1469
TIMEZONE = "America/Mexico_City"
MAX_FORECAST_DAYS = 7

def obtener_historial(start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloudcover",
        "timezone": TIMEZONE
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "hourly" not in data:
        return pd.DataFrame()

    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temp": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "pressure": data["hourly"]["surface_pressure"],
        "clouds": data["hourly"]["cloudcover"]
    })

    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df = df.resample("30min").interpolate()
    
    for col in df.columns:
        df[col] = df[col].interpolate()

    return df

def extraer_features(df):
    df = df.copy()
    horas = df.index.hour
    dias = df.index.dayofyear

    df['sin_hour'] = np.sin(2 * np.pi * horas / 24)
    df['cos_hour'] = np.cos(2 * np.pi * horas / 24)
    df['sin_doy'] = np.sin(2 * np.pi * dias / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * dias / 365.25)

    for lag in [1, 2, 6, 12, 24, 48, 96]:
        df[f'lag_{lag}'] = df['temp'].shift(lag)

    for col in ['humidity', 'pressure', 'clouds']:
        if col in df.columns:
            df[f'{col}_lag_2'] = df[col].shift(2)

    df['roll_6'] = df['temp'].shift(1).rolling(12).mean()
    df['roll_24'] = df['temp'].shift(1).rolling(48).mean()
    df['roll_24_min'] = df['temp'].shift(1).rolling(48).min()
    df['roll_24_max'] = df['temp'].shift(1).rolling(48).max()
    df['roll_24_std'] = df['temp'].shift(1).rolling(48).std()

    df['diff_24h'] = df['temp'].shift(1) - df['temp'].shift(49)
    df['diff_1h'] = df['temp'].shift(1) - df['temp'].shift(3)

    return df

def entrenar_modelo():
    print("Descargando historial")
    start = datetime(2010, 1, 1)
    end = datetime.now()

    raw_df = obtener_historial(start, end)
    if raw_df.empty:
        print("Error descargando datos.")
        return

    print("Generando variables")
    df_feat = extraer_features(raw_df).dropna()
    X = df_feat.drop(columns=['temp'])
    y = df_feat['temp']

    n_val = int(len(X) * 0.1)
    X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
    y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.02,
        'feature_fraction': 0.9,
        'num_leaves': 40,
        'verbosity': -1
    }

    print("Entrenando modelo")
    model = lgb.train(
        params, dtrain, valid_sets=[dval],
        num_boost_round=1500,
        callbacks=[lgb.early_stopping(100)]
    )

    mae = mean_absolute_error(y_val, model.predict(X_val))
    print(f"MAE: {mae:.2f} °C")
    joblib.dump(model, MODEL_FILE)
    print("Modelo guardado.")

def predecir_futuro(fecha_objetivo):
    if not os.path.exists(MODEL_FILE):
        print("Entrena el modelo primero.")
        return

    fecha_objetivo = fecha_objetivo.replace(second=0, microsecond=0)
    if fecha_objetivo.minute < 15: fecha_objetivo = fecha_objetivo.replace(minute=0)
    elif fecha_objetivo.minute < 45: fecha_objetivo = fecha_objetivo.replace(minute=30)
    else: fecha_objetivo = (fecha_objetivo + timedelta(hours=1)).replace(minute=0)

    now_raw = datetime.now()
    now = now_raw.replace(minute=0 if now_raw.minute < 30 else 30, second=0, microsecond=0)
    limite_inferior = now - timedelta(hours=1)

    if fecha_objetivo < limite_inferior:
        print("Fecha inválida. Tolerancia máxima de 1 hora en el pasado.")
        return

    pasos = int((fecha_objetivo - now).total_seconds() / 1800)
    if pasos / 48 > MAX_FORECAST_DAYS:
        print("Excede límite de días.")
        return

    model = joblib.load(MODEL_FILE)
    df_hist = obtener_historial(now - timedelta(days=7), now)

    url_forecast = "https://api.open-meteo.com/v1/forecast"
    params_f = {
        "latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
        "hourly": "relative_humidity_2m,surface_pressure,cloudcover"
    }
    resp_f = requests.get(url_forecast, params=params_f).json()
    
    df_fore = pd.DataFrame({
        "time": pd.to_datetime(resp_f["hourly"]["time"]),
        "humidity": resp_f["hourly"]["relative_humidity_2m"],
        "pressure": resp_f["hourly"]["surface_pressure"],
        "clouds": resp_f["hourly"]["cloudcover"],
        "temp": np.nan
    }).set_index("time").resample("30min").interpolate()

    history = pd.concat([df_hist, df_fore[df_fore.index > df_hist.index.max()]])

    if fecha_objetivo > now:
        for i in range(1, pasos + 1):
            next_dt = now + timedelta(minutes=30 * i)
            if next_dt not in history.index: continue
            feat = extraer_features(history)
            X_next = feat.loc[[next_dt]].drop(columns=['temp'])
            history.loc[next_dt, 'temp'] = model.predict(X_next)[0]

    resultado = history.loc[fecha_objetivo, 'temp']
    if pd.isna(resultado):
        feat = extraer_features(history)
        X_obj = feat.loc[[fecha_objetivo]].drop(columns=['temp'])
        resultado = model.predict(X_obj)[0]

    print(f"\nTemperatura estimada: {resultado:.1f} °C")
    print(f"Fecha: {fecha_objetivo}")

def predecir_tiempo_real():
    if not os.path.exists(MODEL_FILE):
        print("Entrena el modelo primero.")
        return

    now_raw = datetime.now()
    now = now_raw.replace(minute=0 if now_raw.minute < 30 else 30, second=0, microsecond=0)
    model = joblib.load(MODEL_FILE)

    df_hist = obtener_historial(now - timedelta(days=7), now)
    url_forecast = "https://api.open-meteo.com/v1/forecast"
    params_f = {
        "latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
        "current": "temperature_2m",
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloudcover",
        "past_days": 1
    }

    resp_f = requests.get(url_forecast, params=params_f).json()
    temp_now = resp_f.get("current", {}).get("temperature_2m", np.nan)

    df_fore = pd.DataFrame({
        "time": pd.to_datetime(resp_f["hourly"]["time"]),
        "temp": resp_f["hourly"]["temperature_2m"],
        "humidity": resp_f["hourly"]["relative_humidity_2m"],
        "pressure": resp_f["hourly"]["surface_pressure"],
        "clouds": resp_f["hourly"]["cloudcover"]
    }).set_index("time").resample("30min").interpolate()

    history = pd.concat([df_hist, df_fore[df_fore.index > df_hist.index.max()]])
    feat = extraer_features(history)

    if now not in feat.index:
        print("No hay datos disponibles.")
        return

    X_now = feat.loc[[now]].drop(columns=['temp'])
    pred = model.predict(X_now)[0]
    final = 0.5 * temp_now + 0.5 * pred if not np.isnan(temp_now) else pred

    print("\n--- TIEMPO REAL ---")
    print(f"Hora: {now}")
    print(f"Temperatura real (API): {temp_now:.1f} °C")
    print(f"Predicción ajustada: {final:.1f} °C")
    print(f"Predicción modelo: {pred:.1f} °C")

def interfaz():
    while True:
        print("\n1. Entrenar")
        print("2. Predecir")
        print("3. Tiempo Real")
        print("4. Salir")

        op = input("Opción: ")
        if op == "1":
            entrenar_modelo()
        elif op == "2":
            try:
                y, m, d = int(input("Año: ")), int(input("Mes: ")), int(input("Día: "))
                h, mi = int(input("Hora: ")), int(input("Minuto (0 o 30): "))
                predecir_futuro(datetime(y, m, d, h, mi))
            except:
                print("Error en datos.")
        elif op == "3":
            predecir_tiempo_real()
        elif op == "4":
            break

if __name__ == "__main__":
    interfaz()