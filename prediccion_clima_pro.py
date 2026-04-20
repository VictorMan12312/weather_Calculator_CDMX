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

LAT, LON = 19.43, -99.13
TIMEZONE = "America/Mexico_City"

# Límite de predicción hacia futuro
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
    df = df.resample("h").mean()
    
    # Interpolamos todas las columnas para que no haya huecos
    for col in df.columns:
        df[col] = df[col].interpolate()

    return df



def extraer_features(df):
    df = df.copy()

    horas = df.index.hour
    dias = df.index.dayofyear

    # Variables cíclicas
    df['sin_hour'] = np.sin(2 * np.pi * horas / 24)
    df['cos_hour'] = np.cos(2 * np.pi * horas / 24)
    df['sin_doy'] = np.sin(2 * np.pi * dias / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * dias / 365.25)

    # Lags de temperatura
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'lag_{lag}'] = df['temp'].shift(lag)

    # Lags de variables auxiliares (lo que pasó hace 1-6 horas)
    for col in ['humidity', 'pressure', 'clouds']:
        if col in df.columns:
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_lag_6'] = df[col].shift(6)
    
    df['roll_6'] = df['temp'].shift(1).rolling(6).mean()
    df['roll_24'] = df['temp'].shift(1).rolling(24).mean()

        # Mínimo y Máximo del día anterior 
    df['roll_24_min'] = df['temp'].shift(1).rolling(24).min()
    df['roll_24_max'] = df['temp'].shift(1).rolling(24).max()
    df['roll_24_std'] = df['temp'].shift(1).rolling(24).std() # cambio critico que hubo en el dia

    # Diferencias entre el dia anterior y el actual
    df['diff_24h'] = df['temp'].shift(1) - df['temp'].shift(25) # Diferencia con la misma hora
    df['diff_1h'] = df['temp'].shift(1) - df['temp'].shift(2)

    return df

#entrenamos el modelo 
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

    # Separación temporal
    n_val = int(len(X) * 0.1)

    X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
    y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'num_leaves': 31,
        'verbosity': -1
    }

    print("Entrenando modelo")

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=1500,
        callbacks=[lgb.early_stopping(50)]
    )

    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)

    print(f"MAE: {mae:.2f} °C")

    joblib.dump(model, MODEL_FILE)
    print("Modelo guardado.")

#prediccion usando los datos historicos y usando una alimetacion recursiva de los datos
def predecir_futuro(fecha_objetivo):

    if not os.path.exists(MODEL_FILE):
        print("Entrena el modelo primero.")
        return

    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    limite_inferior = now - timedelta(hours=1)

    if fecha_objetivo < limite_inferior:
        print("Fecha inválida. Tolerancia máxima de 1 hora en el pasado.")
        return

    horas = int((fecha_objetivo - now).total_seconds() / 3600)

    if horas / 24 > MAX_FORECAST_DAYS:
        print("Excede límite de días.")
        return

    model = joblib.load(MODEL_FILE)

    # 1. Obtener historial reciente para los lags
    df_hist = obtener_historial(now - timedelta(days=7), now)
    
    # 2. Obtener PRONÓSTICO de variables auxiliares (humedad, presión, nubes)
    # Esto es necesario para que el modelo "sepa" cómo estará el entorno en el futuro
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
        "temp": np.nan # La temperatura es lo que vamos a predecir
    }).set_index("time")

    # Combinamos historial con el hueco del futuro
    history = pd.concat([df_hist, df_fore[df_fore.index > df_hist.index.max()]])

    for i in range(1, horas + 1):
        next_dt = now + timedelta(hours=i)
        
        if next_dt not in history.index: continue

        # Generamos características usando los datos de atmósfera que ya tenemos del pronóstico
        feat = extraer_features(history)
        X_next = feat.loc[[next_dt]].drop(columns=['temp'])
        
        pred = model.predict(X_next)[0]
        history.loc[next_dt, 'temp'] = pred

    if fecha_objetivo not in history.index:
        print("Fecha fuera de rango de pronóstico.")
        return

    resultado = history.loc[fecha_objetivo, 'temp']

    print(f"\nTemperatura estimada: {resultado:.1f} °C")
    print(f"Fecha: {fecha_objetivo}")

def interfaz():
    while True:
        print("\n1. Entrenar")
        print("2. Predecir")
        print("3. Salir")

        op = input("Opción: ")

        if op == "1":
            entrenar_modelo()

        elif op == "2":
            try:
                y = int(input("Año: "))
                m = int(input("Mes: "))
                d = int(input("Día: "))
                h_str = input("Hora: ")
                h = int(h_str.split(":")[0]) if ":" in h_str else int(h_str)

                fecha = datetime(y, m, d, h)
                predecir_futuro(fecha)

            except:
                print("Error en datos.")

        elif op == "3":
            break


if __name__ == "__main__":
    interfaz()