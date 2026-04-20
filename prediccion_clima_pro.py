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
        "hourly": "temperature_2m",
        "timezone": TIMEZONE
    }

    response = requests.get(url, params=params)
    data = response.json()

    
    if "hourly" not in data:
        return pd.DataFrame()

    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temp": data["hourly"]["temperature_2m"]
    })

    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df = df.resample("h").mean()
    df["temp"] = df["temp"].interpolate()

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

    # Lags
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'lag_{lag}'] = df['temp'].shift(lag)

    
    df['roll_6'] = df['temp'].shift(1).rolling(6).mean()
    df['roll_24'] = df['temp'].shift(1).rolling(24).mean()

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
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1
    }

    print("Entrenando modelo")

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=500,
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

    # Contexto reciente
    df = obtener_historial(now - timedelta(days=7), now)

    history = df[['temp']].copy()

    for i in range(1, horas + 1):
        next_dt = now + timedelta(hours=i)

        # Se agrega fila futura vacía
        history.loc[next_dt] = np.nan

        
        feat = extraer_features(history)

        X_next = feat.loc[[next_dt]].drop(columns=['temp'])

        pred = model.predict(X_next)[0]

        history.loc[next_dt, 'temp'] = pred

    if fecha_objetivo not in history.index or pd.isna(history.loc[fecha_objetivo, 'temp']):
        if fecha_objetivo not in history.index:
            history.loc[fecha_objetivo] = np.nan
        feat = extraer_features(history)
        if fecha_objetivo in feat.index:
            X_obj = feat.loc[[fecha_objetivo]].drop(columns=['temp'])
            history.loc[fecha_objetivo, 'temp'] = model.predict(X_obj)[0]

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