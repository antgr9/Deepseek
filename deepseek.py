import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time, timedelta
import yfinance as yf
import os

# Zona horaria NY
ty_ny = pytz.timezone("America/New_York")

# --- Parámetros de trading ---
STOP_LOSS = 4         # puntos
TAKE_PROFIT = 15      # puntos
EVAL_VELAS = 5        # número de velas para evaluar la señal (configurable)

# ====== FUNCIONES DE AYUDA ======

def obtener_datos(ticker="^GSPC", intervalo="1m", periodo="1d"):
    df = yf.Ticker(ticker).history(period=periodo, interval=intervalo)
    if df.empty:
        return pd.DataFrame()
    if df.index.tz is None:
        df.index = df.index.tz_localize(pytz.utc).tz_convert(ty_ny)
    else:
        df.index = df.index.tz_convert(ty_ny)
    return df

def dentro_de_horario(dt, hora_inicio, hora_fin):
    # dt es datetime con tz local NY
    t = dt.time()
    return hora_inicio <= t <= hora_fin

# ====== FUNCIONES MÓDULOS ======

def run_anomalias_retornos(hora_inicio, hora_fin):
    df = obtener_datos()
    if df.empty:
        return pd.DataFrame(columns=['Close', 'z_score', 'Direccion'])
    
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    mean = df['log_ret'].rolling(window=20).mean()
    std = df['log_ret'].rolling(window=20).std()
    df['z_score'] = (df['log_ret'] - mean) / std
    df.dropna(inplace=True)
    
    ultimas = df.tail(10)
    anomalias = ultimas[abs(ultimas['z_score']) > 2]
    
    # Filtrar por horario
    anomalias = anomalias[anomalias.index.to_series().apply(lambda x: dentro_de_horario(x, hora_inicio, hora_fin))]
    
    # Verificar si hay anomalías y si las columnas existen
    if anomalias.empty or not all(col in anomalias.columns for col in ['Close', 'z_score']):
        return pd.DataFrame(columns=['Close', 'z_score', 'Direccion'])
    
    resultado = anomalias[['Close', 'z_score']].copy()
    resultado['Direccion'] = np.where(resultado['z_score'] > 0, 'ALZA', 'BAJA')
    return resultado

def clasificar_vela(open_, high, low, close):
    cuerpo = abs(close - open_)
    rango_total = high - low
    upper = high - max(open_, close)
    lower = min(open_, close) - low
    if rango_total == 0:
        return "Neutral"
    if cuerpo / rango_total < 0.3 and (upper / rango_total > 0.35 or lower / rango_total > 0.35):
        return "Absorcion"
    elif close > open_ and close > (high - 0.2 * rango_total):
        return "Intencion_Alcista"
    elif close < open_ and close < (low + 0.2 * rango_total):
        return "Intencion_Bajista"
    else:
        return "Neutral"

def run_volumen_vela(hora_inicio, hora_fin):
    df = obtener_datos()
    if df.empty:
        return "Sin datos"
    
    df['clasificacion'] = df.apply(lambda row: clasificar_vela(row['Open'], row['High'], row['Low'], row['Close']), axis=1)
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    if df.empty or not dentro_de_horario(df.index[-1], hora_inicio, hora_fin):
        return "Fuera de horario configurado"
    
    ult = df.iloc[-1]
    if 'clasificacion' not in ult or 'vol_ratio' not in ult:
        return "Datos incompletos"
    
    if ult['clasificacion'] in ['Intencion_Alcista', 'Intencion_Bajista'] and ult['vol_ratio'] > 1.5:
        return f"Confluencia: {ult['clasificacion']} con volumen alto ({ult['vol_ratio']:.2f})"
    return "Sin confluencia clara"

def run_orderflow_sintetico(hora_inicio, hora_fin):
    df = obtener_datos()
    if df.empty:
        return pd.DataFrame(columns=['Close', 'Clasificacion'])
    
    df['Clasificacion'] = df.apply(lambda row: clasificar_vela(row['Open'], row['High'], row['Low'], row['Close']), axis=1)
    recientes = df[df['Clasificacion'] != 'Neutral'].tail(10)
    recientes = recientes[recientes.index.to_series().apply(lambda x: dentro_de_horario(x, hora_inicio, hora_fin))]
    
    if recientes.empty:
        return pd.DataFrame(columns=['Close', 'Clasificacion'])
    
    return recientes[['Close', 'Clasificacion']]

def run_reversion_heuristica(hora_inicio, hora_fin):
    df = yf.Ticker("ES=F").history(period="5d", interval="5m")
    if df.empty:
        return "Sin datos"
    
    if df.index.tz is None:
        df.index = df.index.tz_localize(pytz.utc).tz_convert(ty_ny)
    else:
        df.index = df.index.tz_convert(ty_ny)

    # Filtrar horario
    df = df[df.index.to_series().apply(lambda x: dentro_de_horario(x, hora_inicio, hora_fin))]

    if df.empty:
        return "No hay datos en el horario configurado"

    df['ret'] = np.log(df['Close'].pct_change() + 1)
    df['cum_ret_lb'] = df['ret'].rolling(window=6).sum()
    mean = df['cum_ret_lb'].rolling(120).mean()
    std = df['cum_ret_lb'].rolling(120).std().replace(0, np.nan)
    df['z_cum_ret'] = (df['cum_ret_lb'] - mean) / std
    df['dir'] = np.where(df['ret'] > 0, 1, -1)
    df['run'] = df['dir'].groupby((df['dir'] != df['dir'].shift()).cumsum()).cumcount() + 1
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    last = df.iloc[-1]
    prob = 20 + (20 if abs(last['z_cum_ret']) > 1.5 else 0) + (10 if abs(last['z_cum_ret']) > 2.0 else 0)
    prob += (10 if last['vol_ratio'] > 1.5 else 0) + (10 if last['run'] >= 3 else 0)
    prob = min(prob, 95)
    direccion = "ABAJO" if last['cum_ret_lb'] > 0 else "ARRIBA"

    return f"Prob. reversión {direccion}: {prob}% | z={last['z_cum_ret']:.2f} | volumen={last['vol_ratio']:.2f} | streak={int(last['run'])}"

# ====== FUNCIONES DE EVALUACION Y GUARDADO ======

def cargar_resultados_csv():
    fecha = datetime.now(ty_ny).strftime("%Y%m%d")
    archivo = f"resultados_sesion_{fecha}.csv"
    if os.path.exists(archivo):
        return pd.read_csv(archivo, parse_dates=['timestamp'])
    else:
        cols = ['timestamp', 'modulo', 'tipo', 'direccion', 'precio_entrada', 'resultado', 'puntos', 'duracion']
        return pd.DataFrame(columns=cols)

def guardar_resultado(df):
    fecha = datetime.now(ty_ny).strftime("%Y%m%d")
    archivo = f"resultados_sesion_{fecha}.csv"
    df.to_csv(archivo, index=False)

def evaluar_senal_ultima():
    df_historico = cargar_resultados_csv()
    df_senales = st.session_state.get("senal_consolidada", None)
    if df_senales is None or df_senales.empty:
        st.warning("No hay señales consolidadas para evaluar.")
        return

    ultima = df_senales.iloc[-1]
    ts = pd.to_datetime(ultima['timestamp'])
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        ts = ts.tz_localize(ty_ny)
    else:
        ts = ts.tz_convert(ty_ny)

    direccion = ultima['direccion']
    precio_entrada = ultima['precio_entrada']

    fin_eval = ts + timedelta(minutes=EVAL_VELAS)
    df = obtener_datos(intervalo="1m", periodo="5d")
    df_eval = df[(df.index > ts) & (df.index <= fin_eval)]

    if df_eval.empty:
        st.warning("No hay datos suficientes para evaluar la señal.")
        return

    resultado = "NEUTRO"
    puntos = 0
    duracion = len(df_eval)

    for idx, row in df_eval.iterrows():
        diff = row['Close'] - precio_entrada
        if direccion == 'ALZA':
            if diff >= TAKE_PROFIT:
                resultado = "TP"
                puntos = TAKE_PROFIT
                duracion = int((idx - ts).total_seconds() // 60) + 1
                break
            elif diff <= -STOP_LOSS:
                resultado = "SL"
                puntos = -STOP_LOSS
                duracion = int((idx - ts).total_seconds() // 60) + 1
                break
        elif direccion == 'BAJA':
            diff = precio_entrada - row['Close']
            if diff >= TAKE_PROFIT:
                resultado = "TP"
                puntos = TAKE_PROFIT
                duracion = int((idx - ts).total_seconds() // 60) + 1
                break
            elif diff <= -STOP_LOSS:
                resultado = "SL"
                puntos = -STOP_LOSS
                duracion = int((idx - ts).total_seconds() // 60) + 1
                break

    nueva_fila = pd.DataFrame([{
        'timestamp': ts,
        'modulo': 'Consolidado',
        'tipo': 'Señal final',
        'direccion': direccion,
        'precio_entrada': precio_entrada,
        'resultado': resultado,
        'puntos': puntos,
        'duracion': duracion
    }])

    df_historico = pd.concat([df_historico, nueva_fila], ignore_index=True)
    guardar_resultado(df_historico)
    st.session_state["senal_evaluada"] = df_historico

    st.success(f"Señal evaluada: {resultado} | Puntos: {puntos} | Duración: {duracion} velas")

def resumen_efectividad(df):
    if df.empty:
        st.write("No hay señales evaluadas aún.")
        return

    total = len(df)
    tp = len(df[df['resultado'] == 'TP'])
    sl = len(df[df['resultado'] == 'SL'])
    neutro = len(df[df['resultado'] == 'NEUTRO'])
    puntos_prom = df['puntos'].mean()

    st.write(f"Total señales evaluadas: {total}")
    st.write(f"% TP (aciertos): {tp/total*100:.1f}%")
    st.write(f"% SL (fallos): {sl/total*100:.1f}%")
    st.write(f"% Neutro: {neutro/total*100:.1f}%")
    st.write(f"Puntos promedio ganados/perdidos: {puntos_prom:.2f}")

# ====== INTERFAZ STREAMLIT ======

st.set_page_config(page_title="Scalping Dashboard S&P 500", layout="wide")
st.sidebar.title("Herramientas Cuantitativas")

# Sidebar para configurar horario
st.sidebar.markdown("## ⏰ Configuración de horario para señales")
hora_inicio = st.sidebar.time_input("Hora inicio (NY)", value=time(8,30))
hora_fin = st.sidebar.time_input("Hora fin (NY)", value=time(16,0))
if hora_fin <= hora_inicio:
    st.sidebar.error("La hora fin debe ser mayor que la hora inicio")

# Sidebar para configurar número de velas para evaluar
EVAL_VELAS = st.sidebar.number_input("Velas para evaluar señal", min_value=1, max_value=15, value=5, step=1)

opcion = st.sidebar.radio("Selecciona una herramienta:", [
    "1️⃣ Anomalías de Rendimiento",
    "2️⃣ Confluencia Vela + Volumen",
    "3️⃣ Order Flow Sintético",
    "4️⃣ Reversión Heurística",
    "📊 Consolidado General"
])

st.title("📈 Dashboard de Apoyo para Scalping S&P 500")

if opcion.startswith("1"):
    st.subheader("Anomalías de Rendimiento")
    resultado = run_anomalias_retornos(hora_inicio, hora_fin)
    if resultado.empty:
        st.write("No se encontraron anomalías significativas")
    else:
        st.dataframe(resultado)

elif opcion.startswith("2"):
    st.subheader("Confluencia entre Vela y Volumen")
    st.info(run_volumen_vela(hora_inicio, hora_fin))

elif opcion.startswith("3"):
    st.subheader("Order Flow Sintético")
    resultado = run_orderflow_sintetico(hora_inicio, hora_fin)
    if resultado.empty:
        st.write("No se encontraron patrones significativos")
    else:
        st.dataframe(resultado)

elif opcion.startswith("4"):
    st.subheader("Modelo Heurístico de Reversión")
    st.warning(run_reversion_heuristica(hora_inicio, hora_fin))

elif opcion.startswith("📊"):
    st.subheader("📊 Consolidado de Señales Cuantitativas")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Anomalías:")
        res1 = run_anomalias_retornos(hora_inicio, hora_fin)
        if res1.empty:
            st.write("No se encontraron anomalías significativas")
        else:
            st.dataframe(res1)

    with col2:
        st.markdown("### Order Flow:")
        res3 = run_orderflow_sintetico(hora_inicio, hora_fin)
        if res3.empty:
            st.write("No se encontraron patrones significativos")
        else:
            st.dataframe(res3)

    st.markdown("### Confluencia Vela-Volumen:")
    cvv = run_volumen_vela(hora_inicio, hora_fin)
    st.info(cvv)

    st.markdown("### Reversión Heurística:")
    rev = run_reversion_heuristica(hora_inicio, hora_fin)
    st.warning(rev)

    # Consolidar señales para obtener dirección final
    prob_up = 0
    prob_down = 0
    
    # Anomalías
    if not res1.empty:
        prob_up += (res1['Direccion'] == 'ALZA').sum()
        prob_down += (res1['Direccion'] == 'BAJA').sum()
    
    # Confluencia Vela-Volumen
    if "Alcista" in cvv:
        prob_up += 1
    elif "Bajista" in cvv:
        prob_down += 1
    
    # Order Flow
    if not res3.empty:
        prob_up += (res3['Clasificacion'] == 'Intencion_Alcista').sum()
        prob_down += (res3['Clasificacion'] == 'Intencion_Bajista').sum()
    
    # Reversión Heurística
    if "ARRIBA" in rev:
        prob_up += 2
    else:
        prob_down += 2

    direccion_final = 'ALZA' if prob_up >= prob_down else 'BAJA'
    st.success(f"📌 Dirección más probable: {'📈 ALZA' if direccion_final == 'ALZA' else '📉 BAJA'}")
    st.caption("Basado en suma ponderada de señales de los 4 módulos")

    df_datos = obtener_datos()
    precio_actual = df_datos['Close'].iloc[-1] if not df_datos.empty else np.nan
    ts_actual = datetime.now(ty_ny)

    if "senal_consolidada" not in st.session_state:
        st.session_state["senal_consolidada"] = pd.DataFrame(columns=['timestamp','direccion','precio_entrada'])

    # Guardar solo si la última señal es diferente para evitar duplicados
    if st.session_state["senal_consolidada"].empty or (
        st.session_state["senal_consolidada"].iloc[-1]['timestamp'] != ts_actual
        or st.session_state["senal_consolidada"].iloc[-1]['direccion'] != direccion_final
    ):
        nueva_senal = pd.DataFrame([{
            'timestamp': ts_actual,
            'direccion': direccion_final,
            'precio_entrada': precio_actual
        }])
        st.session_state["senal_consolidada"] = pd.concat([st.session_state["senal_consolidada"], nueva_senal], ignore_index=True)

    if st.button("📌 Evaluar Última Señal Consolidada"):
        evaluar_senal_ultima()

    df_evaluadas = cargar_resultados_csv()
    resumen_efectividad(df_evaluadas)
