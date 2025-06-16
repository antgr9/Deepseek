import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time, timedelta
import yfinance as yf
import os
import logging
from logging.handlers import RotatingFileHandler
import plotly.graph_objects as go
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestClassifier
import requests
import joblib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

# ----------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------
st.set_page_config(
    page_title="Quantum Scalping Pro", 
    layout="wide",
    page_icon="🚀"
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('quantum_trading.log', maxBytes=5*1024*1024, backupCount=3)
    ]
)
logger = logging.getLogger()

# Zona horaria NY
ty_ny = pytz.timezone("America/New_York")

# ----------------------------
# PARÁMETROS AVANZADOS
# ----------------------------
class QuantumParams:
    def __init__(self):
        self.STOP_LOSS = 4
        self.TAKE_PROFIT = 15
        self.EVAL_VELAS = 5
        self.Z_SCORE_THRESH = 2.0
        self.VOL_RATIO_THRESH = 1.5
        self.MIN_VOLATILITY = 0.001
        self.ORDER_BOOK_IMB_THRESH = 0.25
        self.MACRO_SENTIMENT_THRESH = 0.1
        self.VIX_HIGH = 25
        self.VIX_LOW = 15
        self.ML_CONFIDENCE_THRESH = 0.7

qp = QuantumParams()

# ----------------------------
# DATOS INSTITUCIONALES
# ----------------------------
class InstitutionalData:
    def __init__(self):
        self.NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'tu_api_key_newsapi')
        self.POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', 'tu_api_key_polygon')
        self.model_rf = self._load_ml_model()
    
    def _load_ml_model(self):
        try:
            if os.path.exists('quantum_model.pkl'):
                return joblib.load('quantum_model.pkl')
            return RandomForestClassifier(n_estimators=100)
        except Exception as e:
            logger.error(f"Error loading ML model: {str(e)}")
            return None
    
    @st.cache_data(ttl=30, show_spinner="Obteniendo Order Book...")
    def get_order_book_imbalance(_self, ticker="ES=F"):
        try:
            if _self.POLYGON_API_KEY.startswith('tu_api_key'):
                return 0.0  # Modo demo
                
            url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}/book?apiKey={_self.POLYGON_API_KEY}"
            response = requests.get(url)
            data = response.json()
            bid_vol = sum([level['size'] for level in data['bids'][:5]])
            ask_vol = sum([level['size'] for level in data['asks'][:5]])
            return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)
        except Exception as e:
            logger.error(f"Error Order Book: {str(e)}")
            return 0.0

    @st.cache_data(ttl=300, show_spinner="Analizando sentimiento macro...")
    def get_macro_sentiment(_self):
        try:
            news = NewsApiClient(api_key=_self.NEWS_API_KEY)
            articles = news.get_everything(
                q="FED OR ECB OR CPI OR GDP OR Employment",
                language='en',
                sort_by='relevancy',
                page_size=10
            )
            titles = [a['title'] for a in articles['articles']]
            positive_words = ['bull', 'up', 'strong', 'growth', 'positive']
            negative_words = ['bear', 'down', 'weak', 'recession', 'negative']
            
            positive = sum(1 for t in titles if any(w in t.lower() for w in positive_words))
            negative = sum(1 for t in titles if any(w in t.lower() for w in negative_words))
            
            return (positive - negative) / len(titles) if titles else 0.0
        except Exception as e:
            logger.error(f"Error Macro Sentiment: {str(e)}")
            return 0.0

inst_data = InstitutionalData()

# ----------------------------
# FUNCIONES DE ANÁLISIS TÉCNICO
# ----------------------------
def calcular_volatilidad(df: pd.DataFrame, window: int = 20) -> float:
    """Calcula la volatilidad histórica usando desviación estándar de retornos logarítmicos"""
    returns = np.log(df['Close'] / df['Close'].shift(1))
    return returns.rolling(window=window).std().iloc[-1]

def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> tuple:
    """Calcula el perfil de volumen y el POC (Point of Control)"""
    try:
        prices = np.linspace(df['Low'].min(), df['High'].max(), bins)
        df['price_bin'] = pd.cut(df['Close'], bins=prices, include_lowest=True)
        vol_profile = df.groupby('price_bin')['Volume'].sum()
        
        if vol_profile.empty:
            return df['Close'].iloc[-1], 0.0
        
        poc_bin = vol_profile.idxmax()
        return poc_bin.mid, vol_profile.max()
    except Exception as e:
        logger.error(f"Error Volume Profile: {str(e)}")
        return df['Close'].iloc[-1], 0.0

def plot_volume_profile(df: pd.DataFrame) -> go.Figure:
    """Genera gráfico interactivo del perfil de volumen"""
    try:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['Close'],
            histnorm='probability density',
            name='Perfil de Volumen',
            marker_color='#1f77b4',
            opacity=0.7,
            xbins=dict(
                size=(df['High'].max() - df['Low'].min())/20
            )
        ))
        fig.update_layout(
            title='Distribución de Volumen por Precio',
            xaxis_title='Precio',
            yaxis_title='Densidad de Volumen',
            bargap=0.01,
            height=400
        )
        return fig
    except Exception as e:
        logger.error(f"Error plotting volume profile: {str(e)}")
        return go.Figure()

def dynamic_risk_management() -> Dict[str, float]:
    """Ajusta parámetros de riesgo basado en condiciones de mercado"""
    try:
        vix = yf.Ticker("^VIX").history(period='1d')['Close'].iloc[-1]
        imb = inst_data.get_order_book_imbalance()
        
        risk_params = {
            'stop_loss': qp.STOP_LOSS,
            'take_profit': qp.TAKE_PROFIT,
            'size_multiplier': 1.0,
            'filter_strength': 1.0
        }
        
        # Ajuste por volatilidad (VIX)
        if vix > qp.VIX_HIGH:
            risk_params.update({
                'stop_loss': qp.STOP_LOSS * 1.3,
                'take_profit': qp.TAKE_PROFIT * 1.2,
                'size_multiplier': 0.7,
                'filter_strength': 1.2
            })
        elif vix < qp.VIX_LOW:
            risk_params.update({
                'stop_loss': qp.STOP_LOSS * 0.8,
                'take_profit': qp.TAKE_PROFIT * 0.9,
                'size_multiplier': 1.2,
                'filter_strength': 0.8
            })
        
        # Ajuste por flujo institucional
        if abs(imb) > qp.ORDER_BOOK_IMB_THRESH:
            risk_params['filter_strength'] *= 1.5 if imb > 0 else 0.7
        
        return risk_params
    except Exception as e:
        logger.error(f"Error Risk Management: {str(e)}")
        return {
            'stop_loss': qp.STOP_LOSS,
            'take_profit': qp.TAKE_PROFIT,
            'size_multiplier': 1.0,
            'filter_strength': 1.0
        }

# ----------------------------
# MÓDULOS ORIGINALES MEJORADOS
# ----------------------------
def run_anomalias_retornos(hora_inicio: time, hora_fin: time, z_thresh: float) -> pd.DataFrame:
    """Detección de anomalías en retornos con filtro de horario"""
    try:
        df = obtener_datos()
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'Close', 'z_score', 'Direccion'])
        
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        mean = df['log_ret'].rolling(window=20).mean()
        std = df['log_ret'].rolling(window=20).std()
        df['z_score'] = (df['log_ret'] - mean) / std
        df.dropna(inplace=True)
        
        # Filtrar por horario y umbral
        anomalias = df[
            (df.index.to_series().apply(lambda x: dentro_de_horario(x, hora_inicio, hora_fin)) &
            (abs(df['z_score']) > z_thresh)
        ].tail(10)
        
        if anomalias.empty:
            return pd.DataFrame(columns=['timestamp', 'Close', 'z_score', 'Direccion'])
        
        resultado = anomalias[['Close', 'z_score']].copy()
        resultado['timestamp'] = anomalias.index
        resultado['Direccion'] = np.where(resultado['z_score'] > 0, 'ALZA', 'BAJA')
        return resultado.reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"Error Anomalías Retornos: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'Close', 'z_score', 'Direccion'])

def run_volumen_vela(hora_inicio: time, hora_fin: time, vol_thresh: float) -> tuple:
    """Confluencia entre patrones de velas y volumen"""
    try:
        df = obtener_datos()
        if df.empty or not dentro_de_horario(df.index[-1], hora_inicio, hora_fin):
            return "Fuera de horario", None
        
        df['clasificacion'] = df.apply(
            lambda row: clasificar_vela(row['Open'], row['High'], row['Low'], row['Close']), 
            axis=1
        )
        ult = df.iloc[-1]
        
        if ult['clasificacion'] in ['Intencion_Alcista', 'Marubozu_Alcista'] and ult['vol_ratio'] > vol_thresh:
            return f"Confluencia Alcista: {ult['clasificacion']} (Vol: {ult['vol_ratio']:.2f}x)", ult
        elif ult['clasificacion'] in ['Intencion_Bajista', 'Marubozu_Bajista'] and ult['vol_ratio'] > vol_thresh:
            return f"Confluencia Bajista: {ult['clasificacion']} (Vol: {ult['vol_ratio']:.2f}x)", ult
        else:
            return "Sin confluencia clara", None
    except Exception as e:
        logger.error(f"Error Volumen Vela: {str(e)}")
        return "Error en análisis", None

def run_orderflow_sintetico(hora_inicio: time, hora_fin: time) -> pd.DataFrame:
    """Order Flow Sintético basado en patrones de velas"""
    try:
        df = obtener_datos()
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'Close', 'Clasificacion'])
        
        df['Clasificacion'] = df.apply(
            lambda row: clasificar_vela(row['Open'], row['High'], row['Low'], row['Close']), 
            axis=1
        )
        
        recientes = df[
            (df['Clasificacion'] != 'Neutral') &
            (df.index.to_series().apply(lambda x: dentro_de_horario(x, hora_inicio, hora_fin)))
        ].tail(10)
        
        if recientes.empty:
            return pd.DataFrame(columns=['timestamp', 'Close', 'Clasificacion'])
        
        return recientes[['Close', 'Clasificacion']].assign(
            timestamp=recientes.index
        ).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error Order Flow: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'Close', 'Clasificacion'])

def run_reversion_heuristica(hora_inicio: time, hora_fin: time) -> tuple:
    """Modelo heurístico de reversión con RSI"""
    try:
        df = yf.Ticker("ES=F").history(period="5d", interval="5m")
        if df.empty:
            return "Sin datos", None
            
        if df.index.tz is None:
            df.index = df.index.tz_localize(pytz.utc).tz_convert(ty_ny)
        else:
            df.index = df.index.tz_convert(ty_ny)

        df = df[df.index.to_series().apply(lambda x: dentro_de_horario(x, hora_inicio, hora_fin))]
        if df.empty:
            return "No hay datos en horario configurado", None

        # Cálculos mejorados
        df['ret'] = np.log(df['Close'].pct_change() + 1)
        df['cum_ret_lb'] = df['ret'].rolling(window=6).sum()
        mean = df['cum_ret_lb'].rolling(120).mean()
        std = df['cum_ret_lb'].rolling(120).std().replace(0, np.nan)
        df['z_cum_ret'] = (df['cum_ret_lb'] - mean) / std
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        last = df.iloc[-1]
        prob = 20 + (20 if abs(last['z_cum_ret']) > 1.5 else 0) + (10 if abs(last['z_cum_ret']) > 2.0 else 0)
        prob += (10 if last['vol_ratio'] > 1.5 else 0) + (10 if last['rsi'] > 70 or last['rsi'] < 30 else 0)
        prob = min(prob, 95)
        
        direccion = "ABAJO" if last['cum_ret_lb'] > 0 else "ARRIBA"
        msg = (
            f"Prob. reversión {direccion}: {prob}% | "
            f"z={last['z_cum_ret']:.2f} | "
            f"vol={last['vol_ratio']:.2f}x | "
            f"RSI={last['rsi']:.1f}"
        )
        return msg, last
    except Exception as e:
        logger.error(f"Error Reversión Heurística: {str(e)}")
        return f"Error: {str(e)}", None

# ----------------------------
# FUNCIONES AUXILIARES
# ----------------------------
@st.cache_data(ttl=60, show_spinner="Obteniendo datos de mercado...")
def obtener_datos(ticker: str = "^GSPC", intervalo: str = "1m", periodo: str = "1d") -> pd.DataFrame:
    """Obtiene datos de yfinance con indicadores técnicos"""
    try:
        df = yf.Ticker(ticker).history(period=periodo, interval=intervalo)
        if df.empty:
            return pd.DataFrame()
            
        if df.index.tz is None:
            df.index = df.index.tz_localize(pytz.utc).tz_convert(ty_ny)
        else:
            df.index = df.index.tz_convert(ty_ny)
        
        # Calcular indicadores
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    except Exception as e:
        logger.error(f"Error obteniendo datos: {str(e)}")
        return pd.DataFrame()

def dentro_de_horario(dt: datetime, hora_inicio: time, hora_fin: time) -> bool:
    """Verifica si un datetime está dentro del horario de trading"""
    t = dt.time()
    return hora_inicio <= t <= hora_fin

def clasificar_vela(open_: float, high: float, low: float, close: float) -> str:
    """Clasificación avanzada de velas japonesas"""
    try:
        cuerpo = abs(close - open_)
        rango_total = high - low
        if rango_total == 0:
            return "Neutral"
            
        upper = high - max(open_, close)
        lower = min(open_, close) - low
        
        # Criterios mejorados
        es_doji = cuerpo / rango_total < 0.1
        es_marubozu = cuerpo / rango_total > 0.9
        sombra_superior_larga = upper / rango_total > 0.4
        sombra_inferior_larga = lower / rango_total > 0.4
        
        if es_doji and (sombra_superior_larga or sombra_inferior_larga):
            return "Doji_Reversión"
        elif es_marubozu and close > open_:
            return "Marubozu_Alcista"
        elif es_marubozu and close < open_:
            return "Marubozu_Bajista"
        elif cuerpo / rango_total < 0.3 and (sombra_superior_larga or sombra_inferior_larga):
            return "Absorción"
        elif close > open_ and close > (high - 0.2 * rango_total):
            return "Intencion_Alcista"
        elif close < open_ and close < (low + 0.2 * rango_total):
            return "Intencion_Bajista"
        else:
            return "Neutral"
    except:
        return "Neutral"

# ----------------------------
# INTERFAZ STREAMLIT
# ----------------------------
def main():
    # Inicialización de session_state
    if 'historico_señales' not in st.session_state:
        st.session_state.historico_señales = pd.DataFrame(
            columns=['timestamp', 'modulo', 'direccion', 'precio', 'confirmada', 'resultado']
        )
    
    if 'senal_consolidada' not in st.session_state:
        st.session_state.senal_consolidada = pd.DataFrame(
            columns=['timestamp', 'direccion', 'precio_entrada', 'confianza']
        )
    
    # Sidebar de configuración
    with st.sidebar:
        st.title("⚙️ Quantum Configuration")
        
        # Selector de instrumento
        ticker = st.selectbox("Instrumento", ["^GSPC", "ES=F", "NQ=F", "YM=F"], index=0)
        
        # Parámetros de trading
        qp.STOP_LOSS = st.number_input("Stop Loss (puntos)", value=qp.STOP_LOSS, min_value=1, max_value=20)
        qp.TAKE_PROFIT = st.number_input("Take Profit (puntos)", value=qp.TAKE_PROFIT, min_value=1, max_value=50)
        qp.Z_SCORE_THRESH = st.slider("Umbral Z-Score", 1.5, 3.0, qp.Z_SCORE_THRESH, 0.1)
        qp.VOL_RATIO_THRESH = st.slider("Umbral Volumen", 1.0, 3.0, qp.VOL_RATIO_THRESH, 0.1)
        
        # Horario de trading
        hora_inicio = st.time_input("Hora inicio (NY)", value=time(8, 30))
        hora_fin = st.time_input("Hora fin (NY)", value=time(16, 0))
        
        if st.button("🔄 Reiniciar Parámetros"):
            qp.__init__()
            st.rerun()
    
    # Contenido principal
    st.title("🚀 Quantum Scalping Pro Dashboard")
    
    # Obtener y procesar datos en paralelo
    with ThreadPoolExecutor() as executor:
        future_data = executor.submit(obtener_datos, ticker)
        future_imb = executor.submit(inst_data.get_order_book_imbalance)
        future_sentiment = executor.submit(inst_data.get_macro_sentiment)
        
        df = future_data.result()
        imb = future_imb.result()
        sentiment = future_sentiment.result()
    
    if df.empty:
        st.error("❌ No se pudieron obtener datos del mercado")
        return
    
    current_price = df['Close'].iloc[-1]
    current_time = datetime.now(ty_ny)
    
    # Panel de control institucional
    with st.expander("🌐 Institutional Dashboard", expanded=True):
        cols = st.columns(4)
        with cols[0]:
            st.metric("Precio Actual", f"{current_price:.2f}")
        with cols[1]:
            st.metric("Order Book Imbalance", f"{imb:.2%}", 
                     delta="Compra" if imb > 0 else "Venta")
        with cols[2]:
            poc, _ = calculate_volume_profile(df)
            st.metric("POC (Price of Control)", f"{poc:.2f}", 
                     delta=f"{(current_price - poc):.2f}")
        with cols[3]:
            st.metric("Sentimiento Macro", 
                     "🟢 Alcista" if sentiment > qp.MACRO_SENTIMENT_THRESH else 
                     "🔴 Bajista" if sentiment < -qp.MACRO_SENTIMENT_THRESH else "⚪ Neutral")
    
    # Visualizaciones
    col1, col2 = st.columns([7, 3])
    with col1:
        st.plotly_chart(plot_volume_profile(df), use_container_width=True)
    
    with col2:
        risk_params = dynamic_risk_management()
        st.metric("Stop Loss Ajustado", f"{risk_params['stop_loss']:.2f}")
        st.metric("Take Profit Ajustado", f"{risk_params['take_profit']:.2f}")
        st.metric("Multiplicador Tamaño", f"{risk_params['size_multiplier']:.2f}x")
        st.metric("Filtro de Señales", f"{risk_params['filter_strength']:.1f}x")
    
    # Ejecutar módulos de análisis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Consolidado", 
        "📈 Anomalías", 
        "🕯️ Vela-Volumen", 
        "🔍 Order Flow", 
        "🔄 Reversión"
    ])
    
    with tab1:  # Consolidado
        st.subheader("Señales Consolidadas")
        
        # Ejecutar todos los módulos
        with st.spinner("Analizando mercados..."):
            res1 = run_anomalias_retornos(hora_inicio, hora_fin, qp.Z_SCORE_THRESH)
            res3 = run_orderflow_sintetico(hora_inicio, hora_fin)
            cvv, cvv_data = run_volumen_vela(hora_inicio, hora_fin, qp.VOL_RATIO_THRESH)
            rev, rev_data = run_reversion_heuristica(hora_inicio, hora_fin)
        
        # Consolidar señales
        prob_up, prob_down = 0, 0
        
        # Anomalías (Peso 2x)
        if not res1.empty:
            prob_up += (res1['Direccion'] == 'ALZA').sum() * 2
            prob_down += (res1['Direccion'] == 'BAJA').sum() * 2
        
        # Confluencia Vela-Volumen (Peso 3x)
        if "Alcista" in cvv:
            prob_up += 3
        elif "Bajista" in cvv:
            prob_down += 3
        
        # Order Flow (Peso 1x)
        if not res3.empty:
            prob_up += (res3['Clasificacion'].str.contains('Alcista')).sum()
            prob_down += (res3['Clasificacion'].str.contains('Bajista')).sum()
        
        # Reversión (Peso 2x)
        if "ARRIBA" in rev:
            prob_up += 2
        else:
            prob_down += 2
        
        # Aplicar filtro institucional
        if imb > qp.ORDER_BOOK_IMB_THRESH:
            prob_up *= risk_params['filter_strength']
        elif imb < -qp.ORDER_BOOK_IMB_THRESH:
            prob_down *= risk_params['filter_strength']
        
        direccion_final = 'ALZA' if prob_up >= prob_down else 'BAJA'
        confianza = abs(prob_up - prob_down) / max(prob_up + prob_down, 1) * 100
        
        # Mostrar resultado consolidado
        st.success(f"""
        📌 **Dirección más probable:** {'📈 ALZA' if direccion_final == 'ALZA' else '📉 BAJA'}  
        🔍 **Confianza:** {confianza:.1f}%  
        ⚖️ **Balance:** Alcistas {prob_up} vs Bajistas {prob_down}
        """)
        
        # Gráfico de balance
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Bar(
            x=['Alcista', 'Bajista'],
            y=[prob_up, prob_down],
            marker_color=['green', 'red'],
            text=[prob_up, prob_down],
            textposition='auto'
        ))
        fig_balance.update_layout(
            title="Balance de Señales",
            yaxis_title="Puntuación",
            height=300
        )
        st.plotly_chart(fig_balance, use_container_width=True)
        
        # Guardar señal consolidada
        nueva_senal = pd.DataFrame([{
            'timestamp': current_time,
            'direccion': direccion_final,
            'precio_entrada': current_price,
            'confianza': confianza
        }])
        st.session_state.senal_consolidada = pd.concat([
            st.session_state.senal_consolidada, 
            nueva_senal
        ], ignore_index=True)
        
        # Botón de evaluación
        if st.button("📌 Evaluar Última Señal", use_container_width=True):
            evaluar_senal()
        
        # Historial de señales
        st.subheader("Historial de Señales")
        st.dataframe(
            st.session_state.senal_consolidada.tail(5),
            column_config={
                "timestamp": "Hora",
                "direccion": "Dirección",
                "precio_entrada": "Precio",
                "confianza": "Confianza %"
            }
        )
        
        # Exportar CSV
        if st.button("💾 Exportar a CSV", use_container_width=True):
            st.session_state.senal_consolidada.to_csv('señales_quantum.csv', index=False)
            st.success("Datos exportados correctamente")
    
    with tab2:  # Anomalías
        st.subheader("Anomalías de Rendimiento")
        if not res1.empty:
            st.dataframe(res1)
            fig_zscore = go.Figure()
            fig_zscore.add_trace(go.Scatter(
                x=res1['timestamp'],
                y=res1['z_score'],
                mode='markers',
                marker=dict(
                    color=np.where(res1['Direccion'] == 'ALZA', 'green', 'red'),
                    size=10
                ),
                name='Z-Score'
            ))
            fig_zscore.update_layout(
                title="Anomalías Recientes",
                yaxis_title="Z-Score",
                height=400
            )
            st.plotly_chart(fig_zscore, use_container_width=True)
        else:
            st.warning("No se encontraron anomalías significativas")
    
    with tab3:  # Vela-Volumen
        st.subheader("Confluencia Vela-Volumen")
        st.info(cvv)
        if cvv_data is not None:
            st.json({
                "Precio": cvv_data['Close'],
                "Volumen": int(cvv_data['Volume']),
                "Vol Ratio": f"{cvv_data['vol_ratio']:.2f}x",
                "Patrón": cvv_data['clasificacion']
            })
    
    with tab4:  # Order Flow
        st.subheader("Order Flow Sintético")
        if not res3.empty:
            st.dataframe(res3)
            
            # Distribución de patrones
            dist_patrones = res3['Clasificacion'].value_counts()
            fig_patrones = go.Figure(go.Pie(
                labels=dist_patrones.index,
                values=dist_patrones.values,
                hole=0.3,
                marker_colors=['green', 'red', 'blue', 'orange']
            ))
            fig_patrones.update_layout(title="Distribución de Patrones")
            st.plotly_chart(fig_patrones, use_container_width=True)
        else:
            st.warning("No se encontraron patrones significativos")
    
    with tab5:  # Reversión
        st.subheader("Reversión Heurística")
        st.warning(rev)
        if rev_data is not None:
            cols = st.columns(4)
            cols[0].metric("Z-Score", f"{rev_data['z_cum_ret']:.2f}")
            cols[1].metric("RSI", f"{rev_data['rsi']:.1f}")
            cols[2].metric("Vol Ratio", f"{rev_data['vol_ratio']:.2f}x")
            cols[3].metric("Retorno Acum.", f"{rev_data['cum_ret_lb']:.4f}")

# ----------------------------
# FUNCIONES DE EVALUACIÓN
# ----------------------------
def evaluar_senal():
    """Evalúa la última señal consolidada"""
    try:
        if st.session_state.senal_consolidada.empty:
            st.warning("No hay señales para evaluar")
            return
        
        ultima = st.session_state.senal_consolidada.iloc[-1]
        ts = pd.to_datetime(ultima['timestamp'])
        if ts.tzinfo is None:
            ts = ts.tz_localize(ty_ny)
        
        # Obtener datos para evaluación
        df_eval = obtener_datos(periodo="1h")
        if df_eval.empty:
            st.warning("No hay datos para evaluación")
            return
        
        df_eval = df_eval[df_eval.index > ts]
        if df_eval.empty:
            st.warning("No hay datos posteriores a la señal")
            return
        
        # Evaluar resultado
        resultado = "NEUTRO"
        puntos = 0
        precio_entrada = ultima['precio_entrada']
        direccion = ultima['direccion']
        
        for idx, row in df_eval.iterrows():
            diff = row['Close'] - precio_entrada
            if direccion == 'ALZA':
                if diff >= qp.TAKE_PROFIT:
                    resultado = "TP"
                    puntos = qp.TAKE_PROFIT
                    break
                elif diff <= -qp.STOP_LOSS:
                    resultado = "SL"
                    puntos = -qp.STOP_LOSS
                    break
            elif direccion == 'BAJA':
                if (precio_entrada - row['Close']) >= qp.TAKE_PROFIT:
                    resultado = "TP"
                    puntos = qp.TAKE_PROFIT
                    break
                elif (precio_entrada - row['Close']) <= -qp.STOP_LOSS:
                    resultado = "SL"
                    puntos = -qp.STOP_LOSS
                    break
        
        # Actualizar historial
        st.session_state.senal_consolidada.at[
            st.session_state.senal_consolidada.index[-1], 'resultado'] = resultado
        
        # Mostrar resultado
        if resultado == "TP":
            st.success(f"✅ Señal Exitosa! {resultado} | +{puntos} puntos")
        elif resultado == "SL":
            st.error(f"❌ Señal Fallida! {resultado} | {puntos} puntos")
        else:
            st.warning(f"⚠️ Señal Neutra. Sin activación de TP/SL")
            
    except Exception as e:
        logger.error(f"Error evaluando señal: {str(e)}")
        st.error(f"Error al evaluar señal: {str(e)}")

if __name__ == "__main__":
    main()
