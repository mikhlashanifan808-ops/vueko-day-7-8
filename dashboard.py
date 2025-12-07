# dashboard_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import pickle
import os
import time
import threading
import paho.mqtt.client as mqtt  # TAMBAHAN: import MQTT

# ==================== KONFIGURASI ====================
BASE_DIR = r"C:\Hanif\momodel"
MODELS_DIR = os.path.join(BASE_DIR, "Trainingdht", "models_v2")
CSV_PATH = os.path.join(BASE_DIR, "Trainingdht", "sensor_data.csv")  # TAMBAHAN: path CSV

# ==================== KONFIGURASI MQTT ====================
MQTT_BROKER = "2a7336927a8a4b7292000aa2485e93dd.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764994920849"
MQTT_PASSWORD = "B0dzc2W*i#&NAe?3JqS8"
DHT_TOPIC = "sic/dibimbing/kelompok-Aktuator/Vueko/pub/dht"
ML_TOPIC = "sic/dibimbing/kelompok-Aktuator/Vueko/pub/ml_prediction"

# ==================== VARIABEL GLOBAL MQTT ====================
mqtt_client = None
latest_sensor_data = {
    "temperature": 24.0,
    "humidity": 65.0,
    "timestamp": datetime.now(),
    "label": "NORMAL",
    "source": "manual"
}
mqtt_connected = False
last_mqtt_update = None

# ==================== SETUP PAGE ====================
st.set_page_config(
    page_title="DHT ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FUNGSI MQTT ====================
def on_mqtt_connect(client, userdata, flags, rc):
    """Callback ketika terhubung ke MQTT"""
    global mqtt_connected, last_mqtt_update
    if rc == 0:
        mqtt_connected = True
        last_mqtt_update = datetime.now()
        client.subscribe(DHT_TOPIC)
        client.subscribe(ML_TOPIC)
        print(f"✅ MQTT Connected! Subscribed to {DHT_TOPIC} and {ML_TOPIC}")
    else:
        print(f"❌ MQTT Connection failed with code: {rc}")

def on_mqtt_message(client, userdata, msg):
    """Callback ketika menerima pesan MQTT"""
    global latest_sensor_data, last_mqtt_update
    try:
        data = json.loads(msg.payload.decode())
        last_mqtt_update = datetime.now()
        
        if msg.topic == DHT_TOPIC:
            # Data sensor dari Wokwi
            latest_sensor_data.update({
                "temperature": float(data.get("temperature", 24.0)),
                "humidity": float(data.get("humidity", 65.0)),
                "timestamp": datetime.now(),
                "source": "wokwi",
                "label": "REAL-TIME"
            })
            print(f"📡 Received sensor data: {latest_sensor_data['temperature']}°C, {latest_sensor_data['humidity']}%")
        
        elif msg.topic == ML_TOPIC:
            # Prediksi ML dari trainingmodel.py
            latest_sensor_data.update({
                "ml_prediction": data.get("label", "UNKNOWN"),
                "ml_model": data.get("model", "Unknown Model"),
                "ml_confidence": data.get("confidence", 0.0),
                "ml_timestamp": data.get("publish_time", "")
            })
            print(f"🤖 Received ML prediction: {data.get('label')} from {data.get('model')}")
            
    except Exception as e:
        print(f"❌ Error processing MQTT message: {e}")

def connect_mqtt():
    """Connect to MQTT broker"""
    global mqtt_client
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        mqtt_client.tls_set()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message
        
        # Connect in a separate thread
        def connect_thread():
            try:
                mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
                mqtt_client.loop_start()
            except Exception as e:
                print(f"❌ MQTT connection error: {e}")
        
        thread = threading.Thread(target=connect_thread)
        thread.daemon = True
        thread.start()
        return True
    except Exception as e:
        st.error(f"MQTT Connection Error: {e}")
        return False

def disconnect_mqtt():
    """Disconnect from MQTT broker"""
    global mqtt_client, mqtt_connected
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        mqtt_connected = False
        print("📡 MQTT Disconnected")

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    """Load ML models dengan caching"""
    models = {}
    try:
        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load models
        model_files = {
            'Decision Tree': 'decision_tree.pkl',
            'KNN': 'knn.pkl',
            'Logistic Regression': 'logistic_regression.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
        
        return models, scaler
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# ==================== FUNGSI PREDIKSI ====================
def predict_temperature(models, scaler, temperature, humidity, hour=None, minute=None):
    """Prediksi dengan semua model"""
    if hour is None or minute is None:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
    
    features = np.array([[temperature, humidity, hour, minute]])
    features_scaled = scaler.transform(features)
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            # Predict
            pred_code = model.predict(features_scaled)[0]
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features_scaled)[0]
                confidence = probs[pred_code]
            else:
                confidence = 1.0
                probs = [0, 0, 0]
            
            # Map to label
            label_map = {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
            label = label_map.get(pred_code, 'UNKNOWN')
            
            predictions[model_name] = {
                'label': label,
                'confidence': float(confidence),
                'probabilities': {
                    'DINGIN': float(probs[0]) if len(probs) > 0 else 0,
                    'NORMAL': float(probs[1]) if len(probs) > 1 else 0,
                    'PANAS': float(probs[2]) if len(probs) > 2 else 0
                },
                'label_encoded': int(pred_code),
                'color': get_label_color(label)
            }
            
        except Exception as e:
            predictions[model_name] = {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    return predictions

def get_label_color(label):
    """Get color based on label"""
    colors = {
        'DINGIN': '#3498db',    # Blue
        'NORMAL': '#2ecc71',    # Green
        'PANAS': '#e74c3c',     # Red
        'UNKNOWN': '#95a5a6',   # Gray
        'ERROR': '#f39c12'      # Orange
    }
    return colors.get(label, '#95a5a6')

# ==================== LOAD HISTORICAL DATA ====================
def load_historical_data():
    """Load historical data dari CSV"""
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH, delimiter=';')
            
            # Parse timestamp
            if 'timestamp' in df.columns:
                df[['hour', 'minute', 'second']] = df['timestamp'].str.split(';', expand=True).astype(int)
                # Create datetime column for plotting
                base_date = datetime.now().date()
                df['datetime'] = df.apply(
                    lambda row: datetime.combine(base_date, datetime.min.time().replace(
                        hour=row['hour'], minute=row['minute'], second=row['second']
                    )), axis=1
                )
            
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return pd.DataFrame()

# ==================== SIDEBAR ====================
def sidebar_controls():
    """Sidebar controls"""
    st.sidebar.title("⚙️ Kontrol Dashboard")
    
    # ===== TAMBAHAN: KONEKSI REAL-TIME =====
    st.sidebar.subheader("🔗 Real-time Connection")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📡 Connect to Wokwi", key="connect_btn"):
            if connect_mqtt():
                st.sidebar.success("Connecting to Wokwi...")
                time.sleep(1)
                st.rerun()
    
    with col2:
        if st.button("🔌 Disconnect", key="disconnect_btn"):
            disconnect_mqtt()
            st.sidebar.info("Disconnected from Wokwi")
            st.rerun()
    
    # Status koneksi
    if mqtt_connected:
        st.sidebar.success("✅ Connected to Wokwi")
        if last_mqtt_update:
            seconds_ago = (datetime.now() - last_mqtt_update).seconds
            st.sidebar.caption(f"Last update: {seconds_ago} seconds ago")
        
        # Tampilkan data real-time
        st.sidebar.markdown("---")
        st.sidebar.subheader("🌡️ Live Sensor Data")
        st.sidebar.metric("Temperature", f"{latest_sensor_data['temperature']}°C", "Live")
        st.sidebar.metric("Humidity", f"{latest_sensor_data['humidity']}%", "Live")
        
        if 'ml_prediction' in latest_sensor_data:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🤖 Latest ML Prediction")
            st.sidebar.write(f"**Model:** {latest_sensor_data.get('ml_model', 'N/A')}")
            st.sidebar.write(f"**Prediction:** {latest_sensor_data.get('ml_prediction', 'N/A')}")
            st.sidebar.write(f"**Confidence:** {latest_sensor_data.get('ml_confidence', 0):.1%}")
    else:
        st.sidebar.warning("⚠️ Not connected to Wokwi")
        st.sidebar.info("Use manual controls below")
    
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    show_dt = st.sidebar.checkbox("Decision Tree", value=True)
    show_knn = st.sidebar.checkbox("K-Nearest Neighbors", value=True)
    show_lr = st.sidebar.checkbox("Logistic Regression", value=True)
    
    # Manual prediction (hanya jika tidak ada koneksi real-time)
    st.sidebar.subheader("🔮 Manual Prediction")
    
    # Gunakan data real-time jika tersedia, otherwise gunakan default
    default_temp = latest_sensor_data['temperature'] if mqtt_connected else 24.0
    default_hum = latest_sensor_data['humidity'] if mqtt_connected else 65.0
    
    manual_temp = st.sidebar.slider("Temperature (°C)", 15.0, 35.0, float(default_temp), 0.5)
    manual_hum = st.sidebar.slider("Humidity (%)", 30.0, 90.0, float(default_hum), 1.0)
    manual_hour = st.sidebar.slider("Hour", 0, 23, datetime.now().hour)
    manual_minute = st.sidebar.slider("Minute", 0, 59, datetime.now().minute)
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Use Real-time Data", "Use Manual Input"],
        index=0 if mqtt_connected else 1
    )
    
    # Time range
    st.sidebar.subheader("📅 Time Range")
    days_back = st.sidebar.slider("Days to display", 1, 30, 7)
    
    # Historical data info
    hist_data = load_historical_data()
    if not hist_data.empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Historical Data")
        st.sidebar.write(f"Total records: {len(hist_data)}")
        if 'timestamp' in hist_data.columns and len(hist_data) > 0:
            last_record = hist_data.iloc[-1]['timestamp'] if 'timestamp' in hist_data.columns else "N/A"
            st.sidebar.write(f"Last record: {last_record}")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Dashboard"):
        st.rerun()
    
    return {
        'models': {'DT': show_dt, 'KNN': show_knn, 'LR': show_lr},
        'manual_input': (manual_temp, manual_hum, manual_hour, manual_minute),
        'data_source': data_source,
        'days_back': days_back,
        'historical_data': hist_data
    }

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.title("🤖 DHT Machine Learning Dashboard")
    
    # Status indicator
    if mqtt_connected:
        st.success("✅ **REAL-TIME MODE**: Connected to Wokwi ESP32")
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.metric("Live Temp", f"{latest_sensor_data['temperature']}°C", delta="Live")
        with status_col2:
            st.metric("Live Humidity", f"{latest_sensor_data['humidity']}%", delta="Live")
        with status_col3:
            if last_mqtt_update:
                seconds_ago = (datetime.now() - last_mqtt_update).seconds
                st.metric("Last Update", f"{seconds_ago}s ago")
    else:
        st.info("📝 **MANUAL MODE**: Using manual input controls")
    
    st.markdown("Real-time temperature prediction with 3 ML models")
    st.markdown("---")
    
    # Load models
    models, scaler = load_models()
    
    if models is None or scaler is None:
        st.error("❌ Failed to load ML models. Please run training first.")
        return
    
    # Sidebar controls
    controls = sidebar_controls()
    show_models = controls['models']
    manual_input = controls['manual_input']
    data_source = controls['data_source']
    historical_data = controls['historical_data']
    
    # Determine which data to use
    if data_source == "Use Real-time Data" and mqtt_connected:
        current_temp = latest_sensor_data['temperature']
        current_hum = latest_sensor_data['humidity']
        data_status = "Live from Wokwi"
    else:
        current_temp = manual_input[0]
        current_hum = manual_input[1]
        data_status = "Manual Input"
    
    # Row 1: Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌡️ Temperature", f"{current_temp}°C", data_status)
    
    with col2:
        st.metric("💧 Humidity", f"{current_hum}%", data_status)
    
    with col3:
        now = datetime.now()
        st.metric("⏰ Current Time", now.strftime("%H:%M"))
    
    with col4:
        if mqtt_connected:
            st.metric("🔗 Connection", "Online", "Wokwi")
        else:
            st.metric("🔗 Connection", "Offline", "Manual")
    
    st.markdown("---")
    
    # Row 2: Prediction Results
    st.subheader(f"🔮 Prediction Results ({data_status})")
    
    # Make prediction
    predictions = predict_temperature(
        models, scaler, 
        current_temp, current_hum,
        manual_input[2], manual_input[3]
    )
    
    # Display predictions in columns
    pred_cols = st.columns(3)
    
    for idx, (model_name, pred) in enumerate(predictions.items()):
        with pred_cols[idx]:
            color = pred.get('color', '#95a5a6')
            
            # Card-like display
            st.markdown(f"""
            <div style="
                background-color: {color}20;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid {color};
                margin-bottom: 20px;
            ">
                <h3 style="color: {color}; margin-top: 0;">{model_name}</h3>
                <h1 style="color: {color}; font-size: 2.5em; margin: 10px 0;">
                    {pred['label']}
                </h1>
                <p style="font-size: 1.2em; margin: 5px 0;">
                    Confidence: <strong>{pred['confidence']:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probabilities bar chart
            if 'probabilities' in pred:
                prob_df = pd.DataFrame({
                    'Class': list(pred['probabilities'].keys()),
                    'Probability': list(pred['probabilities'].values())
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    color='Class',
                    color_discrete_map={
                        'DINGIN': '#3498db',
                        'NORMAL': '#2ecc71',
                        'PANAS': '#e74c3c'
                    },
                    range_y=[0, 1]
                )
                fig.update_layout(
                    showlegend=False,
                    height=200,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 3: Model Comparison
    st.subheader("📊 Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison (sample data)
        accuracy_data = pd.DataFrame({
            'Model': ['Decision Tree', 'KNN', 'Logistic Regression'],
            'Accuracy': [0.92, 0.88, 0.85],
            'F1-Score': [0.91, 0.87, 0.84]
        })
        
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=accuracy_data['Model'], y=accuracy_data['Accuracy']),
            go.Bar(name='F1-Score', x=accuracy_data['Model'], y=accuracy_data['F1-Score'])
        ])
        
        fig.update_layout(
            title="Model Performance",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion matrix summary
        st.markdown("**Model Agreement**")
        
        # Check if all models agree
        labels = [pred['label'] for pred in predictions.values()]
        agreement = len(set(labels)) == 1
        
        if agreement:
            st.success(f"✅ All models agree: **{labels[0]}**")
            st.balloons()
        else:
            st.warning(f"⚠️ Models disagree: {', '.join(set(labels))}")
        
        # Display agreement matrix
        agreement_matrix = pd.DataFrame(index=models.keys(), columns=models.keys())
        
        for m1 in models.keys():
            for m2 in models.keys():
                if m1 == m2:
                    agreement_matrix.loc[m1, m2] = "✓"
                else:
                    agree = predictions[m1]['label'] == predictions[m2]['label']
                    agreement_matrix.loc[m1, m2] = "✓" if agree else "✗"
        
        st.dataframe(agreement_matrix, use_container_width=True)
    
    st.markdown("---")
    
    # Row 4: Historical Data & Trends
    st.subheader("📈 Historical Data & Trends")
    
    # Use real historical data if available, otherwise generate sample
    if not historical_data.empty and len(historical_data) > 0:
        display_data = historical_data.copy()
        data_source_info = "From Sensor Data"
    else:
        display_data = generate_sample_data()
        data_source_info = "Sample Data"
    
    st.caption(f"Data Source: {data_source_info} | Records: {len(display_data)}")
    
    # Filter by date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=controls['days_back'])
    
    if 'datetime' in display_data.columns:
        filtered_data = display_data[
            (display_data['datetime'] >= start_date) &
            (display_data['datetime'] <= end_date)
        ]
    else:
        filtered_data = display_data
    
    tab1, tab2, tab3 = st.tabs(["Temperature Trend", "Humidity Trend", "Label Distribution"])
    
    with tab1:
        if not filtered_data.empty and 'temperature' in filtered_data.columns:
            if 'datetime' in filtered_data.columns:
                x_col = 'datetime'
                title = "Temperature Trend Over Time"
            else:
                x_col = 'timestamp'
                title = "Temperature Readings"
            
            fig = px.line(
                filtered_data, 
                x=x_col, 
                y='temperature',
                color='label' if 'label' in filtered_data.columns else None,
                color_discrete_map={
                    'DINGIN': '#3498db',
                    'NORMAL': '#2ecc71',
                    'PANAS': '#e74c3c'
                } if 'label' in filtered_data.columns else None,
                title=title
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No temperature data available")
    
    with tab2:
        if not filtered_data.empty and 'humidity' in filtered_data.columns:
            fig = px.scatter(
                filtered_data,
                x='temperature' if 'temperature' in filtered_data.columns else 'timestamp',
                y='humidity',
                color='label' if 'label' in filtered_data.columns else None,
                color_discrete_map={
                    'DINGIN': '#3498db',
                    'NORMAL': '#2ecc71',
                    'PANAS': '#e74c3c'
                } if 'label' in filtered_data.columns else None,
                title="Temperature vs Humidity",
                hover_data=['timestamp'] if 'timestamp' in filtered_data.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No humidity data available")
    
    with tab3:
        if not filtered_data.empty and 'label' in filtered_data.columns:
            label_counts = filtered_data['label'].value_counts()
            fig = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                color=label_counts.index,
                color_discrete_map={
                    'DINGIN': '#3498db',
                    'NORMAL': '#2ecc71',
                    'PANAS': '#e74c3c'
                },
                title="Label Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No label data available")
    
    st.markdown("---")
    
    # Row 5: Model Details & Raw Data
    st.subheader("🔧 System Information")
    
    tab_info, tab_raw, tab_ml = st.tabs(["Model Details", "Raw Data", "ML Predictions"])
    
    with tab_info:
        with st.expander("View Model Information"):
            for model_name, model in models.items():
                st.write(f"**{model_name}**")
                st.write(f"Type: {type(model).__name__}")
                
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    st.write(f"Parameters: {len(params)}")
                    
                    with st.expander("View Parameters"):
                        for key, value in list(params.items())[:10]:  # Show first 10
                            st.write(f"- {key}: {value}")
                
                st.write("---")
        
        st.markdown("""
        ### 🎯 System Specifications
        - **ML Models**: Decision Tree, K-Nearest Neighbors, Logistic Regression
        - **Input Features**: Temperature, Humidity, Hour, Minute
        - **Output Labels**: DINGIN (<22°C), NORMAL (22-25°C), PANAS (>25°C)
        - **Data Sources**: Real-time Wokwi ESP32 + Historical CSV
        - **Update Frequency**: Real-time when connected
        """)
    
    with tab_raw:
        # Raw prediction data
        pred_data = []
        for model_name, pred in predictions.items():
            row = {
                'Model': model_name,
                'Prediction': pred['label'],
                'Confidence': f"{pred['confidence']:.1%}",
                'DINGIN Prob': f"{pred.get('probabilities', {}).get('DINGIN', 0):.1%}",
                'NORMAL Prob': f"{pred.get('probabilities', {}).get('NORMAL', 0):.1%}",
                'PANAS Prob': f"{pred.get('probabilities', {}).get('PANAS', 0):.1%}"
            }
            pred_data.append(row)
        
        st.dataframe(pd.DataFrame(pred_data), use_container_width=True)
        
        # Raw sensor data
        if not historical_data.empty:
            st.subheader("Historical Sensor Data")
            st.dataframe(historical_data, use_container_width=True)
    
    with tab_ml:
        st.subheader("Latest ML Predictions")
        
        if 'ml_prediction' in latest_sensor_data:
            col_ml1, col_ml2, col_ml3 = st.columns(3)
            with col_ml1:
                st.metric("Model", latest_sensor_data.get('ml_model', 'N/A'))
            with col_ml2:
                pred_label = latest_sensor_data.get('ml_prediction', 'N/A')
                color = get_label_color(pred_label)
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 10px; border-radius: 5px; border-left: 4px solid {color};">
                    <h3 style="margin: 0; color: {color};">Prediction: {pred_label}</h3>
                </div>
                """, unsafe_allow_html=True)
            with col_ml3:
                st.metric("Confidence", f"{latest_sensor_data.get('ml_confidence', 0):.1%}")
            
            if 'ml_timestamp' in latest_sensor_data:
                st.caption(f"Received at: {latest_sensor_data['ml_timestamp']}")
        else:
            st.info("No ML predictions received yet. Run trainingmodel.py to send predictions.")
    
    # Auto-refresh if connected
    if mqtt_connected:
        st.markdown("---")
        st.caption("🔄 Auto-refresh enabled for real-time data")
        time.sleep(2)  # Refresh every 2 seconds when connected
        st.rerun()
    else:
        if st.button("🔄 Refresh Predictions"):
            st.rerun()

def generate_sample_data():
    """Generate sample data untuk demo"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='H')
    
    data = []
    for date in dates:
        temp = np.random.normal(24, 5)  # Mean 24, std 5
        hum = np.random.normal(60, 15)  # Mean 60, std 15
        
        # Simple logic untuk label
        if temp < 22:
            label = 'DINGIN'
        elif temp > 26:
            label = 'PANAS'
        else:
            label = 'NORMAL'
        
        data.append({
            'timestamp': date,
            'temperature': round(temp, 1),
            'humidity': round(hum, 1),
            'label': label,
            'model': np.random.choice(['Decision Tree', 'KNN', 'Logistic Regression'])
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main()