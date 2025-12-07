import paho.mqtt.client as mqtt
import json
import csv
import time
from datetime import datetime
import os
import sys

# ==================== KONFIGURASI ====================
MQTT_BROKER = "2a7336927a8a4b7292000aa2485e93dd.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764994920849"
MQTT_PASSWORD = "B0dzc2W*i#&NAe?3JqS8"
DHT_TOPIC = "sic/dibimbing/kelompok-Aktuator/Vueko/pub/dht"

# File Configuration
FOLDER_PATH = "Trainingdht"
CSV_FILENAME = "sensor_data.csv"
CSV_PATH = os.path.join(FOLDER_PATH, CSV_FILENAME)

# ==================== SETUP CSV ====================
def setup_csv():
    """Buat file CSV dengan header"""
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)
        print(f"📁 Created folder: {FOLDER_PATH}")
    
    # Cek apakah file sudah ada
    file_exists = os.path.exists(CSV_PATH)
    
    if not file_exists:
        # Buat file baru dengan header
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                'timestamp',        # HH;MM;SS
                'temperature',      # °C
                'humidity',         # %
                'label',            # DINGIN/NORMAL/PANAS
                'label_encoded'     # 0/1/2
            ])
        print(f"✅ Created new CSV file: {CSV_PATH}")
    else:
        print(f"✅ CSV file already exists: {CSV_PATH}")
        
        # Hitung jumlah data yang sudah ada
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            existing_count = len(lines) - 1  # minus header
            print(f"📊 Existing records: {existing_count}")
            
            # Tampilkan 3 data terakhir
            if existing_count > 0:
                print("📄 Last 3 records:")
                for line in lines[max(1, len(lines)-3):]:  # Skip header
                    print(f"   {line.strip()}")

# ==================== MQTT CALLBACKS ====================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to HiveMQ Cloud!")
        print(f"📡 Subscribing to topic: {DHT_TOPIC}")
        client.subscribe(DHT_TOPIC)
        print("⏳ Waiting for DHT data...")
        print("=" * 50)
    else:
        print(f"❌ Connection failed with code: {rc}")

def on_message(client, userdata, msg):
    try:
        # Parse JSON data
        data = json.loads(msg.payload.decode())
        temperature = data.get('temperature', 0)
        humidity = data.get('humidity', 0)
        
        # Get current time
        now = datetime.now()
        timestamp_str = now.strftime('%H;%M;%S')
        
        # Determine label
        if temperature < 22:
            label = "DINGIN"
            label_encoded = 0
        elif temperature > 25:
            label = "PANAS"
            label_encoded = 2
        else:
            label = "NORMAL"
            label_encoded = 1
        
        # Prepare CSV row
        csv_data = [
            timestamp_str,
            round(float(temperature), 2),
            round(float(humidity), 2),
            label,
            label_encoded
        ]
        
        # Write to CSV (APPEND mode)
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(csv_data)
        
        # Print to console
        print(f"📥 Data saved:")
        print(f"   ⏰ Time: {timestamp_str}")
        print(f"   🌡️  Temp: {temperature}°C")
        print(f"   💧 Hum: {humidity}%")
        print(f"   🏷️  Label: {label}")
        print("-" * 40)
        
        # Count records in file
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f) - 1
        
        print(f"📊 Total records: {line_count}")
        
        # Stop after 15 records
        if line_count >= 15:
            print("\n🎯 Reached 15 records! Stopping...")
            client.disconnect()
            
    except Exception as e:
        print(f"❌ Error processing message: {e}")

# ==================== MAIN PROGRAM ====================
def main():
    print("\n" + "="*50)
    print("🌡️  DHT DATA COLLECTOR (15 Records)")
    print("="*50)
    
    # Setup CSV file
    setup_csv()
    
    # Create MQTT client
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set()
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Connect to broker
    try:
        print("🔗 Connecting to HiveMQ...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Start collecting
    print("\n🎯 Target: 15 records")
    print("Press Ctrl+C to stop early")
    print("="*50)
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        client.disconnect()
        
        # Final summary
        print("\n" + "="*50)
        print("📊 COLLECTION SUMMARY")
        print("="*50)
        
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_records = len(lines) - 1
                
                print(f"✅ Total records collected: {total_records}")
                print(f"💾 File saved: {CSV_PATH}")
                print(f"📁 Full path: {os.path.abspath(CSV_PATH)}")
                
                # Show file size
                file_size = os.path.getsize(CSV_PATH)
                print(f"📦 File size: {file_size} bytes")
                
                # Show sample data
                if total_records > 0:
                    print(f"\n📄 Sample data (first 3):")
                    for i, line in enumerate(lines[1:4], 1):
                        print(f"   {i}. {line.strip()}")
                    
                    # Count labels
                    label_counts = {"DINGIN": 0, "NORMAL": 0, "PANAS": 0}
                    for line in lines[1:]:
                        parts = line.strip().split(';')
                        if len(parts) >= 4:
                            label = parts[3]
                            if label in label_counts:
                                label_counts[label] += 1
                    
                    print(f"\n🏷️  Label distribution:")
                    for label, count in label_counts.items():
                        percentage = (count / total_records * 100) if total_records > 0 else 0
                        print(f"   {label}: {count} ({percentage:.1f}%)")
        else:
            print("❌ CSV file was not created!")
        
        print("\n💡 Next: Run 'python trainingmodel.py' to train ML models")
        print("="*50)

if __name__ == "__main__":
    main()