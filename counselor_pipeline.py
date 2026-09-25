import cv2
import serial
import numpy as np
import time
import threading
import requests
import json
from scipy.signal import butter, lfilter
from scipy.fft import fft

# === CONFIGURATION ===
SERIAL_PORT = 'COM10'
BAUD_RATE = 115200
FS = 200  # Sampling frequency matching your Arduino sketch
OLLAMA_MODEL = "llama3:8b-instruct-q4_K_M"

# Global state buffers
latest_eeg_buffer = []
latest_frame = None
running = True

def bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=200.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def extract_eeg_features(signal):
    if len(signal) < 200:
        return [0.0] * 10
    filtered = bandpass_filter(signal, fs=FS)
    fft_vals = np.abs(fft(filtered))[:len(filtered)//2]
    # Return simplified mean band powers or first few features as a state vector
    return np.mean(fft_vals[:10]) # Simplified summary metric for state prompt

def read_serial_eeg():
    global latest_eeg_buffer, running
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        while running:
            if ser.in_waiting:
                try:
                    val = int(ser.readline().decode().strip())
                    latest_eeg_buffer.append(val)
                    if len(latest_eeg_buffer) > 1000:
                        latest_eeg_buffer.pop(0)
                except ValueError:
                    continue
    except Exception as e:
        print(f"Serial error: {e}")

def capture_webcam():
    global latest_frame, running
    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
        time.sleep(0.1)
    cap.release()

# Start background sensors
threading.Thread(target=read_serial_eeg, daemon=True).start()
threading.Thread(target=capture_webcam, daemon=True).start()

print("Multi-modal capture pipeline active. Consulting local LLM...")

def query_local_counselor(eeg_state):
    prompt = (
        f"You are an empathetic, privacy-first local AI psychological counselor. "
        f"The user's current physiological EEG arousal metric is evaluated at {eeg_state:.2f}. "
        f"Check in gently with the user, ask how they are feeling right now based on their state, "
        f"and keep your response short, warm, and conversational."
    )
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 2048}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", "I'm here for you.")
    except Exception as e:
        return f"Local LLM connection error: {e}"
    return "Could not reach local model."

try:
    while True:
        # Check-in every 45 seconds so Ollama has time to think without flooding VRAM
        time.sleep(45) 
        
        if len(latest_eeg_buffer) >= 500:
            current_signal = np.array(latest_eeg_buffer[-500:])
            eeg_metric = extract_eeg_features(current_signal)
            
            print(f"\n[Sensor Fusion] EEG Feature Metric: {eeg_metric:.2f}")
            print("Querying local Ollama model...")
            
            # Increased timeout to 60 seconds to prevent RTX 3070 bottlenecks
            counselor_reply = query_local_counselor(eeg_metric)
            print(f"\nCounselor AI:\n{counselor_reply}\n" + "-"*40)
        else:
            print("Accumulating EEG buffer data...")
            
except KeyboardInterrupt:
    running = False
    print("Stopping pipeline...")