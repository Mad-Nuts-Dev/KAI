<div align="center">

# 🧠⚡ Project KAI (Knowledgeable Artificial Intelligence)
### *A Human-Centric Voice & Prefrontal EEG Brain-Computer Interface Assistant for Mental Health & Biofeedback*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Arduino](https://img.shields.io/badge/Arduino-Uno%20R4%20Minima-00979D?style=flat-square&logo=arduino)](https://www.arduino.cc/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20RTX%203070-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3%3A8b-black?style=flat-square)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🌟 1. Project Title & Overview
**Project KAI (Knowledgeable Artificial Intelligence)** is an edge-powered, offline affective computing and counseling ecosystem designed to bridge the gap between raw biopotential signal acquisition and compassionate, privacy-first generative AI counseling. Built entirely to run locally on consumer hardware (**NVIDIA RTX 3070, 8GB VRAM**), KAI fuses real-time prefrontal cortex EEG biopotentials, computer vision facial expression tracking, and vocal prosody into a closed-loop inference engine.

---

## 🛑 2. Problem Statement
Traditional digital mental health platforms lack objective physiological grounding and compromise user data privacy by transmitting sensitive biometric, facial, and emotional logs to third-party cloud servers, while current wearable biofeedback tools fail to provide real-time, interactive, empathetic interventions.

---

## ⚙️ 3. Solution Proposed (Features, Technology & Mechanisms)
Project KAI is an offline desktop application (`kai.py`) that synchronizes real-time physiological and behavioral telemetry to generate context-aware psychological support entirely on-device.

* **Hardware Acquisition:** Utilizes the **BioAmp EXG Pill** (low-noise analog front-end) paired with an **Arduino Uno R4 Minima** (32-bit RA4M1 ARM processor with a 12-bit ADC) running precise microsecond sampling (`micros()`) at **512 Hz**. 
* **3-Lead Electrode Placement:** Configured with `IN+` on the prefrontal cortex (forehead center) and `IN-`/`REF` on the earlobes for optimal common-mode noise rejection.
* **Multimodal Sensor Fusion:** 
  * *EEG Pipeline:* Streams asynchronous USB serial data, applying Welch's PSD and Fast Fourier Transform (FFT) feature extraction across Theta ($4–8\text{ Hz}$), Alpha ($8–13\text{ Hz}$), Beta ($13–30\text{ Hz}$), and Gamma ($30+\text{ Hz}$) bands. Trained via Scikit-Learn Random Forest classifiers.
  * *Vision Pipeline:* Employs MediaPipe Face Mesh (468-point landmark tessellation) with a dynamic self-calibrating baseline engine to track facial micro-expressions, smile lift, and brow tension.
  * *Speech Pipeline:* Features offline Speech-to-Text via Faster-Whisper (`tiny.en`) and Edge-TTS neural text-to-speech for zero cloud latency.
* **Edge Intelligence:** Powered by a quantized local large language model (`llama3:8b-instruct-q4_K_M`) executed locally via **Ollama**.

---

## 🚀 4. Implementation Plan & Deployment
* **Target Beneficiaries:** Students, remote professionals, and high-stress operational personnel who require confidential, stigma-free mental health support.
* **Phase-Wise Deployment:**
  * *Phase 1:* Hardware calibration of the BioAmp EXG Pill and Arduino 512Hz microsecond serial stream.
  * *Phase 2:* Integration of the Python Tkinter dashboard (`kai.py`), connecting MediaPipe vision, Faster-Whisper STT, and persistent JSON memory (`counselor_memory.json`).
  * *Phase 3:* Pilot testing to measure stress mitigation accuracy, facial tension tracking, and local Ollama inference speeds.
* **Expected Deliverables:** Fully functional local desktop application, open-source hardware wiring guides, trained ML models (`eeg_model.pkl`), and evaluation pitch deck documentation (`docs/KAI_PitchDeck_Final.pdf`).

---

## 🔒 5. Safety and Efficiency Analysis
* **Absolute Data Sovereignty & Privacy:** By running 100% locally on user hardware (RTX 3070), sensitive biometric feeds, facial video matrices, audio transcripts, and mental health conversation logs never leave the device, completely eliminating cloud storage vulnerabilities and third-party tracking.
* **Hardware Safety:** The BioAmp EXG Pill operates on low-voltage DC powered via USB-isolated microcontrollers, guaranteeing complete electrical safety for skin-contact biomedical recording.
* **Operational Efficiency:** Optimized 4-bit quantization (`Q4_K_M`) keeps VRAM consumption under 5.5 GB, leaving ample headroom for real-time sensor processing and maintaining sub-second conversational response rates (~45 tokens/sec).

---

## 📈 6. Scalability and Future Development
* **Multi-Channel Expansion:** Architectural readiness to scale from a single prefrontal EXG Pill channel to multi-channel headset arrays (e.g., Fp1/Fp2 frontal asymmetry mapping).
* **Hardware Portability:** Migration from desktop setups to standalone edge devices (such as NVIDIA Jetson Nano or Raspberry Pi 5 with dedicated neural accelerators).
* **Clinical & Healthcare Adaptations:** Integration with professional clinical dashboards for authorized therapists to review encrypted progress metrics with patient consent, alongside assistive BCI applications for motor-impaired and ALS patients to restore digital communication autonomy.

---

## 💡 7. Innovation Advantage
Unlike commercial mental health apps that rely solely on subjective self-reporting (ignoring biological reality) or expensive clinical EEGs requiring cloud uploads, Project KAI uniquely bridges **low-cost DIY biopotential hardware (BioAmp EXG Pill)** with **fully offline multi-modal edge AI**. It introduces true biometric grounding into AI counseling, detecting affective masking (such as smiling while experiencing cognitive fatigue or deep stress) while maintaining absolute data privacy.

---

## 👁️ 8. Visual Impact & User Journey
The user journey transitions from reactive, unmeasured stress to proactive, quantified self-awareness:

| Metric / Stage | Traditional Online Counseling (Before) | Project KAI Ecosystem (After) |
| :--- | :--- | :--- |
| **Data Privacy** | Cloud-stored sensitive mental health logs; high third-party breach risk. | 100% local offline processing; absolute bio-privacy and zero cloud exposure. |
| **Emotional Insight** | Subjective self-reporting; highly prone to emotional masking or bias. | Objective multi-modal fusion (Prefrontal EEG arousal + Mic prosody + MediaPipe facial tracking). |
| **Intervention Quality** | Delayed clinical appointments or generic, context-blind chatbot scripts. | Immediate, empathetic edge AI counseling matched in real-time to cognitive load. |

> **User Journey Flow:** User attaches EXG Pill $\rightarrow$ Background script captures 512Hz EEG stream & MediaPipe face mesh $\rightarrow$ Local LLM analyzes fused state vector with memory context $\rightarrow$ Empathetic guidance delivered instantly offline via Edge-TTS.

---

## 🛠️ 9. Step-by-Step Installation & Setup Guide

Step 1: Clone the Repository

  ```bash
  git clone [https://github.com/YourUsername/KAI.git](https://github.com/YourUsername/KAI.git)
  cd KAI
  ```
Step 2: Flash the Arduino Firmware

1. Open the Arduino IDE.

2. Load the sketch located in arduino/eeg_sampler_512hz/arduino_code_2.ino.

3. Select your board (Arduino Uno R4 Minima) and correct COM port.

4. Upload the code (analogReadResolution(12); ensures 12-bit precision).

Step 3: Set Up Python Environment & Dependencies

Ensure Python 3.10+ is installed. Create a virtual environment and install requirements:

```bash
  python -m venv venv
  # On Windows:
  venv\Scripts\activate
  # On Linux/macOS:
  source venv/bin/activate

  pip install pyserial numpy scipy scikit-learn pandas opencv-python mediapipe sounddevice faster-whisper edge-tts ollama pygame pillow joblib requests
  ```

Step 4: Configure and Run Ollama (Local LLM)

1. Download and install [Ollama](https://ollama.com/).

2. Pull the required local model:

  ```bash
  ollama pull llama3:8b-instruct-q4_K_M
  ```

3. Ensure Ollama is running locally at http://localhost:11434.

Step 5: Run Project KAI

1. Open python_backend/kai.py and verify your Arduino serial port configuration (e.g., COM10):

   ```bash
   self.exg = PersistentEXGEngine(port="COM10", baudrate=115200, sampling_rate=512)
   ```

2. Launch the application:

   ```bash
   python python_backend/kai.py
   ```

---

## 🏆 10. Acknowledgments

Developed for national technical evaluations and hackathons (including INSPIRE Awards MANAK, DST, Govt. of India, and MNIT Jaipur, and the Vishwakarma Awards 2026, IIT Indore). Built indigenously under Atmanirbhar Bharat principles for accessible, stigma-free mental health support.

License: MIT
   
---

## 📂 11. Repository Structure
  ```text
  ├── arduino/
  │   └── eeg_sampler_512hz/        # Arduino 12-bit 512Hz microsecond sampling sketch
  ├── python_backend/
  │   ├── kai.py                    # Main Tkinter application & multi-modal core runtime
  │   ├── counselor_memory.json     # Persistent psychological memory engine
  │   └── eeg_training_data.csv     # Local dataset for BioAmp emotion classification
  ├── docs/
  │   └── KAI_PitchDeck_Final.pdf   # Official presentation deck for evaluation juries
  └── README.md


