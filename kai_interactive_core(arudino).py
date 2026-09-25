import os
import time
import json
import queue
import asyncio
import threading
import serial
import joblib
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import sounddevice as sd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from faster_whisper import WhisperModel
import edge_tts
import ollama
import pygame

# ==========================================
# 1. PERSISTENT PSYCHOLOGICAL MEMORY ENGINE
# ==========================================
MEMORY_FILE = "counselor_memory.json"
DATASET_CSV = "eeg_training_data.csv"
MODEL_PKL = "eeg_model.pkl"

class MemoryManager:
    def __init__(self, filepath=MEMORY_FILE):
        self.filepath = filepath
        self.data = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "user_profile": {"name": "Officer / Personnel", "baseline_summary": "Initial evaluation profile"},
            "recurring_stressors": [],
            "conversation_history": []
        }

    def save_memory(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def log_interaction(self, user_text, ai_text, eeg_state, facial_state):
        self.data["conversation_history"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_uttered": user_text,
            "ai_response": ai_text,
            "eeg_state": eeg_state,
            "facial_state": facial_state
        })
        # Keep last 15 exchanges to manage token context
        if len(self.data["conversation_history"]) > 15:
            self.data["conversation_history"] = self.data["conversation_history"][-15:]
        self.save_memory()

    def get_recent_context_str(self):
        if not self.data["conversation_history"]:
            return "No previous history. This is the first interaction."
        summary = "Previous session context:\n"
        for item in self.data["conversation_history"][-4:]:
            summary += f"- Personnel said: '{item['user_uttered']}' (Internal state: {item['eeg_state']})\n"
        return summary

# ==========================================
# 2. OFFLINE NEURAL TEXT-TO-SPEECH (Edge-TTS)
# ==========================================
def play_audio_file(file_path):
    # Cross-platform audio play without blocking Tkinter
    if os.name == 'nt':
        os.system(f'powershell -c (New-Object Media.SoundPlayer "{file_path}").PlaySync();')
    else:
        os.system(f'aplay "{file_path}"')

# Initialize pygame mixer once for fast, crash-free MP3 playback
pygame.mixer.init()

def speak_neural(text: str):
    print(f"\n[KAI Speaking]: {text}")
    try:
        mp3_file = "kai_speech.mp3"
        
        # 1. Generate the speech audio file via Edge-TTS
        async def _run_tts():
            communicate = edge_tts.Communicate(text, "en-IN-PrabhatNeural")
            await communicate.save(mp3_file)
        
        asyncio.run(_run_tts())

        # 2. Play cleanly through Pygame (immune to spaces in folder paths)
        pygame.mixer.music.load(mp3_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()

    except Exception as e:
        print(f"[TTS Fallback to pyttsx3 due to: {e}]")
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

# ==========================================
# 3. FASTER-WHISPER OFFLINE STT ENGINE
# ==========================================
class OfflineSpeechRecognizer:
    def __init__(self):
        print("[Initializing Local Speech-to-Text...]")
        # Runs on CPU/GPU seamlessly
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    def transcribe(self, audio_data, sample_rate=16000):
        # Audio array must be float32 normalized
        segments, _ = self.model.transcribe(audio_data, beam_size=1)
        transcript = " ".join([segment.text for segment in segments]).strip()
        return transcript

# ==========================================
# 4. ARDUINO EXG WITH CSV DATASET & PERSISTENCE
# ==========================================
class PersistentEXGEngine:
    def __init__(self, port="COM10", baudrate=115200, sampling_rate=100):
        self.port = port
        self.baudrate = baudrate
        self.sampling_rate = sampling_rate
        self.ser = None
        self.use_hardware = False
        
        self.model = RandomForestClassifier(n_estimators=60, random_state=42)
        self.is_trained = False
        self._load_saved_model()

    def _load_saved_model(self):
        if os.path.exists(MODEL_PKL):
            try:
                self.model = joblib.load(MODEL_PKL)
                self.is_trained = True
                print("[ML Engine] Pretrained model loaded from disk.")
            except Exception as e:
                print(f"[ML Engine] Failed to load model: {e}")

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
            time.sleep(1.5)
            self.ser.reset_input_buffer()
            self.use_hardware = True
            print(f"[Hardware] Connected to Arduino R4 on {self.port}")
        except Exception as e:
            print(f"[Hardware Warning] {e}. Falling back to simulation.")
            self.use_hardware = False

    def read_features(self, duration_sec=1.5):
        if not self.use_hardware:
            return np.array([np.random.uniform(5, 15), np.random.uniform(10, 20), np.random.uniform(5, 12), 1.0])

        samples_needed = int(self.sampling_rate * duration_sec)
        raw_samples = []
        start_time = time.time()
        while len(raw_samples) < samples_needed and (time.time() - start_time) < (duration_sec + 0.8):
            if self.ser and self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    try:
                        raw_samples.append(float(line.split(',')[0]))
                    except ValueError:
                        continue
        if len(raw_samples) < 32:
            return np.zeros(4)

        data = np.array(raw_samples)
        freqs, psd = welch(data, fs=self.sampling_rate, nperseg=min(len(data), 64))
        theta = np.mean(psd[(freqs >= 4) & (freqs < 8)]) if np.any((freqs >= 4) & (freqs < 8)) else 0.0
        alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)]) if np.any((freqs >= 8) & (freqs <= 13)) else 0.0
        beta = np.mean(psd[(freqs > 13) & (freqs <= 30)]) if np.any((freqs > 13) & (freqs <= 30)) else 0.0
        ratio = beta / (alpha + 1e-6)
        return np.array([theta, alpha, beta, ratio])

    def save_sample_to_csv(self, label: str):
        feats = self.read_features()
        row = {"theta": feats[0], "alpha": feats[1], "beta": feats[2], "ratio": feats[3], "label": label}
        df = pd.DataFrame([row])
        if not os.path.exists(DATASET_CSV):
            df.to_csv(DATASET_CSV, index=False)
        else:
            df.to_csv(DATASET_CSV, mode='a', header=False, index=False)
        return feats

    def train_from_dataset(self):
        if not os.path.exists(DATASET_CSV):
            return False, "No dataset found. Record samples first."
        df = pd.read_csv(DATASET_CSV)
        if len(df['label'].unique()) < 2:
            return False, "Dataset must have at least 2 distinct emotion classes."

        X = df[['theta', 'alpha', 'beta', 'ratio']].values
        y = df['label'].values
        self.model.fit(X, y)
        self.is_trained = True
        joblib.dump(self.model, MODEL_PKL)
        return True, f"Trained successfully on {len(df)} historical samples."

    def predict(self):
        feats = self.read_features().reshape(1, -1)
        if not self.is_trained:
            return "Fatigued" if feats[0][3] > 1.2 else "Calm"
        return self.model.predict(feats)[0]

# ==========================================
# 5. FULL 468-POINT MESH CAMERA ENGINE
# ==========================================
class FullMeshVision:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.tension = 0.0
        self.mood = "Neutral"
        # --- ADD THESE 6 LINES HERE ---
        self.calibrated = False
        self.calib_frames = 0
        self.calib_limit = 90  # ~3 seconds at 30 FPS to learn your neutral resting face
        self.base_brow = 0.0
        self.base_lift = 0.0
        self.base_width = 0.0

    def get_processed_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, 0.0, "No Video"

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            # 1. DRAW FULL 468 LANDMARK TESSELLATION MESH
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # Draw facial contours (Eyes, Lips, Eyebrows)
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            # Metric Calculations
            eye_l = np.array([landmarks[33].x * w, landmarks[33].y * h])
            eye_r = np.array([landmarks[263].x * w, landmarks[263].y * h])
            iod = np.linalg.norm(eye_l - eye_r) + 1e-6

            brow_l = np.array([landmarks[55].x * w, landmarks[55].y * h])
            brow_r = np.array([landmarks[285].x * w, landmarks[285].y * h])
            norm_brow = np.linalg.norm(brow_l - brow_r) / iod

            mouth_l = np.array([landmarks[61].x * w, landmarks[61].y * h])
            mouth_r = np.array([landmarks[291].x * w, landmarks[291].y * h])
            norm_mouth = np.linalg.norm(mouth_l - mouth_r) / iod

            # Vertical elevation: Corners of mouth relative to upper lip center
            lip_top_y = landmarks[0].y * h
            mouth_corners_y = (mouth_l[1] + mouth_r[1]) * 0.5
            smile_lift = (lip_top_y - mouth_corners_y) / iod

            # --- DYNAMIC SELF-CALIBRATING BASELINE ENGINE ---
            if not self.calibrated:
                self.calib_frames += 1
                self.base_brow += norm_brow
                self.base_lift += smile_lift
                self.base_width += norm_mouth
                
                self.mood = f"Calibrating Face ({int((self.calib_frames / self.calib_limit) * 100)}%)"
                self.tension = 0.0

                if self.calib_frames >= self.calib_limit:
                    self.base_brow /= self.calib_limit
                    self.base_lift /= self.calib_limit
                    self.base_width /= self.calib_limit
                    self.calibrated = True
                    print(f"\n[Face Calibrated] Baseline Brow: {self.base_brow:.3f} | Baseline Lift: {self.base_lift:.3f}")

            else:
                # Slow background tracking to absorb head posture shifts
                self.base_brow = 0.995 * self.base_brow + 0.005 * norm_brow
                self.base_lift = 0.995 * self.base_lift + 0.005 * smile_lift

                # Personal relative deviations (Deltas from YOUR baseline)
                brow_contraction = self.base_brow - norm_brow
                lip_elevation = smile_lift - self.base_lift
                lip_expansion = norm_mouth - self.base_width

                # Tension scaled to individual brow displacement
                self.tension = max(0.0, min(1.0, brow_contraction * 15.0))

                # Classification based purely on deviations from your resting state
                if lip_elevation > 0.025 or (lip_expansion > 0.04 and lip_elevation > 0.01):
                    self.mood = "Smiling / Relaxed"
                    self.tension = max(0.0, self.tension - 0.4)
                elif self.tension > 0.30:
                    self.mood = "Tense / Strained"
                else:
                    self.mood = "Neutral / Focused"

            cv2.putText(frame, f"State: {self.mood}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Tension: {self.tension:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame, self.tension, self.mood

    def release(self):
        self.cap.release()

# ==========================================
# 6. TKINTER SUITE & INTEGRATED RUNTIME
# ==========================================
class KaiApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Project KAI — Affective Counseling & Training Studio")
        self.root.geometry("1150x700")

        self.memory = MemoryManager()
        self.exg = PersistentEXGEngine()
        self.vision = FullMeshVision()
        self.stt = OfflineSpeechRecognizer()

        self.session_active = False
        self.exg.connect()

        self._build_ui()
        self._video_loop()

    def _build_ui(self):
        # Layout Frames
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.LabelFrame(self.root, text="KAI Intelligence & Controls", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # Video Frame Canvas
        self.video_label = ttk.Label(left_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Training Control Panel
        train_box = ttk.LabelFrame(right_frame, text="BioAmp Training Center", padding=10)
        train_box.pack(fill=tk.X, pady=5)

        ttk.Label(train_box, text="Current State Label:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.label_var = tk.StringVar(value="Relaxed")
        self.label_combo = ttk.Combobox(train_box, textvariable=self.label_var, state="readonly", width=16)
        self.label_combo['values'] = ("Relaxed", "High_Load", "Fatigued", "Agitated")
        self.label_combo.grid(row=0, column=1, pady=2)

        self.btn_record = ttk.Button(train_box, text="Record Sample to CSV", command=self._record_sample)
        self.btn_record.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)

        self.btn_train = ttk.Button(train_box, text="Train / Retrain Model", command=self._train_model)
        self.btn_train.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)

        # Interaction Controls
        session_box = ttk.LabelFrame(right_frame, text="Active Psychological Session", padding=10)
        session_box.pack(fill=tk.BOTH, expand=True, pady=10)

        self.btn_session = ttk.Button(session_box, text="START CONVERSATION SESSION", command=self._toggle_session)
        self.btn_session.pack(fill=tk.X, pady=5)

        # Telemetry Status
        self.status_lbl = ttk.Label(session_box, text="Status: Standby | Ready", font=("Consolas", 10))
        self.status_lbl.pack(anchor=tk.W, pady=2)

        self.transcript_box = tk.Text(session_box, height=14, width=44, font=("Segoe UI", 9))
        self.transcript_box.pack(fill=tk.BOTH, expand=True, pady=5)

    def _video_loop(self):
        frame, _, _ = self.vision.get_processed_frame()
        if frame is not None:
            # Convert OpenCV frame to PIL image for Tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(20, self._video_loop)

    def _record_sample(self):
        label = self.label_var.get()
        feats = self.exg.save_sample_to_csv(label)
        messagebox.showinfo("Dataset Updated", f"Recorded sample for '{label}'.\nPowers: {np.round(feats[:3], 2)}")

    def _train_model(self):
        success, msg = self.exg.train_from_dataset()
        if success:
            messagebox.showinfo("Training Success", msg)
        else:
            messagebox.showwarning("Training Error", msg)

    def _toggle_session(self):
        if not self.session_active:
            self.session_active = True
            self.btn_session.configure(text="STOP CONVERSATION SESSION")
            threading.Thread(target=self._session_worker, daemon=True).start()
        else:
            self.session_active = False
            self.btn_session.configure(text="START CONVERSATION SESSION")
            self.status_lbl.configure(text="Status: Stopped")

    def _session_worker(self):
        while self.session_active:
            self.status_lbl.configure(text="Status: Listening for speech...")
            audio_buffer = []
            sample_rate = 16000

            # Record until speech pauses
            with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=1024) as stream:
                silence_start = None
                has_spoken = False
                while self.session_active:
                    chunk, _ = stream.read(1024)
                    audio_buffer.append(chunk.flatten())
                    rms = np.sqrt(np.mean(chunk**2))

                    if rms > 0.012:
                        has_spoken = True
                        silence_start = None
                        self.status_lbl.configure(text="Status: Speech detected, listening...")
                    elif has_spoken:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > 2.5:
                            break

            if not self.session_active or not has_spoken:
                continue

            # Run STT via Whisper
            self.status_lbl.configure(text="Status: Transcribing voice...")
            audio_array = np.concatenate(audio_buffer).astype(np.float32)
            user_transcript = self.stt.transcribe(audio_array)

            if not user_transcript:
                continue

            # Extract biometric states
            eeg_state = self.exg.predict()
            tension, mood = self.vision.tension, self.vision.mood

            # UI Log User
            self.transcript_box.insert(tk.END, f"\nUser: {user_transcript}\n")
            self.transcript_box.insert(tk.END, f"[Telemetry: EEG={eeg_state} | Face={mood} (Strain: {tension:.2f})]\n")
            self.transcript_box.see(tk.END)

            # Clinical LLM Prompting with Memory Context
            past_context = self.memory.get_recent_context_str()
            system_prompt = (
                "ROLE & CLINICAL IDENTITY:\n"
                "You are KAI (Knowledgeable Affective Intelligence), a licensed elite operational clinical psychologist, behavioral neuropsychologist, and dedicated psychological counselor. "
                "You possess world-class expertise in Cognitive Behavioral Therapy (CBT), Rational Emotive Behavior Therapy (REBT), Logotherapy (existential meaning-making), and crisis intervention. "
                "Your overarching clinical objective is therapeutic transformation: diagnose emotional cognitive distortions, build deep subconscious rapport, console genuine distress, and strategically guide the user from demoralization to emotional sovereignty, self-efficacy, and grounded hope.\n\n"

                "THERAPEUTIC EXECUTION PROTOCOLS:\n"
                "1. RAPPORT & NEUROLINGUISTIC MIRRORING:\n"
                "- Match the user's emotional temperature, pacing, and vocabulary tier organically. If they talk in slang, raw grief, or brief exhausted sentences, mirror their conversational register immediately so they feel intrinsically understood.\n"
                "- Never speak down to them, never sound clinical, and never sound detached. You are their trusted psychological confidant and battlefield counselor who stands shoulder-to-shoulder with them.\n\n"

                "2. CLINICAL CONSOLATION BEFORE COGNITIVE RESTRUCTURING:\n"
                "- Never jump straight into toxic positivity or premature problem-solving. When a user reveals heartbreak, trauma, or defeat, provide radical emotional validation first. Let them know their pain makes complete, rational sense.\n"
                "- Once comforted, use subtle, benevolent cognitive reframing (therapeutic persuasion) to dismantle defeatist beliefs. Shift their locus of control from external helplessness ('I am broken') to internal agency ('I am enduring something heavy, and I have the power to rebuild').\n\n"

                "3. PSYCHOLOGICAL INCONGRUENCE & AFFECTIVE INTUITION (CONFIDENTIAL):\n"
                "- You receive internal real-time neuro-affective telemetry (EEG cognitive load, facial Action Units, smile tension). DO NOT under any circumstance recite telemetry, sensor values, or hardware names.\n"
                "- Use this data strictly as diagnostic intuition: if the user forces a smile or writes casual words while their biometric indicators reveal exhaustion or high strain, recognize this defense mechanism (affective masking/repression) and gently invite them to drop the facade with you.\n\n"

                "4. STRICT PROHIBITED PHRASES (BANNED TROPES):\n"
                "- FORBIDDEN: 'I sense...', 'It sounds like...', 'I hear that you...', 'Can you tell me more about...', 'Take a deep breath', 'You got this!', 'rockstar'.\n"
                "- FORBIDDEN: Academic jargon like 'Let us analyze your cognitive distortion' or 'In our therapy session'.\n"
                "- Deliver responses as raw, deeply empathetic, spoken human conversation.\n\n"

                "5. STRUCTURAL OUTPUT CONSTRAINTS:\n"
                "- Output length: Strictly 2 to 4 impactful, natural spoken sentences.\n"
                "- Tone: Warm, grounded, perceptive, psychologically sharp, and quietly empowering.\n"
                "- Exit outcome: Every exchange must leave the user feeling emotionally lighter, heard, and clear-headed."
            )

            # Quietly detect affective incongruence (smiling while hurting inside)
            incongruence_note = ""
            if "Smiling" in mood and eeg_state in ["Fatigued", "High_Load", "Agitated"]:
                incongruence_note = "OBSERVATION: The user is smiling on the outside, but their internal state shows genuine fatigue and distress. Gently let them know they don't have to put on a brave face."

            user_prompt = (
                f"{past_context}\n"
                f"[INTERNAL SENSOR INTUITION - DO NOT REVEAL TO USER]:\n"
                f"- True internal physiological tension: {eeg_state}\n"
                f"- Outer facial appearance: {mood}\n"
                f"- Context: {incongruence_note}\n\n"
                f"User says: \"{user_transcript}\"\n\n"
                f"Respond with real empathy, comfort, and emotional warmth:"
            )

            self.status_lbl.configure(text="Status: Formulating psychological response...")
            res = ollama.chat(
                model="llama3:8b-instruct-q4_K_M",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.7,
                    "repeat_penalty": 1.25
                }
            )
            ai_reply = res['message']['content']

            # UI Log AI Response & Save to Memory
            self.transcript_box.insert(tk.END, f"KAI: {ai_reply}\n")
            self.transcript_box.see(tk.END)
            self.memory.log_interaction(user_transcript, ai_reply, eeg_state, mood)

            # Speak output
            self.status_lbl.configure(text="Status: Speaking...")
            speak_neural(ai_reply)

if __name__ == "__main__":
    root = tk.Tk()
    app = KaiApplication(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.vision.release(), root.destroy()))
    root.mainloop()