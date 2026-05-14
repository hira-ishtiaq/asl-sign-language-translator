import subprocess
import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle
import time
import pyttsx3
import threading
import os
from collections import Counter

# ── Load model ──────────────────────────────────────────────────────────────────
MODEL_PATH = './model/asl_model.pkl'
if not os.path.exists(MODEL_PATH):
    print("❌ Model not found. Run train_model.py first.")
    exit()

with open(MODEL_PATH, 'rb') as f:
    saved = pickle.load(f)
model  = saved['model']
labels = saved['labels']
print(f"✅ Model loaded — {len(labels)} classes")

# ── MediaPipe setup ─────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                             min_detection_confidence=0.7)

# ── Text to Speech ──────────────────────────────────────────────────────────────
def speak(text):
    def _speak():
        import subprocess
        subprocess.Popen(
            ['powershell', '-Command',
             f"Add-Type -AssemblyName System.Speech; "
             f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             f"$s.Speak('{text}')"],
            creationflags=0x08000000
        )
    threading.Thread(target=_speak, daemon=True).start()


# ── Extract landmarks ───────────────────────────────────────────────────────────
def extract_landmarks(hand_landmarks):
    coords = []
    x_vals = [lm.x for lm in hand_landmarks.landmark]
    y_vals = [lm.y for lm in hand_landmarks.landmark]
    min_x, min_y = min(x_vals), min(y_vals)
    for lm in hand_landmarks.landmark:
        coords.append(lm.x - min_x)
        coords.append(lm.y - min_y)
    return coords

# ── Settings ────────────────────────────────────────────────────────────────────
CONFIRM_FRAMES    = 20    # consecutive frames needed to confirm a letter
CONFIDENCE_THRESH = 0.55  # minimum confidence to consider a prediction valid
COOLDOWN_FRAMES   = 15    # frames to wait after confirming before next letter

# ── State ───────────────────────────────────────────────────────────────────────
prediction_buffer = []
current_word      = []
sentence          = []
cooldown_counter  = 0

# ── UI drawing ──────────────────────────────────────────────────────────────────
def draw_ui(frame, predicted, confidence, buffer_progress, word, sentence_words, cooldown):
    h, w = frame.shape[:2]

    # Top bar
    cv.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)

    # Predicted letter
    color = (0, 255, 150) if confidence >= CONFIDENCE_THRESH else (80, 80, 80)
    cv.putText(frame, predicted if predicted else '?', (25, 75),
               cv.FONT_HERSHEY_SIMPLEX, 2.8, color, 5)

    # Confidence
    cv.putText(frame, f'{confidence*100:.0f}%', (130, 55),
               cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 210, 20, w - 230, 35
    cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

    if cooldown > 0:
        fill = int(bar_w * (1 - cooldown / COOLDOWN_FRAMES))
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 140, 255), -1)
        cv.putText(frame, 'Cooldown...', (bar_x + 8, bar_y + 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        fill = int(bar_w * min(buffer_progress, 1.0))
        fill_color = (0, 255, 255) if buffer_progress >= 1.0 else (0, 220, 100)
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), fill_color, -1)
        cv.putText(frame, 'Hold steady to confirm', (bar_x + 8, bar_y + 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Current word
    word_str = ''.join(word)
    cv.rectangle(frame, (0, h - 115), (w, h - 65), (30, 30, 30), -1)
    cv.putText(frame, f'Word:  {word_str}_', (20, h - 78),
               cv.FONT_HERSHEY_SIMPLEX, 1.1, (255, 220, 0), 2)

    # Sentence
    cv.rectangle(frame, (0, h - 60), (w, h), (20, 20, 20), -1)
    cv.putText(frame, f'Sentence: {" ".join(sentence_words)}', (20, h - 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

    # Controls hint
    cv.putText(frame, 'SPACE=word  ENTER=speak  BACKSPACE=delete  C=clear  Q=quit',
               (10, h - 125), cv.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

    return frame

# ── Main loop ───────────────────────────────────────────────────────────────────
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("\n" + "=" * 55)
print("  ✋  ASL Sign Language Translator — Live")
print("=" * 55)
print("  Hold any ASL sign steady to confirm the letter")
print("  SPACE      → complete current word")
print("  ENTER      → speak the full sentence")
print("  BACKSPACE  → delete last letter")
print("  C          → clear everything")
print("  Q          → quit")
print("=" * 55 + "\n")

speak("ASL Translator ready.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    predicted  = ''
    confidence = 0.0

    if results.multi_hand_landmarks:
        hand_lms   = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        landmarks  = extract_landmarks(hand_lms)
        proba      = model.predict_proba([landmarks])[0]
        confidence = float(np.max(proba))
        predicted  = labels[int(np.argmax(proba))]

        if confidence >= CONFIDENCE_THRESH and cooldown_counter == 0:
            prediction_buffer.append(predicted)
            if len(prediction_buffer) > CONFIRM_FRAMES:
                prediction_buffer.pop(0)

            # Confirm if last CONFIRM_FRAMES are all the same letter
            if len(prediction_buffer) == CONFIRM_FRAMES:
                most_common, count = Counter(prediction_buffer).most_common(1)[0]
                if count == CONFIRM_FRAMES:
                    current_word.append(most_common)
                    speak(most_common)
                    print(f"  ✅ Confirmed: {most_common}  |  Word: {''.join(current_word)}")
                    prediction_buffer.clear()
                    cooldown_counter = COOLDOWN_FRAMES
        else:
            if confidence < CONFIDENCE_THRESH:
                prediction_buffer.clear()
    else:
        prediction_buffer.clear()
        predicted  = ''
        confidence = 0.0

    # Tick cooldown
    if cooldown_counter > 0:
        cooldown_counter -= 1

    # Draw UI
    buffer_progress = len(prediction_buffer) / CONFIRM_FRAMES
    frame = draw_ui(frame, predicted, confidence, buffer_progress,
                    current_word, sentence, cooldown_counter)
    cv.imshow('ASL Translator  ✋', frame)

    # ── Key controls ─────────────────────────────────────────────────────────
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        if current_word:
            word = ''.join(current_word)
            sentence.append(word)
            speak(word)
            print(f"  📝 Word: {word}")
            current_word = []
            prediction_buffer.clear()

    elif key == 13:  # ENTER
        if sentence:
            full = ' '.join(sentence)
            speak(full)
            print(f"  🔊 Speaking: {full}")

    elif key == 8:  # BACKSPACE
        if current_word:
            removed = current_word.pop()
            print(f"  ⌫ Deleted: {removed}")
        elif sentence:
            last = sentence.pop()
            current_word = list(last)
            print(f"  ⌫ Restored: {last}")
        prediction_buffer.clear()

    elif key == ord('c'):
        current_word = []
        sentence     = []
        prediction_buffer.clear()
        print("  🗑️  Cleared")

cap.release()
cv.destroyAllWindows()
print("\n  👋 Goodbye!")