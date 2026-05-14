import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR        = './data'
SAMPLES_PER_CLASS = 100        # how many samples to collect per letter
LABELS          = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # A-Z

# ── MediaPipe setup ─────────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
hands       = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                              min_detection_confidence=0.5)

# ── Create data directory ───────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
for label in LABELS:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

# ── Helper: extract 21 landmarks as flat list of (x, y) ────────────────────────
def extract_landmarks(hand_landmarks):
    coords = []
    x_vals = [lm.x for lm in hand_landmarks.landmark]
    y_vals = [lm.y for lm in hand_landmarks.landmark]
    min_x, min_y = min(x_vals), min(y_vals)
    for lm in hand_landmarks.landmark:
        coords.append(lm.x - min_x)   # normalize relative to wrist
        coords.append(lm.y - min_y)
    return coords                      # 42 values total (21 points × x,y)

# ── Main collection loop ────────────────────────────────────────────────────────
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("=" * 55)
print("  ASL Data Collection — Hand Landmark Recorder")
print("=" * 55)
print(f"  Collecting {SAMPLES_PER_CLASS} samples for each of {len(LABELS)} letters")
print("  Press SPACE to start collecting for each letter")
print("  Press Q to quit early")
print("=" * 55)

dataset     = []
data_labels = []

for label in LABELS:
    label_dir    = os.path.join(DATA_DIR, label)
    existing     = len(os.listdir(label_dir))

    if existing >= SAMPLES_PER_CLASS:
        print(f"  ✅ '{label}' already has {existing} samples — skipping")
        continue

    # ── Wait for user to get ready ──────────────────────────────────────────────
    print(f"\n  📸 Get ready for letter: [ {label} ]")
    print(f"     Make the ASL sign for '{label}' and press SPACE")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)

        # Overlay instruction
        cv.rectangle(frame, (0, 0), (frame.shape[1], 80), (30, 30, 30), -1)
        cv.putText(frame, f"Sign: [ {label} ]  —  Press SPACE to start",
                   (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 200), 2)

        cv.imshow('Data Collection', frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            print("\n  Exiting early...")
            exit()

    # ── Collect samples ─────────────────────────────────────────────────────────
    count = existing
    print(f"  🔴 Recording... (need {SAMPLES_PER_CLASS - existing} more samples)")

    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break
        frame     = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            landmarks = extract_landmarks(hand_lms)

            # Save as numpy file
            np.save(os.path.join(label_dir, f'{count}.npy'), landmarks)
            dataset.append(landmarks)
            data_labels.append(label)
            count += 1

        # Progress bar overlay
        progress = int((count / SAMPLES_PER_CLASS) * frame.shape[1])
        cv.rectangle(frame, (0, 0), (frame.shape[1], 50), (20, 20, 20), -1)
        cv.rectangle(frame, (0, 0), (progress, 50), (0, 200, 100), -1)
        cv.putText(frame, f"[ {label} ]  {count}/{SAMPLES_PER_CLASS}",
                   (20, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Data Collection', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"  ✅ '{label}' done — {count} samples saved")

cap.release()
cv.destroyAllWindows()

# ── Save full dataset as pickle ─────────────────────────────────────────────────
print("\n  💾 Saving full dataset...")
with open(os.path.join(DATA_DIR, 'dataset.pkl'), 'wb') as f:
    pickle.dump({'data': dataset, 'labels': data_labels}, f)

print(f"  ✅ Dataset saved → {DATA_DIR}/dataset.pkl")
print(f"  📊 Total samples collected: {len(dataset)}")
print("\n  Next step: run  python train_model.py  to train the model!")