# 🤟 ASL Sign Language Translator

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.x-orange?style=flat)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-RandomForest-red?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

A real-time American Sign Language (ASL) translator that uses your webcam to recognize hand gestures, convert them into text, and speak them out loud — built entirely with computer vision and machine learning.

> Built by **Hira Ishtiaq** — AI/ML Enthusiast  
> [github.com/hira-ishtiaq](https://github.com/hira-ishtiaq)

---

## 🎯 Project Overview

This project is an end-to-end machine learning pipeline that:

1. **Collects** hand landmark data from a webcam for all 26 ASL letters
2. **Trains** a Random Forest classifier on that data
3. **Translates** live hand gestures into text and speech in real time

The goal was to build something practically useful while diving deep into **computer vision**, **gesture recognition**, and **real-time ML inference** — all from scratch.

---

## 🧠 How It Works

### Step 1 — Hand Landmark Detection
Google's **MediaPipe** framework detects 21 key points on the hand per frame in real time — from the wrist to the tip of every finger. Each point is captured as an (x, y) coordinate, normalized relative to the wrist position so predictions work regardless of hand placement on screen.

This gives us **42 features per sample** (21 points × x and y).

### Step 2 — Gesture Classification
A **Random Forest Classifier** (200 decision trees) is trained on the collected landmark data. Each tree votes on which ASL letter the hand is showing, and the majority vote is the final prediction.

### Step 3 — Stability Buffer
To prevent jitter and accidental confirmations, the app uses a **frame buffer** — it collects the last 20 predictions and only confirms a letter if all 20 agree on the same letter. This makes the translator stable and accurate in real use.

### Step 4 — Text to Speech
Once a letter is confirmed, it is spoken out loud. Letters build into words, words build into sentences, and the full sentence can be spoken on demand.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Training Samples | 2,600 (100 per letter × 26 letters) |
| Test Accuracy | **99.81%** |
| Classes | 26 (A–Z) |
| Algorithm | Random Forest (200 estimators) |

---

## 🖐️ Gestures & Controls

| ASL Sign | Action |
|---|---|
| Hold any letter steady | Confirms the letter after 20 stable frames |
| **SPACE** | Completes the current word |
| **ENTER** | Speaks the full sentence out loud |
| **BACKSPACE** | Deletes the last letter |
| **C** | Clears everything |
| **Q** | Quit the app |

---

## 🗂️ Project Structure

```
asl-sign-language-translator/
├── collect_data.py      # Stage 1: collect hand landmark training data
├── train_model.py       # Stage 2: train the Random Forest classifier
├── app.py               # Stage 3: live real-time translator
├── model/
│   ├── asl_model.pkl        # trained model
│   └── accuracy_chart.png   # per-letter accuracy visualization
├── data/                # collected landmark samples (A-Z)
└── requirements.txt     # dependencies
```

---

## ⚙️ Setup & Installation

> Requires **Python 3.10**. MediaPipe does not support Python 3.12+.

### 1. Clone the repository
```bash
git clone https://github.com/hira-ishtiaq/asl-sign-language-translator.git
cd asl-sign-language-translator
```

### 2. Create virtual environment
```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install opencv-python mediapipe==0.10.5 numpy scikit-learn matplotlib pyttsx3
```

### 4. Collect your data (A–Z)
```bash
py -3.10 collect_data.py
```
Follow the on-screen instructions — make each ASL sign and press SPACE to record 100 samples per letter.

### 5. Train the model
```bash
py -3.10 train_model.py
```

### 6. Run the live translator
```bash
py -3.10 app.py
```

---

## 💡 What I Learned

- Building a **real-time computer vision pipeline** from scratch
- Using **MediaPipe** to extract and normalize hand landmark features
- **Training a custom ML classifier** on self-collected data
- Designing a **frame buffer voting system** for stable gesture recognition
- Integrating **text-to-speech** output into a live CV application
- End-to-end thinking: data collection → training → inference → output

---

## 🚀 Future Improvements

- [ ] Support for full ASL words (not just letters)
- [ ] Add PSL (Pakistan Sign Language) support
- [ ] Web app version using TensorFlow.js + WebRTC
- [ ] Two-hand gesture support
- [ ] Mobile app using Flutter + TFLite

---

## 🛠️ Tech Stack

- **Python 3.10**
- **MediaPipe** — hand landmark detection
- **OpenCV** — webcam feed and real-time rendering
- **Scikit-Learn** — Random Forest classifier
- **NumPy** — feature processing
- **pyttsx3** — text to speech

---

## 👩‍💻 Author

**Hira Ishtiaq** — AI/ML Enthusiast  
[github.com/hira-ishtiaq](https://github.com/hira-ishtiaq)

---

*MIT License — free to use, modify, and share*
