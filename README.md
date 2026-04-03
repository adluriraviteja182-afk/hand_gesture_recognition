# 🖐️ Hand Gesture Recognition System
### SkillCraft Technology — Task 04

A complete computer vision pipeline for recognizing hand gestures in real-time using **MediaPipe** for hand landmark detection and **ML classifiers** for gesture prediction.

---

## 📁 Project Structure

```
hand_gesture_recognition/
├── src/
│   ├── gesture_recognizer.py   ← Main recognizer (webcam + image mode)
│   ├── collect_data.py         ← Data collection tool
│   └── train_model.py          ← Model training (RF + MLP)
├── models/                     ← Saved .pkl model files
├── data/
│   └── raw/dataset.json        ← Collected landmark dataset
├── notebooks/
│   └── exploration.ipynb       ← EDA + training notebook
└── requirements.txt
```

---

## ⚡ Quick Start (No Training Required)

The system works **immediately** using a rule-based classifier — no data collection needed.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch webcam (rule-based mode)
python src/gesture_recognizer.py --mode webcam

# 3. Run on an image
python src/gesture_recognizer.py --mode image --image hand.jpg --output result.jpg
```

---

## 🧠 System Architecture

```
Input (Webcam / Image)
        ↓
  Frame Preprocessing
  (flip, BGR→RGB)
        ↓
  MediaPipe Hands
  (21 landmark detection)
        ↓
  Feature Extraction
  (normalized coords + distances + angles = 78 features)
        ↓
  ┌─────────────────────┐
  │  Rule-Based Mode    │  ← Works immediately, no training
  │  ML Model Mode      │  ← Higher accuracy after training
  └─────────────────────┘
        ↓
  Gesture Label + Bounding Box
        ↓
  Annotated Output
```

---

## 🎯 Supported Gestures

| ID | Gesture          | Emoji |
|----|------------------|-------|
| 0  | Fist             | ✊    |
| 1  | Open Palm        | ✋    |
| 2  | Pointing Up      | ☝️   |
| 3  | Peace / Victory  | ✌️   |
| 4  | Thumbs Up        | 👍    |
| 5  | Thumbs Down      | 👎    |
| 6  | Call Me          | 🤙    |
| 7  | Stop             | 🖐️   |
| 8  | Fingers Crossed  | 🤞    |
| 9  | OK Sign          | 👌    |

---

## 📊 Training a Custom ML Model (Optional but Recommended)

### Step 1: Collect Data

```bash
# Collect 200 samples per gesture (adjust --samples as needed)
python src/collect_data.py --gesture thumbs_up     --samples 200
python src/collect_data.py --gesture peace         --samples 200
python src/collect_data.py --gesture open_palm     --samples 200
python src/collect_data.py --gesture fist          --samples 200
python src/collect_data.py --gesture pointing_up   --samples 200
# ... repeat for each gesture

# Check dataset stats
python src/collect_data.py --summary
```

**During collection:** Press `SPACE` to start/pause recording, `q` to quit.

### Step 2: Train Model

```bash
# Random Forest (fast, usually >95% accuracy)
python src/train_model.py --model random_forest

# MLP Neural Network (slower, sometimes more accurate)
python src/train_model.py --model mlp
```

### Step 3: Run with Trained Model

```bash
python src/gesture_recognizer.py --model models/gesture_model.pkl
```

---

## 🔬 Features Used (78 total)

1. **Normalized Landmark Coordinates** (63 values) — x, y, z for all 21 MediaPipe hand landmarks, normalized relative to wrist position and scale
2. **Fingertip Distances** (5 values) — distance from each fingertip to wrist
3. **Finger Bend Angles** (5 values) — cosine of joint angle at MCP→PIP→DIP for each finger
4. **Wrist-Scale Normalization** — makes features invariant to hand size and distance from camera

---

## 🛠️ Technologies

| Component         | Tool                  |
|-------------------|-----------------------|
| Hand Detection    | MediaPipe Hands       |
| Image Processing  | OpenCV                |
| Feature Extraction| NumPy                 |
| ML Classifiers    | scikit-learn          |
| Notebooks         | Jupyter               |

---

## 💡 Tips for Best Results

- **Lighting**: Use good, consistent lighting. Avoid backlighting.
- **Background**: Plain background improves detection confidence.
- **Distance**: Keep hand 30–80 cm from camera.
- **Data variety**: When collecting, vary hand angle and lighting slightly.
- **Samples**: Aim for 200+ samples per gesture class for ML mode.

---

## 🚀 Extending the System

- **Sign Language**: Map gestures to ASL alphabet (A–Z)
- **Mouse Control**: Use `pyautogui` to move cursor based on index finger position
- **Smart Home**: Trigger GPIO/MQTT events on gesture detection
- **Gaming**: Map gestures to keyboard inputs with `pynput`
