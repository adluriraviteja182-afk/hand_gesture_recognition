"""
Data Collection Script
=======================
Collect hand gesture samples using your webcam.
Creates a labeled dataset for training a custom ML model.

Usage:
    python collect_data.py --gesture "thumbs_up" --samples 200
    python collect_data.py --gesture "peace" --samples 200
    ... (repeat for each gesture class)
"""

import cv2
import os
import time
import numpy as np
import mediapipe as mp
import json
import argparse
from pathlib import Path

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Import feature extractor from main module
import sys
sys.path.insert(0, os.path.dirname(__file__))
from gesture_recognizer import extract_landmarks

# ─── Gesture Definitions ──────────────────────────────────────────────────────

GESTURE_MAP = {
    "fist":            0,
    "open_palm":       1,
    "pointing_up":     2,
    "peace":           3,
    "thumbs_up":       4,
    "thumbs_down":     5,
    "call_me":         6,
    "stop":            7,
    "fingers_crossed": 8,
    "ok":              9,
}


# ─── Collector ────────────────────────────────────────────────────────────────

class DataCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Load existing dataset if present
        self.dataset_path = self.output_dir / "dataset.json"
        if self.dataset_path.exists():
            with open(self.dataset_path) as f:
                self.dataset = json.load(f)
            print(f"[INFO] Loaded existing dataset: {len(self.dataset['X'])} samples")
        else:
            self.dataset = {"X": [], "y": [], "gesture_map": GESTURE_MAP}

    def collect(self, gesture_name, n_samples=200, camera_id=0):
        if gesture_name not in GESTURE_MAP:
            print(f"[ERROR] Unknown gesture: {gesture_name}")
            print(f"        Valid options: {list(GESTURE_MAP.keys())}")
            return

        label = GESTURE_MAP[gesture_name]
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        collected = 0
        recording = False
        countdown = 3

        print(f"\n[INFO] Collecting '{gesture_name}' (label={label})")
        print(f"[INFO] Target: {n_samples} samples")
        print("[INFO] Press SPACE to start collecting, 'q' to quit\n")

        last_countdown = time.time()

        while collected < n_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            hand_detected = False
            if results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                hand_detected = True

                if recording:
                    features = extract_landmarks(hand_lm, w, h)
                    self.dataset["X"].append(features.tolist())
                    self.dataset["y"].append(label)
                    collected += 1

            # UI overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 40), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            status_color = (0, 255, 100) if recording else (200, 200, 0)
            status = "● RECORDING" if recording else "○ PAUSED"
            cv2.putText(frame, f"Gesture: {gesture_name.upper()}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"{status}  |  {collected}/{n_samples}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            if not hand_detected:
                cv2.putText(frame, "⚠ No hand detected", (w//2 - 120, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)

            # Progress bar
            pct = collected / n_samples
            cv2.rectangle(frame, (0, h - 12), (w, h), (40, 40, 40), -1)
            cv2.rectangle(frame, (0, h - 12), (int(w * pct), h), (0, 200, 100), -1)

            cv2.imshow("Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                recording = not recording
                print(f"[INFO] Recording {'started' if recording else 'paused'}")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save dataset
        self.dataset["gesture_map"] = GESTURE_MAP
        with open(self.dataset_path, "w") as f:
            json.dump(self.dataset, f)

        print(f"\n[INFO] Collected {collected} samples for '{gesture_name}'")
        print(f"[INFO] Total dataset size: {len(self.dataset['X'])} samples")
        print(f"[INFO] Saved to {self.dataset_path}")

    def summary(self):
        """Print dataset summary."""
        from collections import Counter
        if not self.dataset["X"]:
            print("[INFO] Dataset is empty.")
            return

        counts = Counter(self.dataset["y"])
        inv_map = {v: k for k, v in GESTURE_MAP.items()}

        print("\n📊 Dataset Summary")
        print("─" * 40)
        for label, count in sorted(counts.items()):
            name = inv_map.get(label, f"class_{label}")
            bar = "█" * (count // 5)
            print(f"  {name:<20} {count:>4} samples  {bar}")
        print(f"\n  Total: {len(self.dataset['X'])} samples")
        print("─" * 40)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect gesture training data")
    parser.add_argument("--gesture", type=str, help="Gesture name to collect")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--summary", action="store_true", help="Show dataset stats")
    args = parser.parse_args()

    collector = DataCollector()

    if args.summary:
        collector.summary()
    elif args.gesture:
        collector.collect(args.gesture, n_samples=args.samples, camera_id=args.camera)
    else:
        print("Available gestures:", list(GESTURE_MAP.keys()))
        print("\nUsage: python collect_data.py --gesture thumbs_up --samples 200")
