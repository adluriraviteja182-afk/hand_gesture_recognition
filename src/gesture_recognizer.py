"""
Hand Gesture Recognition System
================================
Uses MediaPipe for hand landmark detection and a trained classifier
for gesture classification. Supports both static images and live webcam.
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from collections import deque


# ─── MediaPipe Setup ──────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ─── Gesture Labels ───────────────────────────────────────────────────────────

GESTURE_LABELS = {
    0: "✊ Fist",
    1: "✋ Open Palm",
    2: "☝️ Pointing Up",
    3: "✌️ Peace / Victory",
    4: "👍 Thumbs Up",
    5: "👎 Thumbs Down",
    6: "🤙 Call Me",
    7: "🖐️ Stop",
    8: "🤞 Fingers Crossed",
    9: "👌 OK Sign",
}

GESTURE_COLORS = {
    0: (0, 80, 200),
    1: (0, 180, 100),
    2: (200, 120, 0),
    3: (160, 0, 200),
    4: (0, 200, 200),
    5: (200, 0, 80),
    6: (100, 200, 0),
    7: (200, 200, 0),
    8: (0, 100, 255),
    9: (255, 100, 0),
}


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_landmarks(hand_landmarks, image_width, image_height):
    """
    Extract normalized landmark coordinates from MediaPipe result.
    Returns a flattened array of (x, y, z) for all 21 landmarks — 63 features.
    Also computes distances between key points for extra robustness.
    """
    landmarks = []
    raw = []

    for lm in hand_landmarks.landmark:
        raw.append((lm.x, lm.y, lm.z))
        landmarks.extend([lm.x, lm.y, lm.z])

    # Normalize relative to wrist (landmark 0)
    wrist = np.array(raw[0])
    middle_mcp = np.array(raw[9])
    scale = np.linalg.norm(middle_mcp - wrist) + 1e-6

    normalized = []
    for pt in raw:
        rel = (np.array(pt) - wrist) / scale
        normalized.extend(rel.tolist())

    # Pairwise distances between fingertips (4, 8, 12, 16, 20) and wrist
    fingertips = [4, 8, 12, 16, 20]
    distances = []
    for tip in fingertips:
        d = np.linalg.norm(np.array(raw[tip]) - wrist)
        distances.append(d / scale)

    # Finger bend angles (MCP → PIP → DIP)
    finger_joints = [
        (1, 2, 3),   # Thumb
        (5, 6, 7),   # Index
        (9, 10, 11), # Middle
        (13, 14, 15),# Ring
        (17, 18, 19),# Pinky
    ]
    angles = []
    for a_idx, b_idx, c_idx in finger_joints:
        a = np.array(raw[a_idx])
        b = np.array(raw[b_idx])
        c = np.array(raw[c_idx])
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angles.append(np.clip(cosine, -1, 1))

    return np.array(normalized + distances + angles, dtype=np.float32)


# ─── Rule-Based Classifier (No Training Required) ─────────────────────────────

def rule_based_classify(hand_landmarks):
    """
    Lightweight rule-based classifier using finger extension states.
    Works immediately without any training data.
    """
    lm = hand_landmarks.landmark

    def is_finger_extended(tip, pip, wrist_y):
        """Check if a finger is extended (tip above pip in image coords)."""
        return lm[tip].y < lm[pip].y

    def is_thumb_extended(side="right"):
        if side == "right":
            return lm[4].x > lm[3].x
        else:
            return lm[4].x < lm[3].x

    wrist_y = lm[0].y

    thumb_up = is_thumb_extended()
    index_up = is_finger_extended(8, 6, wrist_y)
    middle_up = is_finger_extended(12, 10, wrist_y)
    ring_up = is_finger_extended(16, 14, wrist_y)
    pinky_up = is_finger_extended(20, 18, wrist_y)

    fingers = [index_up, middle_up, ring_up, pinky_up]
    count = sum(fingers)

    # Classification logic
    if not any(fingers) and not thumb_up:
        return 0  # Fist

    if all(fingers) and thumb_up:
        return 1  # Open Palm

    if all(fingers) and not thumb_up:
        return 7  # Stop

    if index_up and not middle_up and not ring_up and not pinky_up:
        return 2  # Pointing Up

    if index_up and middle_up and not ring_up and not pinky_up:
        return 3  # Peace

    if thumb_up and not any(fingers):
        if lm[4].y < lm[3].y:
            return 4  # Thumbs Up
        else:
            return 5  # Thumbs Down

    if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
        return 6  # Call Me

    if index_up and middle_up and ring_up and not pinky_up:
        return 8  # Fingers Crossed (approximation)

    if not index_up and not middle_up and not ring_up and not pinky_up and thumb_up:
        return 9  # OK (simplified)

    return -1  # Unknown


# ─── Main Recognizer Class ────────────────────────────────────────────────────

class GestureRecognizer:
    def __init__(self, model_path=None, mode="rule_based",
                 min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Args:
            model_path: Path to a trained .pkl model (optional)
            mode: 'rule_based' or 'ml_model'
            min_detection_confidence: MediaPipe detection threshold
            min_tracking_confidence: MediaPipe tracking threshold
        """
        self.mode = mode
        self.model = None

        if model_path and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.mode = "ml_model"
            print(f"[INFO] Loaded ML model from {model_path}")
        else:
            print("[INFO] Using rule-based classifier (no model file found)")
            self.mode = "rule_based"

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Smoothing buffer per hand
        self.prediction_buffer = [deque(maxlen=5), deque(maxlen=5)]

    def predict(self, hand_landmarks):
        """Predict gesture for a single hand."""
        if self.mode == "ml_model" and self.model is not None:
            features = extract_landmarks(hand_landmarks, 1, 1)
            pred = self.model.predict([features])[0]
            return int(pred)
        else:
            return rule_based_classify(hand_landmarks)

    def smooth_prediction(self, hand_idx, pred):
        """Apply majority-vote smoothing over last N frames."""
        if pred == -1:
            return pred
        self.prediction_buffer[hand_idx].append(pred)
        if len(self.prediction_buffer[hand_idx]) == 0:
            return pred
        from collections import Counter
        return Counter(self.prediction_buffer[hand_idx]).most_common(1)[0][0]

    def process_frame(self, frame):
        """
        Process a single BGR frame.
        Returns annotated frame + list of (gesture_id, label, bbox) per hand.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        detections = []

        if results.multi_hand_landmarks:
            for i, (hand_lm, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Bounding box
                xs = [lm.x * w for lm in hand_lm.landmark]
                ys = [lm.y * h for lm in hand_lm.landmark]
                x1, y1 = int(min(xs)) - 20, int(min(ys)) - 20
                x2, y2 = int(max(xs)) + 20, int(max(ys)) + 20
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Predict & smooth
                pred = self.predict(hand_lm)
                smoothed = self.smooth_prediction(i % 2, pred)
                label = GESTURE_LABELS.get(smoothed, "❓ Unknown")
                color = GESTURE_COLORS.get(smoothed, (180, 180, 180))

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label background
                hand_type = handedness.classification[0].label
                text = f"{label} ({hand_type})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
                cv2.putText(frame, text, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                detections.append((smoothed, label, (x1, y1, x2, y2)))

        return frame, detections

    def run_webcam(self, camera_id=0):
        """Launch live webcam gesture recognition."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam. Check camera_id.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n[INFO] Webcam started. Press 'q' to quit, 's' to save screenshot.\n")

        fps_buffer = deque(maxlen=30)
        import time

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror
            annotated, detections = self.process_frame(frame)

            # FPS overlay
            fps_buffer.append(1.0 / (time.time() - t0 + 1e-6))
            fps = np.mean(fps_buffer)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Mode indicator
            mode_txt = f"Mode: {self.mode.replace('_', ' ').title()}"
            cv2.putText(annotated, mode_txt, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 0), 1)

            # Instruction
            cv2.putText(annotated, "Press 'q' to quit", (10, annotated.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("Hand Gesture Recognition", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, annotated)
                print(f"[INFO] Saved {fname}")

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam closed.")

    def process_image(self, image_path, output_path=None):
        """Run gesture recognition on a static image."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return None, []

        annotated, detections = self.process_frame(frame)

        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"[INFO] Saved annotated image to {output_path}")
        else:
            cv2.imshow("Gesture Recognition Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        for pred, label, bbox in detections:
            print(f"  → Detected: {label} at {bbox}")

        return annotated, detections


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hand Gesture Recognition")
    parser.add_argument("--mode", choices=["webcam", "image"], default="webcam")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, help="Path to save output image")
    parser.add_argument("--model", type=str, default=None, help="Path to .pkl model")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    args = parser.parse_args()

    recognizer = GestureRecognizer(model_path=args.model)

    if args.mode == "webcam":
        recognizer.run_webcam(camera_id=args.camera)
    elif args.mode == "image":
        if not args.image:
            print("[ERROR] Provide --image path when using image mode.")
        else:
            recognizer.process_image(args.image, output_path=args.output)
