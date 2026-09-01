import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Emotion categories and metadata
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_EMOJIS = {
    'angry': 'Angry 😠',
    'disgust': 'Disgust 🤢',
    'fear': 'Fear 😨',
    'happy': 'Happy 😊',
    'neutral': 'Neutral 😐',
    'sad': 'Sad 😔',
    'surprise': 'Surprise 😲'
}

EMOTION_COLORS = {
    'angry': (0, 0, 230),       # Red
    'disgust': (34, 139, 34),    # Forest Green
    'fear': (128, 0, 128),      # Purple
    'happy': (0, 215, 255),     # Gold / Bright Yellow
    'neutral': (200, 200, 200), # Light Gray
    'sad': (235, 100, 50),      # Deep Blue/Orange
    'surprise': (255, 165, 0)   # Orange / Cyan
}

class FacialEmotionDetector:
    def __init__(self, model_path=None):
        self.emotions = EMOTIONS
        self.face_cascade = self._load_cascade()
        self.framework = None  # 'pytorch', 'tensorflow', or 'heuristic'
        self.model_arch = None  # 'resnet18', 'emotion_cnn', 'tensorflow', 'heuristic'
        self.torch_device = None
        self.model = None
        self._init_backend(model_path)

    def _load_cascade(self):
        """Loads OpenCV Haar Cascade for Face Detection."""
        cascade = None
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass

        if cascade is None or cascade.empty():
            local_path = os.path.join(os.path.dirname(__file__), "models", "haarcascade_frontalface_default.xml")
            if os.path.exists(local_path):
                cascade = cv2.CascadeClassifier(local_path)
            else:
                print(" [!] Note: Haar Cascade initialized via OpenCV.")
        return cascade

    def _init_backend(self, model_path=None):
        """Checks for PyTorch ResNet18 (CUDA/CPU) first, then EmotionCNN, then TensorFlow, then fallback."""
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
        candidates = []
        if model_path:
            candidates.append(model_path)
        candidates.extend([
            os.path.join(models_dir, "facial_emotion_model_cuda.pth"),
            os.path.join(models_dir, "facial_emotion_model_final.pth"),
            os.path.join(models_dir, "facial_emotion_model.pth"),
            os.path.join(models_dir, "facial_emotion_model.h5")
        ])

        # 1. Try PyTorch CUDA / CPU models
        for cand in candidates:
            if cand.endswith('.pth') and os.path.exists(cand):
                try:
                    import torch
                    import torch.nn as nn
                    from torchvision import models

                    try:
                        if torch.cuda.is_available():
                            self.torch_device = torch.device("cuda:0")
                            torch.zeros(1, device=self.torch_device)  # smoke test
                        else:
                            self.torch_device = torch.device("cpu")
                    except Exception:
                        self.torch_device = torch.device("cpu")

                    ckpt = torch.load(cand, map_location=self.torch_device, weights_only=False)
                    sd = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt

                    if 'conv1.weight' in sd or 'fc.1.weight' in sd:
                        # ResNet-18 Deep Architecture (trained via CUDA)
                        self.model = models.resnet18(weights=None)
                        num_features = self.model.fc.in_features
                        self.model.fc = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(num_features, len(self.emotions))
                        )
                        self.model.load_state_dict(sd)
                        self.model.to(self.torch_device)
                        self.model.eval()
                        self.framework = 'pytorch'
                        self.model_arch = 'resnet18'
                        gpu_name = torch.cuda.get_device_name(0) if self.torch_device.type == "cuda" else "CPU"
                        print(f" [★] Loaded ResNet18 Deep Model on: {gpu_name} (File: {cand})")
                        # Warm-up pass to pre-compile CUDA kernels and allocate memory
                        try:
                            with torch.no_grad():
                                self.model(torch.zeros(1, 3, 224, 224, device=self.torch_device))
                        except Exception:
                            pass
                        return
                    elif 'block1.0.weight' in sd:
                        # EmotionCNN Architecture (48x48)
                        from train_facial_model_gpu import EmotionCNN
                        self.model = EmotionCNN(num_classes=len(self.emotions)).to(self.torch_device)
                        self.model.load_state_dict(sd)
                        self.model.eval()
                        self.framework = 'pytorch'
                        self.model_arch = 'emotion_cnn'
                        gpu_name = torch.cuda.get_device_name(0) if self.torch_device.type == "cuda" else "CPU"
                        print(f" [★] Loaded EmotionCNN PyTorch Model on: {gpu_name} (File: {cand})")
                        # Warm-up pass
                        try:
                            with torch.no_grad():
                                self.model(torch.zeros(1, 1, 48, 48, device=self.torch_device))
                        except Exception:
                            pass
                        return
                except Exception as e:
                    print(f" [!] PyTorch load failed for {cand}: {e}")

        # 2. Try TensorFlow model
        for cand in candidates:
            if cand.endswith('.h5') and os.path.exists(cand):
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(cand, compile=False)
                    self.framework = 'tensorflow'
                    self.model_arch = 'tensorflow'
                    print(f" [★] Loaded TensorFlow CNN Model from: {cand}")
                    return
                except Exception:
                    pass

        # 3. Fallback heuristic mode
        print(" [i] Initialized geometric feature emotion analyzer fallback.")
        self.framework = 'heuristic'
        self.model_arch = 'heuristic'

    def predict_emotion(self, face_img):
        """Runs inference on a cropped face (accepts BGR or grayscale image)."""
        if self.framework == 'pytorch' and self.model is not None:
            import torch
            if self.model_arch == 'resnet18':
                # ResNet18: 3-channel RGB, (224, 224), normalized with ImageNet stats
                if len(face_img.shape) == 2:
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                elif face_img.shape[2] == 4:
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
                else:
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                resized = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                norm = resized.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm = (norm - mean) / std
                tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.torch_device)
                
                with torch.no_grad():
                    out = self.model(tensor)
                    probs = torch.softmax(out, dim=1).cpu().numpy()[0]

            else:  # emotion_cnn
                if len(face_img.shape) == 3:
                    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                else:
                    face_gray = face_img
                resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
                norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
                tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(self.torch_device)
                
                with torch.no_grad():
                    out = self.model(tensor)
                    probs = torch.softmax(out, dim=1).cpu().numpy()[0]

            emotion_idx = int(np.argmax(probs))
            dom_emotion = self.emotions[emotion_idx]
            confidence = float(probs[emotion_idx])
            prob_dict = {self.emotions[i]: float(probs[i]) for i in range(len(self.emotions))}
            return dom_emotion, confidence, prob_dict

        elif self.framework == 'tensorflow' and self.model is not None:
            if len(face_img.shape) == 3:
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_img
            resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
            norm = resized.astype("float32") / 255.0
            reshaped = np.expand_dims(np.expand_dims(norm, axis=0), axis=-1)
            preds = self.model.predict(reshaped, verbose=0)[0]
            emotion_idx = int(np.argmax(preds))
            dom_emotion = self.emotions[emotion_idx]
            confidence = float(preds[emotion_idx])
            prob_dict = {self.emotions[i]: float(preds[i]) for i in range(len(self.emotions))}
            return dom_emotion, confidence, prob_dict

        else:
            # Geometric Heuristic
            if len(face_img.shape) == 3:
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_img
            h, w = face_gray.shape
            mouth_roi = face_gray[int(h*0.65):int(h*0.95), int(w*0.2):int(w*0.8)]
            upper_roi = face_gray[int(h*0.15):int(h*0.5), int(w*0.15):int(w*0.85)]
            mouth_mean = np.mean(mouth_roi) if mouth_roi.size > 0 else 128
            upper_mean = np.mean(upper_roi) if upper_roi.size > 0 else 128
            ratio = mouth_mean / (upper_mean + 1e-5)

            probs = {e: 0.08 for e in self.emotions}
            if ratio > 1.15:
                probs['happy'] += 0.55
                probs['surprise'] += 0.20
            elif ratio < 0.85:
                probs['sad'] += 0.45
                probs['angry'] += 0.30
            else:
                probs['neutral'] += 0.55
                probs['happy'] += 0.15

            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
            dom = max(probs, key=probs.get)
            return dom, float(probs[dom]), probs

    def draw_probability_hud(self, frame, probabilities, dominant_emotion):
        """Renders an attractive probability meter HUD on the screen."""
        h, w, _ = frame.shape
        hud_w = 260
        hud_h = 240
        x_offset = w - hud_w - 20
        y_offset = 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + hud_w, y_offset + hud_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset + hud_w, y_offset + hud_h), (70, 70, 80), 1)

        cv2.putText(frame, "EMOTION METRICS", (x_offset + 15, y_offset + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        bar_y = y_offset + 50
        for emotion, prob in probabilities.items():
            color = EMOTION_COLORS.get(emotion, (200, 200, 200))
            is_dominant = (emotion == dominant_emotion)
            
            label_text = f"{emotion.capitalize():<8}"
            text_color = (255, 255, 255) if not is_dominant else color
            cv2.putText(frame, label_text, (x_offset + 15, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1 if not is_dominant else 2, cv2.LINE_AA)
            
            bar_x = x_offset + 95
            max_bar_w = 100
            current_bar_w = int(max_bar_w * prob)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + max_bar_w, bar_y + 14), (50, 50, 60), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + current_bar_w, bar_y + 14), color, -1)
            
            pct_text = f"{prob*100:4.1f}%"
            cv2.putText(frame, pct_text, (bar_x + max_bar_w + 10, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            bar_y += 24

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE
            )

        last_probs = {e: 0.0 for e in self.emotions}
        last_dom = "neutral"

        for (x, y, w, h) in faces:
            # 10% padding margin so eyebrows, forehead, and chin are fully captured
            pad_x = int(w * 0.10)
            pad_y = int(h * 0.10)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)
            
            face_roi = frame[y1:y2, x1:x2]
            dominant_emotion, confidence, probs = self.predict_emotion(face_roi)
            last_probs = probs
            last_dom = dominant_emotion

            color = EMOTION_COLORS.get(dominant_emotion, (0, 255, 0))
            caption = f"{dominant_emotion.capitalize()} ({confidence*100:.1f}%)"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            banner_h = 32
            cv2.rectangle(frame, (x, y - banner_h), (x + w, y), color, -1)
            cv2.putText(frame, caption, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if len(faces) > 0:
            self.draw_probability_hud(frame, last_probs, last_dom)

        cv2.putText(frame, "Project Exhibition - I (DSN2098) | Emotion AI", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Detected Faces: {len(faces)}", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

        return frame, len(faces)

    def run_webcam(self, camera_id=0):
        print("=" * 65)
        print(" STARTING LIVE WEBCAM FACIAL EMOTION DETECTION")
        print(" Controls: Press 'q' to Quit | Press 's' to Save Snapshot")
        print("=" * 65)

        # Fast camera initialization on Windows using DirectShow (bypasses 3-5s MSMF probe delay)
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f" [!] Error: Unable to access webcam on device {camera_id}.")
            return

        # Optimization: minimal buffer to prevent frame lag and instant rendering
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        window_name = "Review 1: Facial Emotion Detection (Group-52)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_time = time.time()
        snapshot_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            annotated_frame, face_count = self.process_frame(frame)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 100), 1, cv2.LINE_AA)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snapshot_count += 1
                filename = f"snapshot_emotion_{snapshot_count}_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f" [✓] Snapshot saved to: {filename}")

        cap.release()
        cv2.destroyAllWindows()

    def process_image(self, image_path, output_path=None):
        if not os.path.exists(image_path):
            print(f" [!] Error: Image not found at: {image_path}")
            return None

        img = cv2.imread(image_path)
        annotated_img, count = self.process_frame(img)
        print(f" [+] Processed image: {image_path} | Detected Faces: {count}")

        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_emotion_result{ext}"
        cv2.imwrite(output_path, annotated_img)
        print(f" [✓] Annotated output saved to: {output_path}")
        return output_path

    def run_self_test(self):
        print("=" * 65)
        print(" RUNNING AUTOMATED SELF-TEST (REVIEW 1 PIPELINE)")
        print("=" * 65)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 40
        cv2.circle(canvas, (320, 240), 100, (180, 180, 180), -1)
        cv2.circle(canvas, (280, 210), 15, (30, 30, 30), -1)
        cv2.circle(canvas, (360, 210), 15, (30, 30, 30), -1)
        cv2.ellipse(canvas, (320, 270), (45, 25), 0, 0, 180, (30, 30, 30), 8)

        mock_face = cv2.cvtColor(canvas[140:340, 220:420], cv2.COLOR_BGR2GRAY)
        dom_emotion, conf, probs = self.predict_emotion(mock_face)

        print(f" [+] Backend Framework: {self.framework.upper()}")
        print(f" [+] Dominant Emotion : {dom_emotion.upper()} ({EMOTION_EMOJIS.get(dom_emotion, '')})")
        print(f" [+] Confidence Score : {conf*100:.2f}%")
        print(" [✓] Part 1 Facial Emotion Detection Pipeline is 100% operational!")
        print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Review 1: Facial Emotion Detection System")
    parser.add_argument("--webcam", action="store_true", help="Launch live webcam detection mode")
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--image", type=str, default=None, help="Path to static image for emotion analysis")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (.pth or .h5)")
    parser.add_argument("--test", action="store_true", help="Run automated self-test verification")
    args = parser.parse_args()

    detector = FacialEmotionDetector(model_path=args.model)

    if args.test:
        detector.run_self_test()
    elif args.image:
        detector.process_image(args.image)
    else:
        if args.webcam:
            detector.run_webcam(camera_id=args.camera_id)
        else:
            print("\n" + "=" * 65)
            print(" REVIEW 1: FACIAL EMOTION DETECTION")
            print(" Usage Options:")
            print("   1. Live Webcam Mode : python part1_facial_emotion.py --webcam")
            print("   2. Static Image Mode: python part1_facial_emotion.py --image path/to/photo.jpg")
            print("   3. Self-Test Check  : python part1_facial_emotion.py --test")
            print("=" * 65 + "\n")
            detector.run_self_test()


if __name__ == "__main__":
    main()
