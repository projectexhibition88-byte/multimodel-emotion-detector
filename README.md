# EmotionX: Real-Time Multimodal Emotion Recognition System

An end-to-end, high-performance real-time Multimodal Emotion Recognition platform powered by **PyTorch**, **Flask**, and an interactive **React Dashboard**.

---

## Overview

**EmotionX** is an advanced artificial intelligence system capable of recognizing human emotional states with high precision by synchronously combining three distinct modalities:
1. **Visual Modality (Facial Expressions)**: Real-time high-FPS facial landmark localization (YuNet) paired with a deep **ResNet-18** feature extractor.
2. **Acoustic Modality (Speech Prosody and Tone)**: Audio silence trimming and pre-emphasis followed by Mel-spectrogram and MFCC feature extraction with a **Deep CRNN** (1D-CNN, Bi-GRU, and Self-Attention).
3. **Linguistic Modality (Text Semantics and NLP)**: Word embedding sequences analyzed by a **Bidirectional LSTM** with dense highway connections.
4. **Multimodal Late Fusion**: A deep **Adaptive Tensor Attention Fusion** neural network that synthesizes all three channels with dynamic modality dropout gating to resolve ambiguity and boost accuracy even when sensors fail.

The system categorizes emotional states into seven core classes:
Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

---

## System Features
- **Real-Time Inference**: Operates locally through a Web Dashboard.
- **Microservice Architecture**: Python Flask Backend combined with an HTML/React Frontend via REST API.
- **High-Fidelity Noise Cancellation**: Real-time silence trimming and high-frequency noise gating for audio processing.
- **Robust Modality Dropout**: System dynamically adapts using cross-modal attention if audio, video, or text is unavailable.

---

## Setup and Installation

**Prerequisites:** Python 3.9+ with an NVIDIA GPU (CUDA) recommended for fast inference.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Multimodal Server:**
   ```bash
   python multimodal_emotion_detection.py
   ```
   Or run the batch script on Windows:
   ```cmd
   run_app.bat
   ```
3. **Open the Dashboard:** Navigate to `http://127.0.0.1:5000` in your web browser.

---

## Quantitative Results & Benchmarks

The system has been rigorously evaluated on benchmark datasets to quantify both prediction accuracy and real-time inference latency.

### 1. Multimodal Fusion Model Accuracy
- **Synthetic Tri-Modal Dataset**: **98.63%** Test-Set Accuracy 
- **MELD (Multimodal EmotionLines Dataset) Benchmark**: **42.38%** Test-Set Accuracy (a highly challenging real-world dialogue dataset)

*Detailed classification reports, confusion matrices, and training curves are available in the `models/results/` directory.*

### 2. Real-Time Performance (FPS & Latency)
End-to-end multimodal inference (processing video frame, audio chunk, and text concurrently through all three deep networks and the fusion engine) was benchmarked on an NVIDIA RTX 5050 Laptop GPU:
- **Average Inference Latency**: **~54.6 ms** per cycle
- **Average Throughput**: **~18.3 FPS** 

This latency comfortably exceeds the requirement for real-time responsiveness (>10 FPS) in human-computer interaction applications.

---

## Repository Structure

- `multimodal_emotion_detection.py`: The main Application entry point & Flask API.
- `models/`: Directory containing pre-trained PyTorch/ONNX model files.
- `training_scripts/`: Scripts used for training high-accuracy models and evaluating the models.
- `static/` & `templates/`: Assets and views for the React Dashboard.

*(Note: Large raw datasets are omitted from version control.)*
