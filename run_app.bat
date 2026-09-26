@echo off
title Serene Earth AI - Multimodal Emotion Recognition
echo ======================================================================
echo  Starting Serene Earth AI Emotion Recognition Server...
echo ======================================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" multimodal_emotion_detection.py
) else (
    python multimodal_emotion_detection.py
)
pause
