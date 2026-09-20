# Real-Time Facial Emotion Recognition

This directory contains the OpenCV-based real-time inference application.

## Pipeline

Webcam
→ Face Detection
→ Face Crop
→ RGB Conversion
→ Resize to 224 × 224
→ ImageNet Normalisation
→ ResNet50
→ Softmax
→ Emotion Prediction
→ Display

## Run

Install the project dependencies and place the exported TorchScript model in
the required model location.

Then run:

python realtime_emotion_recognition.py

Press `q` to close the webcam application.

## Important

The GPU benchmark reported in the dissertation measures model-only inference
latency.

Complete webcam FPS also depends on:

- frame acquisition;
- face detection;
- preprocessing;
- inference;
- drawing;
- display.

Therefore, theoretical model inference throughput should not be reported as
end-to-end webcam FPS.
