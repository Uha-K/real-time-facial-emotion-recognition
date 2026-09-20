import cv2
import time
import torch
import numpy as np

from PIL import Image
from torchvision import transforms


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = "emotion_resnet50_weighted_traced.pt"

EMOTION_CLASSES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================================================
# Load trained TorchScript model
# =========================================================

model = torch.jit.load(
    MODEL_PATH,
    map_location=device
)

model.eval()

print("Emotion recognition model loaded successfully.")


# =========================================================
# Preprocessing
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# OpenCV face detector
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError(
        "Unable to load OpenCV face detector."
    )


# =========================================================
# Open webcam
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Unable to open webcam."
    )


# =========================================================
# Performance tracking
# =========================================================

fps_history = []
inference_history = []

previous_time = time.perf_counter()

print("\nWebcam started.")
print("Press Q to exit.\n")


# =========================================================
# Main application loop
# =========================================================

while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read webcam frame.")
        break

    # -----------------------------------------------------
    # Face detection
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )


    # -----------------------------------------------------
    # Emotion recognition for each detected face
    # -----------------------------------------------------

    for x, y, w, h in faces:

        face = frame[
            y:y+h,
            x:x+w
        ]

        if face.size == 0:
            continue


        # OpenCV uses BGR.
        # PyTorch/PIL pipeline expects RGB.

        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            face_rgb
        )


        input_tensor = transform(
            pil_image
        )

        input_tensor = input_tensor.unsqueeze(
            0
        ).to(
            device
        )


        # -------------------------------------------------
        # Model inference
        # -------------------------------------------------

        inference_start = time.perf_counter()

        with torch.no_grad():

            output = model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted = probabilities.max(
                dim=1
            )


        if device.type == "cuda":
            torch.cuda.synchronize()


        inference_end = time.perf_counter()


        inference_ms = (
            inference_end
            - inference_start
        ) * 1000


        inference_history.append(
            inference_ms
        )


        predicted_index = predicted.item()

        predicted_emotion = (
            EMOTION_CLASSES[
                predicted_index
            ]
        )

        confidence_value = (
            confidence.item()
        )


        # -------------------------------------------------
        # Display prediction
        # -------------------------------------------------

        label_text = (
            f"{predicted_emotion}: "
            f"{confidence_value:.2f}"
        )


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            label_text,
            (
                x,
                max(y - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # =====================================================
    # End-to-end FPS
    # =====================================================

    current_time = time.perf_counter()

    frame_time = (
        current_time
        - previous_time
    )

    previous_time = current_time


    if frame_time > 0:

        fps = (
            1.0 / frame_time
        )

        fps_history.append(
            fps
        )

    else:
        fps = 0


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # =====================================================
    # Display application
    # =====================================================

    cv2.imshow(
        "Real-Time Facial Emotion Recognition",
        frame
    )


    key = cv2.waitKey(
        1
    ) & 0xFF


    if key == ord("q"):
        break


# =========================================================
# Cleanup
# =========================================================

camera.release()

cv2.destroyAllWindows()


# =========================================================
# Final performance results
# =========================================================

print("\n===== Performance Results =====")


if len(fps_history) > 0:

    print(
        "Mean end-to-end FPS:",
        round(
            np.mean(fps_history),
            2
        )
    )

    print(
        "Median end-to-end FPS:",
        round(
            np.median(fps_history),
            2
        )
    )


if len(inference_history) > 0:

    print(
        "Mean model latency:",
        round(
            np.mean(inference_history),
            2
        ),
        "ms"
    )

    print(
        "Median model latency:",
        round(
            np.median(inference_history),
            2
        ),
        "ms"
    )