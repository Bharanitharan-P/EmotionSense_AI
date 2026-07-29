import cv2
import numpy as np
import tempfile
import os
from deepface import DeepFace

# -----------------------------
# Emotion Colors (BGR for OpenCV)
# -----------------------------
emotion_colors = {
    "angry":    (0, 0, 255),       # Red
    "disgust":  (139, 69, 19),     # Brown
    "fear":     (0, 140, 255),     # Orange
    "happy":    (0, 200, 50),      # Green
    "neutral":  (255, 200, 0),     # Cyan-Yellow
    "sad":      (255, 80, 0),      # Blue
    "surprise": (180, 60, 200)     # Purple
}

emotion_emojis = {
    "angry":    "ANGRY",
    "disgust":  "DISGUST",
    "fear":     "FEAR",
    "happy":    "HAPPY",
    "neutral":  "NEUTRAL",
    "sad":      "SAD",
    "surprise": "SURPRISE"
}

# -----------------------------
# Face Detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

print("✅ Face Detector Loaded!")
print("✅ DeepFace Model Ready (High Accuracy Mode)!")

# -----------------------------
# Open Webcam
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Cannot Open Webcam")
    exit()

print("📷 Webcam Started! Press Q to quit.")

# Analyze every N frames (for performance)
frame_count = 0
ANALYZE_EVERY = 10       # Run DeepFace every 10 frames

last_emotion = "neutral"
last_confidence = 0.0

# -----------------------------
# Webcam Loop
# -----------------------------
while True:

    success, frame = camera.read()

    if not success:
        print("❌ Failed to Read Frame")
        break

    frame_count += 1

    # Detect Faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:

        # Analyze emotion every ANALYZE_EVERY frames
        if frame_count % ANALYZE_EVERY == 0:
            try:
                face_img = frame[y:y+h, x:x+w]

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, face_img)

                result = DeepFace.analyze(
                    img_path=tmp_path,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True
                )

                os.unlink(tmp_path)

                if isinstance(result, list):
                    result = result[0]

                last_emotion = result["dominant_emotion"].lower()
                last_confidence = result["emotion"][last_emotion]

            except Exception as e:
                pass  # Keep last known emotion on error

        color = emotion_colors.get(last_emotion, (255, 255, 255))
        label = f"{last_emotion.upper()}: {last_confidence:.1f}%"

        # Draw Rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display Emotion Label
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2
        )

    # Show Webcam
    cv2.imshow("EmotionSense AI - DeepFace (Press Q to quit)", frame)

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()





