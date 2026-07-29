import cv2
import numpy as np


# --------------------------
# Load Face Detector
# --------------------------
def load_face_detector():

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    return detector

# --------------------------
# Detect Face
# --------------------------
def detect_face(image, detector):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40)
    )

    return faces
# --------------------------
# Crop Face
# --------------------------
def crop_face(image, faces):

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    face = image[y:y+h, x:x+w]

    return face
# --------------------------
# Prepare Image
# --------------------------
def prepare_image(face):

    face = cv2.resize(face, (224, 224))

    img_array = np.array(
        face,
        dtype=np.float32
    )

    img_array = np.expand_dims(img_array, axis=0)

    # EfficientNetB0 built-in rescaling expects 0-255 inputs
    # img_array = img_array / 255.0

    return img_array
# --------------------------
# Predict Emotion
# --------------------------
def predict_emotion(model, img_array):

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_class = np.argmax(predictions)

    confidence = np.max(predictions) * 100

    return (
        predictions,
        predicted_class,
        confidence
    )