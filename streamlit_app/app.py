import streamlit as st
import numpy as np
import cv2
import os
import tempfile
from deepface import DeepFace

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="😊",
    layout="centered"
)
# --------------------------
# Load Face Detector
# --------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# --------------------------
# Emotion Labels & Emojis
# --------------------------
emotion_emojis = {
    "angry":    "😡",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲"
}

emotion_colors = {
    "angry":    "#FF4B4B",
    "disgust":  "#7B5EA7",
    "fear":     "#FF8C00",
    "happy":    "#00CC44",
    "neutral":  "#4B8BF5",
    "sad":      "#0099FF",
    "surprise": "#FF69B4"
}

# --------------------------
# Title
# --------------------------
st.title("😊 EmotionSense AI")
st.success("✅ DeepFace Model Ready (High Accuracy Mode)")
st.success("✅ Face Detector Loaded!")
st.write("Upload an image to detect facial emotion.")

# --------------------------
# Upload Image
# --------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------
# Process Image
# --------------------------
if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --------------------------
    # Detect Face
    # --------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        st.error("❌ No face detected! Please upload a clear front-facing photo.")
        st.stop()

    # --------------------------
    # Crop Face
    # --------------------------
    x, y, w, h = faces[0]
    face_crop = rgb[y:y+h, x:x+w]

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            rgb,
            caption="📷 Uploaded Image",
            use_container_width=True
        )

    with col2:
        st.image(
            face_crop,
            caption="😀 Detected Face",
            use_container_width=True
        )

    st.success("✅ Face Ready for Prediction!")

    # --------------------------
    # DeepFace Prediction
    # --------------------------
    with st.spinner("🔍 Analyzing emotion..."):
        try:
            # Save temp image for DeepFace (needs file path)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, img)

            result = DeepFace.analyze(
                img_path=tmp_path,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            os.unlink(tmp_path)  # delete temp file

            # DeepFace returns list or dict
            if isinstance(result, list):
                result = result[0]

            emotion_scores = result["emotion"]   # dict: {"happy": 98.2, "sad": 0.5, ...}
            dominant_emotion = result["dominant_emotion"].lower()

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.stop()

    # --------------------------
    # Display Result
    # --------------------------
    confidence = emotion_scores[dominant_emotion]
    emoji = emotion_emojis.get(dominant_emotion, "🎭")
    color = emotion_colors.get(dominant_emotion, "#ffffff")

    st.markdown("## 🎯 Prediction Result")
    st.markdown(
        f"<div style='background-color:{color}22; border-left:6px solid {color}; "
        f"padding:16px; border-radius:8px; font-size:28px; font-weight:bold;'>"
        f"{emoji} {dominant_emotion.capitalize()}</div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Confidence")
    st.progress(float(confidence) / 100)
    st.write(f"**{confidence:.2f}%**")

    st.markdown("---")
    st.subheader("📊 Emotion Probabilities")

    # Sort emotions from highest to lowest
    sorted_emotions = sorted(
        emotion_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for emotion, score in sorted_emotions:
        em = emotion_emojis.get(emotion.lower(), "🎭")
        st.write(f"{em} {emotion.capitalize()}: {score:.2f}%")
        st.progress(float(score) / 100)

# --------------------------
# Footer
# --------------------------
st.markdown("---")

st.caption(
    "© 2026 P Bharanitharan | EmotionSense AI | Built with DeepFace, OpenCV & Streamlit"
)