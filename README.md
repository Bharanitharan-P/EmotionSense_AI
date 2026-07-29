# 😊 EmotionSense AI
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end Facial Emotion Recognition System built using **TensorFlow**, **EfficientNetB0**, **OpenCV**, and **Streamlit**. The application detects human facial emotions from images and webcam input, classifying them into seven emotion categories.

---

## 📌 Project Overview

EmotionSense AI is a deep learning-based application that recognizes facial emotions in real time.

The project uses **Transfer Learning** with **EfficientNetB0**, followed by **Fine-Tuning** to improve classification performance on the FER2013 facial emotion dataset.

The application supports:

- 📷 Single Image Emotion Prediction
- 🎥 Real-Time Webcam Emotion Detection
- 📊 Model Evaluation
- 🌐 Streamlit Web Application

---

## ✨ Features

- Detects faces using OpenCV Haar Cascade
- Classifies 7 facial emotions
- Built with EfficientNetB0 Transfer Learning
- Fine-Tuned deep learning model
- Real-time webcam prediction
- Single image prediction
- Interactive Streamlit interface
- Confusion Matrix and Classification Report

---

## 😊 Supported Emotions

| Label | Emotion |
|--------|----------|
| 0 | Angry 😡 |
| 1 | Disgust 🤢 |
| 2 | Fear 😨 |
| 3 | Happy 😊 |
| 4 | Neutral 😐 |
| 5 | Sad 😢 |
| 6 | Surprise 😲 |

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Jupyter Notebook

---

## 📂 Project Structure

```text
EmotionSense_AI/
│
├── notebooks/
├── models/
├── streamlit_app/
├── outputs/
├── test_images/
├── utils/
├── vision/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧠 Model Architecture

- EfficientNetB0 (Pretrained on ImageNet)
- Transfer Learning
- Fine-Tuning
- Global Average Pooling
- Dense Layer
- Dropout
- Softmax Output Layer

---

## 📊 Dataset

Dataset Used:

**FER2013 (Facial Expression Recognition 2013)**

Contains 7 emotion classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## 📈 Results

| Metric | Value |
|--------|--------|
| Test Accuracy | **56.00%** |
| Number of Classes | 7 |
| Model | EfficientNetB0 |
| Input Image Size | 224 × 224 |
| Framework | TensorFlow / Keras |
---

## 🚀 Installation

Clone the repository

```bash
git clone <repository-link>
```

Move into the project folder

```bash
cd EmotionSense_AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

---

## 📷 Screenshots

### 🏠 Home Page

![Home Page](outputs/screenshots/home_page.png)

---

### 🖼 Image Prediction

![Image Prediction](outputs/screenshots/image_prediction.png)

---

### 🎥 Webcam Prediction

![Webcam Prediction](outputs/screenshots/webcam_prediction.png)

---

### 📊 Confusion Matrix

![Confusion Matrix](outputs/screenshots/confusion_matrix.png)

---

## 🔮 Future Improvements

- Improve model accuracy
- Deploy on cloud
- Mobile application
- Face tracking
- Multiple face detection
- Video file prediction

---

## 👨‍💻 Author

**P Bharanitharan**

M.Sc. Data Science

Built as an end-to-end Facial Emotion Recognition System using TensorFlow, EfficientNetB0, OpenCV and Streamlit.
---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub.