# 🌿 Plant Disease Recognition System

🚀 **Live Demo:** https://plant-disease-detector-dipen.streamlit.app

A deep learning web application that detects plant leaf diseases from images in real-time, achieving **97% classification accuracy** across 38 disease categories using TensorFlow and Streamlit.

---

## 📌 Overview

This project leverages transfer learning to classify plant diseases across multiple crop types. It provides a fast, accurate, and user-friendly interface for farmers and researchers to identify diseases early and take corrective action.

---

## ✨ Key Features

- **97% Classification Accuracy** — achieved via transfer learning, hyperparameter tuning, and image augmentation
- **38 Disease Categories** — multi-class classification across multiple crop types
- **Real-time Prediction** — upload an image and get instant results
- **Transfer Learning** — fine-tuned ResNet50 and VGG16 models pretrained on ImageNet
- **Interactive UI** — clean Streamlit dashboard for easy use
- **Live Deployment** — fully deployed and accessible at the link above

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Models | ResNet50, VGG16 |
| Computer Vision | OpenCV, NumPy |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📊 Model Details

- **Dataset:** PlantVillage
- **Architecture:** ResNet50 & VGG16 (Transfer Learning)
- **Image Size:** 128 × 128
- **Output Classes:** 38 disease categories
- **Accuracy:** 97% (validated on test split)

---

## 🚀 How It Works

1. Upload a plant leaf image
2. The model processes and analyzes visual patterns
3. It predicts the disease class using the fine-tuned model
4. The app displays the result with the disease name instantly

---

## 📁 Project Structure

```text
Plant_Disease_Dataset/
├── main.py
├── trained_model.h5
├── requirements.txt
├── home.jpg
├── Train_plant_disease.ipynb
└── README.md
```

---

## 🎯 Use Case

Designed for:
- Farmers 🌱 — quick field diagnosis without expert consultation
- Agricultural researchers 🌾 — dataset exploration and model benchmarking
- Students 🤖 — learning AI + Computer Vision in a real-world context

---

## 📸 Demo

<img width="1455" height="873" alt="App Screenshot 1" src="https://github.com/user-attachments/assets/c5a6ae1d-51f1-4c36-aa05-95235690ea8a" />
<img width="1188" height="1080" alt="App Screenshot 2" src="https://github.com/user-attachments/assets/be53c0c5-ad97-44b7-b941-9dd80ffde4ac" />

---

## 👤 Author

**Dipendra Pandey**  
EE + CS Student @ Tennessee State University  
[LinkedIn](https://www.linkedin.com/in/dipendrapandey6)
