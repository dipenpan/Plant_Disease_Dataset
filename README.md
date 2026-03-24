# 🌿 Plant Disease Detection

A deep learning system that detects plant leaf diseases from images using transfer learning, achieving **97% classification accuracy** on the PlantVillage dataset.

---

## 📌 Overview

This project applies transfer learning with **ResNet50** and **VGG16** to classify plant diseases across multiple crop types. The goal is to provide farmers and agricultural researchers with a fast, accurate diagnostic tool that reduces crop loss through early disease identification.

---

## ✨ Key Features

- **97% Classification Accuracy** — achieved via hyperparameter tuning, image augmentation, and preprocessing pipelines
- **Transfer Learning** — fine-tuned ResNet50 and VGG16 models pretrained on ImageNet
- **Data Preprocessing** — processed 1,600+ images with RGB-to-HSV color space transformation, segmentation, and feature scaling
- **Interactive Web Interface** — real-time image upload with instant disease classification and mitigation recommendations
- **Multi-class Detection** — identifies diseases across multiple plant species from the PlantVillage dataset

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Models | ResNet50, VGG16 |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |
| Web Interface | Flask |

---

## 📊 Results

| Model | Accuracy |
|---|---|
| VGG16 (fine-tuned) | 95% |
| ResNet50 (fine-tuned) | **97%** |
| Baseline CNN | ~80% |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow opencv-python flask numpy pandas matplotlib scikit-learn
```

### Installation
```bash
git clone https://github.com/dipenpan/Plant_Disease_Dataset
cd Plant_Disease_Dataset
```

### Run the Web App
```bash
python app.py
```
Then open `http://localhost:5000` in your browser and upload a plant leaf image to get an instant diagnosis.

---

## 📁 Project Structure

```
Plant_Disease_Dataset/
├── models/
│   ├── resnet50_model.h5
│   └── vgg16_model.h5
├── static/
├── templates/
├── app.py
├── train.py
├── training_hist.json
└── README.md
```

---

## 🎯 Use Case

Designed for agricultural researchers and farmers who need a lightweight, accessible tool to identify plant diseases early — reducing crop loss and supporting data-driven farming decisions.

---

## 👤 Author

**Dipendra Pandey**  
EE + CS Student @ Tennessee State University  
[LinkedIn](https://www.linkedin.com/in/dipendrapandey6)
