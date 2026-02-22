# 🛵 A Deep Learning Framework for Automated Helmet and Triple Riding Violation Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Abstract

This research presents a deep learning-based automated traffic violation detection framework capable of identifying helmet violations and triple riding behavior in real-world traffic scenarios.

The system leverages YOLOv8 for high-accuracy object detection and integrates hyperparameter optimization using Optuna. It is designed for intelligent traffic monitoring applications and can be deployed on CCTV camera feeds for automated enforcement systems.

The framework detects:

- Helmet
- NoHelmet
- Motorbike
- Number Plate (PNumber)
- Triple riding violations (logic-based detection)

---

## 🔗 Dataset

This work uses the publicly available Roboflow dataset:

Helmet Detection Dataset (Roboflow Universe)  
https://universe.roboflow.com/hele/helmet-detection-nsbwm-ftr3v/dataset/4

Dataset Format: YOLO  
Classes:
- Helmet  
- NoHelmet  
- Motorbike  
- PNumber  

The dataset includes both day and night traffic images to improve generalization under varying lighting conditions.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/rohithredddy/Final-Year-Project.git
cd Final-Year-Project
