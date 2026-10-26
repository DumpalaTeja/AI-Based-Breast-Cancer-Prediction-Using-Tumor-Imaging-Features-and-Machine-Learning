# AI-Based Breast Cancer Prediction Using Tumour Imaging and Machine Learning

## 📌 Project Overview

This project aims to classify breast tumour images as **Benign or Malignant** using Artificial Intelligence and Machine Learning. It uses image feature extraction and machine learning techniques to support tumour classification through an interactive web application.

## 🎯 Objectives

- Classify breast tumour images into benign and malignant categories.
- Extract meaningful image features using a deep learning model.
- Provide predictions through a user-friendly web interface.
- Visualize prediction results and estimated malignancy probability.

## 🛠️ Tools and Technologies

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn, Random Forest
- **Deep Learning:** TensorFlow/Keras, ResNet50
- **Web Framework:** Flask
- **Data Processing:** Pandas, NumPy
- **Image Processing:** Pillow
- **Model Serialization:** Joblib
- **Development Environment:** Visual Studio Code

## 📂 Dataset

**BreakHis Breast Cancer Dataset**

The BreakHis dataset contains microscopic histopathological images of benign and malignant breast tumour tissue.

Dataset source: [BreakHis on Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis)

A numerical dataset, Breast Cancer Wisconsin (Diagnostic), was also used in the initial development of this project.

## ⚙️ Project Workflow

1. Collect and organize the tumour image dataset.
2. Preprocess images and resize them to the required input dimensions.
3. Extract image features using the pretrained ResNet50 model.
4. Train a Random Forest classifier using the extracted features.
5. Evaluate the model using appropriate classification metrics.
6. Integrate the trained model into a Flask web application.
7. Display predictions and probability visualizations.

## 🌐 Web Application

The Flask interface is designed to support image uploads and display the predicted tumour category and an estimated malignancy probability.

## 🚀 How to Run

1. Clone or download this repository.
2. Install the required dependencies:

   ```bash
   pip install flask tensorflow scikit-learn pandas numpy pillow joblib
   ```

3. Ensure the trained model and required project files are available in the correct locations.
4. Run the application:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000/` in your browser.

## 📊 Evaluation

Model performance should be reported using measured results, including accuracy, precision, recall, F1-score, and a confusion matrix. Actual values should be added after evaluating the trained image model.

## ⚠️ Disclaimer

This project is for educational and research purposes only. Its predictions are not a substitute for professional medical diagnosis.

## 👨‍💻 Author

**Dumpala Teja**

B.Tech – Computer Science and Engineering (Artificial Intelligence and Machine Learning)

SRM Institute of Science and Technology, Tiruchirappalli
