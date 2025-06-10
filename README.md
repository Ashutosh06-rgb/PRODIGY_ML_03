# 🐶🐱 Dogs vs Cats Image Classifier using SVM

This project implements a simple **Support Vector Machine (SVM)** model to classify cat and dog images from the Kaggle **Dogs vs Cats** dataset. It uses classical machine learning (not deep learning) and performs basic image processing to make the data compatible with SVMs.

---

## 📁 Dataset

- Source: [Kaggle Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats/data)
- Download `train.zip` manually from Kaggle
- Extract it into a folder named `train` in the project directory

---

## 🚀 Features

- Loads and processes image data (resize + flatten)
- Trains an SVM classifier with a linear kernel
- Evaluates model using accuracy, confusion matrix, and classification report
- Saves trained model to disk (`svm_model.pkl`)

---

## 🛠️ Requirements

Install required packages using pip:

```bash
pip install numpy opencv-python scikit-learn joblib
