# dogs_vs_cats_svm.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ====== STEP 1: DOWNLOAD MANUALLY ======
# Go to https://www.kaggle.com/c/dogs-vs-cats/data
# Download train.zip and extract it in the same directory as this script.
# After extracting, you'll get: ./train/cat.0.jpg, dog.0.jpg, etc.

DATA_DIR = "./train" 
IMG_SIZE = 64        

X = []
y = []

print("[INFO] Loading and preprocessing images...")
for img_name in os.listdir(DATA_DIR)[:5000]:  # Use subset for speed
    label = 1 if 'dog' in img_name else 0
    img_path = os.path.join(DATA_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_flatten = img.flatten()
    X.append(img_flatten)
    y.append(label)

X = np.array(X)
y = np.array(y)

# dogs_vs_cats_svm.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ====== STEP 1: DOWNLOAD MANUALLY ======
# Go to https://www.kaggle.com/c/dogs-vs-cats/data
# Download train.zip and extract it in the same directory as this script.
# After extracting, you'll get: ./train/cat.0.jpg, dog.0.jpg, etc.

DATA_DIR = "./train"  # Update path if needed
IMG_SIZE = 64          # Resize images for SVM

X = []
y = []

print("[INFO] Loading and preprocessing images...")
for img_name in os.listdir(DATA_DIR)[:5000]:  # Use subset for speed
    label = 1 if 'dog' in img_name else 0
    img_path = os.path.join(DATA_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_flatten = img.flatten()
    X.append(img_flatten)
    y.append(label)

X = np.array(X)
y = np.array(y)

# ====== STEP 2: TRAIN-TEST SPLIT ======
print("[INFO] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== STEP 3: TRAIN SVM ======
print("[INFO] Training SVM model...")
svm_clf = SVC(kernel='linear', verbose=True)
svm_clf.fit(X_train, y_train)

# ====== STEP 4: EVALUATE ======
print("[INFO] Evaluating model...")
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ====== STEP 5: SAVE MODEL ======
print("[INFO] Saving model to svm_model.pkl")
joblib.dump(svm_clf, "svm_model.pkl")

print("[INFO] All done!")

print("[INFO] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== STEP 3: TRAIN SVM ======
print("[INFO] Training SVM model...")
svm_clf = SVC(kernel='linear', verbose=True)
svm_clf.fit(X_train, y_train)

# ====== STEP 4: EVALUATE ======
print("[INFO] Evaluating model...")
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ====== STEP 5: SAVE MODEL ======
print("[INFO] Saving model to svm_model.pkl")
joblib.dump(svm_clf, "svm_model.pkl")

print("[INFO] All done!")
