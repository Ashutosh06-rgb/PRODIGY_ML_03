from google.colab import files
uploaded = files.upload()  # Upload your cat_and_dog.zip

import zipfile
with zipfile.ZipFile("cat and dog.zip", 'r') as zip_ref:
    zip_ref.extractall("cat and dog")
    
import os
import cv2
import numpy as np

def load_images(folder_path, label, img_size=64):
    X, y = [], []
    # Add a check to see if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return X, y

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            X.append(img.flatten())   # flatten for SVM
            y.append(label)
    return X, y

# Set paths to inner folders
cat_folder = "cat and dog/cat and dog/cat_small"
dog_folder = "cat and dog/cat and dog/dog_small"


# Load images
X_cat, y_cat = load_images(cat_folder, label=0)
X_dog, y_dog = load_images(dog_folder, label=1)

# Combine
X = np.array(X_cat + X_dog)
y = np.array(y_cat + y_dog)

print("Total Images:", len(X))

import os

# List the contents of the extracted directory
extracted_folder_path = "cat and dog"
if os.path.exists(extracted_folder_path):
    print(os.listdir(extracted_folder_path))
else:
    print(f"The directory '{extracted_folder_path}' does not exist.")    

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# for testing the model
from google.colab import files
uploaded = files.upload()  # Upload an image

import matplotlib.pyplot as plt

for fn in uploaded:
    img = cv2.imread(fn)
    img_resized = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)
    prediction = model.predict(img_resized)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Predicted: " + ("Dog 🐶" if prediction[0] == 1 else "Cat 🐱"))
    plt.axis('off')
    plt.show()
