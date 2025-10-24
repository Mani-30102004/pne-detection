import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import sys

sys.stdout.reconfigure(encoding='utf-8')

model_path = r"C:\Users\singo\OneDrive\Desktop\Pneumonia-Detection\pneumonia_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model file not found: {model_path}. Train the model first.")

model = keras.models.load_model(model_path)

test_folder = r"C:\Users\singo\OneDrive\Desktop\Pneumonia-Detection\chest_xray\chest_xray\test\PNEUMONIA"

if not os.path.exists(test_folder):
    raise FileNotFoundError(f" Test folder not found: {test_folder}")

image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    raise FileNotFoundError(f" No images found in {test_folder}")

random_image = random.choice(image_files)
random_image_path = os.path.join(test_folder, random_image)

print(f"Evaluating: {random_image_path}".encode("utf-8", errors="ignore").decode("utf-8"))

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  

    prediction = model.predict(img_array)

    return "PNEUMONIA" if prediction[0] > 0.5 else "NORMAL"

result = predict_image(random_image_path)

print(f"Prediction: {result}".encode("utf-8", errors="ignore").decode("utf-8"))


