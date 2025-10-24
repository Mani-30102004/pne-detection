import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, 'pneumonia_model.h5')
IMG_PATH = os.path.join(ROOT_DIR, 'uploaded.jpg')

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")
    raise SystemExit(1)

if not os.path.exists(IMG_PATH):
    print(f"Image file not found at {IMG_PATH}")
    raise SystemExit(1)

print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print('Model loaded.')

print(f"Loading image from: {IMG_PATH}")
img = image.load_img(IMG_PATH, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

print('Running prediction...')
prob = model.predict(img_array)[0][0]
label = 'PNEUMONIA' if prob > 0.5 else 'NORMAL'
print(f'Probability: {prob:.6f} — Label: {label}')
