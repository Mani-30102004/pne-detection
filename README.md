# pneumonia-detection

This project is an Machine Learning-based Pneumonia Detection System which diagnoses chest X-ray images to predict whether a patient is Pneumonia or Normal. This model is developed using TensorFlow & Keras and hosted on Streamlit for an interactive web application.

Project files
- app.py — Streamlit web UI
- train.py — training script
- preprocessing.py — data preprocessing utilities
- evaluate.py — evaluation utilities
- pneumonia_model.h5 — trained Keras model (not included by default)
- requirements.txt — Python dependencies

Dataset layout expected: chest_xray/
	├── train/
	├── test/
	└── val/

How to use the web app
1. Start the Streamlit app: `streamlit run app.py` (or use the venv python module)
2. Upload a chest X-ray image
3. Click Predict to receive the diagnosis (PNEUMONIA or NORMAL)

Dataset
Chest X-ray Images (Pneumonia) — Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Libraries used: TensorFlow, Keras, OpenCV, Streamlit
