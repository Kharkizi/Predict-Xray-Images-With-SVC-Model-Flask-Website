import joblib
import numpy as np


def load_model():
    # Load mô hình SVM từ file
    model = joblib.load(r"model_directory\svc_model.pkl")
    return model

def model_predict(input_data):
    # Load mô hình
    model = load_model()
    # Dự đoán nhãn
    predicted_label = model.predict(input_data)[0]
    return predicted_label

        
    