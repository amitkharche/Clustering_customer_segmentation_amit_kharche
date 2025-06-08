"""
Inference module: Load model, transform and predict cluster for new data.
"""
import joblib
from sklearn.preprocessing import StandardScaler

def load_model(model_path="models/kmeans_model.pkl"):
    return joblib.load(model_path)

def predict_cluster(model, X):
    return model.predict(X)
