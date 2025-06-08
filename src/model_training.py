
"""
Train clustering model (KMeans), evaluate using silhouette score, save model and scaler.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os

# Import preprocessing functions
from preprocessing import load_data, preprocess  # make sure src is in PYTHONPATH or adjust path

def train_model(X, n_clusters=5, model_dir="models"):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    score = silhouette_score(X, model.labels_)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/kmeans_model.pkl")
    return model, score

# === Main Execution ===
if __name__ == "__main__":
    # Step 1: Load data from correct path
    df = load_data("data/raw/mall_customers.csv")

    # Step 2: Preprocess (drop ID, encode, scale)
    X_scaled, scaler = preprocess(df)

    # Step 3: Train model
    model, score = train_model(X_scaled)

    # Step 4: Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Step 5: Confirm success
    print(f"✅ Model trained with silhouette score: {score:.4f}")
    print("✅ Model saved to models/kmeans_model.pkl")
    print("✅ Scaler saved to models/scaler.pkl")
