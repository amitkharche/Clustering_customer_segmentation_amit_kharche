"""
Train clustering model (KMeans), evaluate using silhouette score, save model and scaler.
"""

import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === Importing from other modules ===
from preprocessing import load_data, preprocess
from feature_engineering import feature_selection
from utils import setup_logging

def train_model(X, n_clusters=5, model_dir="models"):
    """Train and save KMeans model, return model and silhouette score."""
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    score = silhouette_score(X, model.labels_)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "kmeans_model.pkl"))
    return model, score

# === Main Execution ===
if __name__ == "__main__":
    # Step 0: Setup logging
    setup_logging()
    import logging
    logging.info("Started training pipeline")

    # Step 1: Load data
    df = load_data("data/raw/mall_customers.csv")
    logging.info(f"Loaded data with shape {df.shape}")

    # Step 2: Feature selection (optional step for enhancement)
    df = feature_selection(df)
    logging.info("Feature selection completed")

    # Step 3: Preprocess data
    X_scaled, scaler = preprocess(df)
    logging.info("Preprocessing completed")

    # Step 4: Train model
    model, score = train_model(X_scaled)
    logging.info(f"Model trained with silhouette score: {score:.4f}")

    # Step 5: Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    logging.info("Scaler saved to models/scaler.pkl")

    # Step 6: Final messages
    print(f"✅ Model trained with silhouette score: {score:.4f}")
    print("✅ Model saved to models/kmeans_model.pkl")
    print("✅ Scaler saved to models/scaler.pkl")
