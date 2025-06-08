"""
Model Training Script for Customer Segmentation using KMeans Clustering.
"""

import os
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "data/mall_customers.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
N_CLUSTERS = 5
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load customer dataset from CSV."""
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    logger.info(f"Loading data from: {path}")
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """Extract features and scale them."""
    logger.info("Preprocessing data")
    if not {"Annual Income (k$)", "Spending Score (1-100)"}.issubset(df.columns):
        raise ValueError("Required columns not found in dataset.")
    
    features = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


def train_kmeans(data: np.ndarray, n_clusters: int = 5) -> KMeans:
    """Train KMeans clustering model."""
    logger.info(f"Training KMeans with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    kmeans.fit(data)
    return kmeans


def save_model(obj, path: str):
    """Save a model or object to disk."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to: {path}")


def main():
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and process data
    df = load_data(DATA_PATH)
    scaled_data, scaler = preprocess_data(df)

    # Train model
    kmeans = train_kmeans(scaled_data, N_CLUSTERS)

    # Save model and scaler
    save_model(kmeans, MODEL_PATH)
    save_model(scaler, SCALER_PATH)
    logger.info("Model training complete")


if __name__ == "__main__":
    main()
