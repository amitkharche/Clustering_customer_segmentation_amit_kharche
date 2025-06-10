"""
Streamlit App for Mall Customer Segmentation
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

from inference import load_model, predict_cluster
from feature_engineering import feature_selection  # Optional, depends on pipeline

REQUIRED_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("🛍️ Mall Customer Segmentation App")

def validate_data(df: pd.DataFrame) -> bool:
    """Check if required columns are present."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return False
    return True

def predict_and_visualize(df: pd.DataFrame, model, scaler):
    """Apply preprocessing, predict clusters, and show PCA chart."""

    # Step 1: Feature engineering (if any)
    df = feature_selection(df)

    # Step 2: Drop ID column
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Step 3: One-hot encode categorical (e.g., Gender)
    df = pd.get_dummies(df, drop_first=True)

    # Step 4: Align features with training columns
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Step 5: Scale and predict
    scaled = scaler.transform(df)
    df["Cluster"] = predict_cluster(model, scaled)

    # Step 6: PCA for visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)
    df["PCA1"], df["PCA2"] = reduced[:, 0], reduced[:, 1]

    # Step 7: Show table and plot
    st.subheader("Segmented Customers")
    st.dataframe(df)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100, ax=ax)
    ax.set_title("Customer Segments (PCA Projection)")
    st.pyplot(fig)

def main():
    st.write("📌 Current working directory:", os.getcwd())

    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data", df.head())

            if not validate_data(df):
                return

            model, scaler = load_model()
            if model and scaler:
                predict_and_visualize(df, model, scaler)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
