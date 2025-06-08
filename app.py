"""
Streamlit App for Mall Customer Segmentation
"""

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

REQUIRED_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("🛍️ Mall Customer Segmentation App")

@st.cache_resource
def load_model():
    """Load trained KMeans model and scaler."""
    try:
        with open("model/kmeans_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please run model_training.py first.")
        return None, None


def validate_data(df: pd.DataFrame) -> bool:
    """Check if required columns are present."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return False
    return True


def predict_and_visualize(df: pd.DataFrame, model, scaler):
    """Generate cluster predictions and PCA visualization."""
    features = df[REQUIRED_COLUMNS]
    scaled = scaler.transform(features)
    clusters = model.predict(scaled)

    df["Cluster"] = clusters

    # PCA for 2D projection
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    df["PCA1"], df["PCA2"] = components[:, 0], components[:, 1]

    # Show clustered data
    st.subheader("Segmented Customers")
    st.dataframe(df.head())

    # Show visualization
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100, ax=ax)
    ax.set_title("Customer Segments (PCA Projection)")
    st.pyplot(fig)


def main():
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
