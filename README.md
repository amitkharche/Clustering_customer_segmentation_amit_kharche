# 🧠 Customer Segmentation Using Clustering

## 📌 Business Problem
Understanding customer behavior is crucial for businesses to tailor marketing strategies, improve product recommendations, and personalize experiences. Clustering customers into groups based on demographic and transactional behavior helps target campaigns more effectively.

### Real-World Use Case:
This project uses the Mall Customer Dataset to segment customers for **marketing campaigns**, enabling:
- **Personalized promotions**
- **Budget optimization**
- **Product placement strategy**

## 📂 Dataset
- **Source**: Simulated (similar to Mall Customer Dataset)
- **Features**:
  - `CustomerID`
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`
- **Rows**: 10
- **Distribution**: Balanced by gender with income and spending diversity

## ⚙️ Pipeline
1. Load and clean data
2. Exploratory Data Analysis (EDA)
3. Preprocessing (scaling, encoding)
4. Feature engineering
5. Clustering with KMeans
6. Model evaluation using silhouette score
7. Visualization and interpretation
8. Streamlit app for predictions

## 💻 Folder Structure
```
customer_segmentation_project/
├── data/               # Raw and processed data
├── models/             # Saved model artifacts
├── notebooks/          # EDA and training analysis
├── src/                # Modular Python scripts
├── app/                # Streamlit app
├── outputs/            # Visualizations
├── logs/               # Logs
├── tests/              # Unit tests (TBD)
├── .github/            # GitHub templates
├── Dockerfile          # For container deployment
├── requirements.txt    # Dependencies
├── README.md           # This file
```

## 🚀 Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python src/model_training.py
```

### Run Streamlit App
```bash
streamlit run app/app.py
```

## 🐳 Docker
```bash
docker build -t customer-segmentation .
docker run -p 8501:8501 customer-segmentation
```

## 🧪 Explainability
- Add SHAP or LIME visualizations to enhance model transparency

## 📄 License
MIT License
