
---

#  Customer Segmentation Using Clustering

A machine learning solution to **segment mall customers** based on demographic and behavioral features. This enables businesses to deliver **personalized marketing**, improve customer retention, and optimize product placement.

---

##  Business Objective

Customer segmentation helps companies group similar customers together to:

* Design **targeted promotions**
* **Personalize** communication and offers
* Optimize **budget allocation** in campaigns
* Improve **product recommendations**

This project uses clustering (KMeans) to group customers by **annual income** and **spending score**.

---

##  Dataset Overview

| Feature              | Description                           |
| -------------------- | ------------------------------------- |
| `CustomerID`         | Unique customer identifier            |
| `Gender`             | Male/Female                           |
| `Age`                | Age of the customer                   |
| `Annual Income (k$)` | Yearly income in thousand dollars     |
| `Spending Score`     | Spending behavior score from 1 to 100 |


---

##  Features Used

| Type            | Features                                      |
| --------------- | --------------------------------------------- |
| **Numerical**   | `Age`, `Annual Income (k$)`, `Spending Score` |
| **Categorical** | `Gender` (encoded)                            |

---

##  ML Pipeline

### 1. **Data Preprocessing**

* Drop ID column
* One-hot encode `Gender`
* Scale features using `StandardScaler`

### 2. **Feature Engineering**

* Placeholder for future derived features

### 3. **Model Training**

* KMeans clustering (`n_clusters=5`)
* Evaluate with `silhouette_score`

### 4. **Deployment via Streamlit**

* Upload new customer data
* Predict cluster segment
* Visualize with PCA-based 2D scatter plot

---

##  Project Structure

```
customer_segmentation_project/
├── data/                        # Raw CSVs or uploaded data
├── models/                      # Saved model and scaler
├── src/
│   ├── preprocessing.py         # Load and transform data
│   ├── inference.py             # Load model, predict cluster
│   ├── feature_engineering.py   # Feature selection pipeline
│   ├── utils.py                 # Logging utilities
│   └── model_training.py        # Training pipeline script
├── app/
│   └── app.py                   # Streamlit app
├── outputs/                     # Visualizations and results
├── notebooks/                   # EDA or analysis notebooks
├── logs/                        # Training logs
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker setup
├── README.md                    # Project documentation
```

---

##  How to Use

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python src/model_training.py
```

This will generate:

* `models/kmeans_model.pkl`
* `models/scaler.pkl`

### Step 3: Launch Streamlit App

```bash
streamlit run app/app.py
```

You can upload a customer CSV and get segment predictions with PCA plots.

---

##  Docker Usage

```bash
docker build -t customer-segmentation .
docker run -p 8501:8501 customer-segmentation
```

---

##  Future Enhancements

* SHAP-based explainability
* Support for DBSCAN or GMM clustering
* Interactive dashboard for marketing insights
* Automated EDA and feature reports

---

##  License

This project is licensed under the **MIT License** – free for personal and commercial use.

---

## 📬 Contact

If you have questions or want to collaborate, feel free to connect with me on
- [LinkedIn](https://www.linkedin.com/in/amit-kharche)  
- [Medium](https://medium.com/@amitkharche14)  
- [GitHub](https://github.com/amitkharche)
