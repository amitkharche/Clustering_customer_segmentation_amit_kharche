"""
Preprocessing module: Load data, handle missing values, scaling, encoding.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()
    df = df.drop(columns=["CustomerID"])
    df = pd.get_dummies(df, drop_first=True)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler
