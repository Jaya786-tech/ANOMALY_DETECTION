import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_df(df, scaler_path=None):
    X = df.select_dtypes(include=["float64", "int64"]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric columns found")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_imp)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return Xs, scaler
