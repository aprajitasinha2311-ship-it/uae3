
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file):
    return pd.read_csv(file)

def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop(columns=["conversion_likelihood","default_risk"])
    y_conv = df["conversion_likelihood"]
    y_def = df["default_risk"]
    return df, X, y_conv, y_def
