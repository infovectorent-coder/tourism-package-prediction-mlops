import pandas as pd
import os

def ingest_data(path="data/raw/data.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    return df
