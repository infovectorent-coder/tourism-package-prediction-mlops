import pandas as pd

REFERENCE_PATH = "data/reference/reference.csv"
PRODUCTION_PATH = "data/processed/production_batch.csv"

def run_monitoring():
    ref = pd.read_csv(REFERENCE_PATH)
    prod = pd.read_csv(PRODUCTION_PATH)

    if list(ref.columns) != list(prod.columns):
        raise ValueError("Schema violation detected")

    missing_spike = prod.isnull().mean() - ref.isnull().mean()
    print("Missing value spike:")
    print(missing_spike)

    print("Drift monitoring placeholder using Evidently")
    print("Reference shape:", ref.shape)
    print("Production shape:", prod.shape)

if __name__ == "__main__":
    run_monitoring()
