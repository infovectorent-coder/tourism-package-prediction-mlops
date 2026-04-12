from mlflow import MlflowClient

MODEL_NAME = "quick-mlops-model"
ALIAS = "champion"

def set_alias(version: str):
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    print(f"Alias {ALIAS} -> version {version}")

if __name__ == "__main__":
    set_alias("1")
