import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.ingest import ingest_data
from src.validate import validate_data
from src.transform import transform_data

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "quick-mlops-model"

def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("quick-mlops-exp")

    df = ingest_data()
    validate_data(df)
    X_train, X_test, y_train, y_test = transform_data(df)

    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        return acc, f1
