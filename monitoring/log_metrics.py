import csv
import os
from datetime import datetime

LOG_FILE = "monitoring/model_metrics_log.csv"

def log_metrics(accuracy, f1):
    os.makedirs("monitoring", exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "accuracy", "f1_score"])
        writer.writerow([datetime.utcnow().isoformat(), accuracy, f1])

if __name__ == "__main__":
    log_metrics(0.82, 0.78)
