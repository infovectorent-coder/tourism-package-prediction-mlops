name: CD

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Start MLflow
        run: |
          mlflow server --host 127.0.0.1 --port 5000 &
          sleep 10

      - name: Train and evaluate
        run: python src/evaluate.py

      - name: Build Docker image
        run: docker build -t quick-mlops .

      - name: Deploy
        run: echo "Deploy step placeholder"
