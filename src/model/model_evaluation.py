import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "tushar.dataexpert"
repo_name = "End-to-end-mlops-platform"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/tushar.dataexpert/End-to-end-mlops-platform.mlflow')
# dagshub.init(repo_owner='tushar.dataexpert', repo_name='End-to-end-mlops-platform', mlflow=True)

# -------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import pickle
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.logger import logging

import mlflow
import mlflow.sklearn
import dagshub


# ==========================
# Load Model
# ==========================
def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logging.error('Error loading model: %s', e)
        raise


# ==========================
# Load Data
# ==========================
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data: %s', e)
        raise


# ==========================
# Evaluate Model
# ==========================
def evaluate_model(clf, X_test, y_test):
    try:
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        logging.info('Model evaluation completed')
        return metrics

    except Exception as e:
        logging.error('Error during evaluation: %s', e)
        raise


# ==========================
# Save Metrics
# ==========================
def save_metrics(metrics: dict, file_path: str):
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving metrics: %s', e)
        raise


# ==========================
# Save Model Info (YOUR CODE)
# ==========================
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise


# ==========================
# Main
# ==========================
def main():
    mlflow.set_experiment("sentiment-analysis-tfidf-logreg")

    with mlflow.start_run() as run:
        try:
            # Load model
            clf = load_model('./models/model.pkl')

            # Load TF-IDF processed data
            test_data = load_data('./data/processed/test_tfidf.csv')

            # Separate features & target
            X_test = test_data.drop(columns=['sentiment'])
            y_test = test_data['sentiment']

            # Evaluate
            metrics = evaluate_model(clf, X_test, y_test)

            save_metrics(metrics, 'reports/metrics.json')

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log params
            if hasattr(clf, 'get_params'):
                for k, v in clf.get_params().items():
                    mlflow.log_param(k, v)

            # 🔥🔥 FORCE LOG MODEL (WORKS 100%)
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = os.path.join(tmp_dir, "model")

                # Save model locally
                mlflow.sklearn.save_model(clf, model_path)

                # Log as artifact to MLflow
                mlflow.log_artifacts(model_path, artifact_path="model")

            print("✅ Model saved and logged under 'model/'")
            print("Artifact URI:", mlflow.get_artifact_uri())

            # Save run info (keep same path)
            save_model_info(
                run.info.run_id,
                "model",
                'reports/experiment_info.json'
            )

            # Log artifacts
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/experiment_info.json')

            print("\n✅ Evaluation completed and logged")

        except Exception as e:
            logging.error('Evaluation failed: %s', e)
            print(f"Error: {e}")

if __name__ == "__main__":
    main()