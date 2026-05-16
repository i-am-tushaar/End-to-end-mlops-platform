# register model

import json
import logging
from src.logger import logging
import os
from mlflow.tracking import MlflowClient
import time

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os

# 🔹 Load environment variables FIRST
load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

import mlflow
import dagshub

# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/tushar.dataexpert/End-to-end-mlops-platform.mlflow')
dagshub.init(repo_owner='tushar.dataexpert', repo_name='End-to-end-mlops-platform', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        print("Model URI:", model_uri)

        # Register model
        model_version = mlflow.register_model(model_uri, model_name)

        client = MlflowClient()

        # 🔥 WAIT until model is READY
        for _ in range(10):
            mv = client.get_model_version(
                name=model_name,
                version=model_version.version
            )
            if mv.status == "READY":
                break
            print("⏳ Waiting for model to be READY...")
            time.sleep(2)

        # ✅ Now transition stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True
        )

        print(f"✅ Model '{model_name}' version {model_version.version} moved to Staging")

    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)

    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
