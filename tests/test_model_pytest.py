import mlflow
import os
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv

load_dotenv()

# ==========================
# Fixture (replaces setUpClass)
# ==========================
@pytest.fixture(scope="module")
def model_setup():
    # Load env variables
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "tushar.dataexpert"
    repo_name = "End-to-end-mlops-platform"

    # MLflow setup
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    model_name = "sentiment_model"
    model_version = get_latest_model_version(model_name)

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    test_data = pd.read_csv("data/processed/test_tfidf.csv")

    return model, test_data


# ==========================
# Helper function
# ==========================
def get_latest_model_version(model_name, stage=None):
    client = mlflow.MlflowClient()

    if stage:
        versions = client.get_latest_versions(model_name, stages=[stage])
    else:
        versions = client.get_latest_versions(model_name)

    return versions[0].version if versions else None


# ==========================
# Test 1: Model Loading
# ==========================
def test_model_loaded(model_setup):
    model, _ = model_setup
    assert model is not None


# ==========================
# Test 2: Model Signature
# ==========================
def test_model_signature(model_setup):
    model, test_data = model_setup

    sample_input = test_data.drop(columns=['sentiment']).iloc[:1]
    prediction = model.predict(sample_input)

    assert sample_input.shape[1] == test_data.drop(columns=['sentiment']).shape[1]
    assert len(prediction) == sample_input.shape[0]
    assert len(prediction.shape) == 1


# ==========================
# Test 3: Model Performance
# ==========================
def test_model_performance(model_setup):
    model, test_data = model_setup

    X_test = test_data.drop(columns=['sentiment'])
    y_test = test_data['sentiment']

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    assert accuracy >= 0.60
    assert precision >= 0.60
    assert recall >= 0.60
    assert f1 >= 0.60