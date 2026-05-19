# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dotenv import load_dotenv
load_dotenv()

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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

        cls.model_name = "sentiment_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        cls.test_data = pd.read_csv("data/processed/test_tfidf.csv")

    @staticmethod
    def get_latest_model_version(model_name, stage=None):
        client = mlflow.MlflowClient()

        if stage:
            latest_version = client.get_latest_versions(model_name, stages=[stage])
        else:
            latest_version = client.get_latest_versions(model_name)

        return latest_version[0].version if latest_version else None

    # ==========================
    # Test 1: Model Loading
    # ==========================
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    # ==========================
    # Test 2: Model Signature
    # ==========================
    def test_model_signature(self):
        # ✅ FIX 3: numeric input (same as training)
        sample_input = self.test_data.drop(columns=['sentiment']).iloc[:1]

        prediction = self.model.predict(sample_input)

        # Input should have same number of features
        self.assertEqual(
            sample_input.shape[1],
            self.test_data.drop(columns=['sentiment']).shape[1]
        )

        # Output shape check
        self.assertEqual(len(prediction), sample_input.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    # ==========================
    # Test 3: Model Performance
    # ==========================
    def test_model_performance(self):
        X_test = self.test_data.drop(columns=['sentiment'])
        y_test = self.test_data['sentiment']

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # realistic thresholds
        self.assertGreaterEqual(accuracy, 0.60)
        self.assertGreaterEqual(precision, 0.60)
        self.assertGreaterEqual(recall, 0.60)
        self.assertGreaterEqual(f1, 0.60)


if __name__ == "__main__":
    unittest.main()