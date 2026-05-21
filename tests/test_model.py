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
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(
            "https://dagshub.com/tushar.dataexpert/End-to-end-mlops-platform.mlflow"
        )

        cls.model_name = "sentiment_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)

        if cls.model_version is None:
            raise Exception("No model versions found in MLflow")

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        cls.test_data = pd.read_csv("data/processed/test_tfidf.csv")

    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        return sorted(versions, key=lambda x: int(x.version))[-1].version if versions else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        sample_input = self.test_data.drop(columns=['sentiment']).iloc[:1]
        prediction = self.model.predict(sample_input)

        self.assertEqual(sample_input.shape[1],
                         self.test_data.drop(columns=['sentiment']).shape[1])
        self.assertEqual(len(prediction), sample_input.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_test = self.test_data.drop(columns=['sentiment'])
        y_test = self.test_data['sentiment']

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        self.assertGreaterEqual(accuracy, 0.60)
        self.assertGreaterEqual(precision, 0.60)
        self.assertGreaterEqual(recall, 0.60)
        self.assertGreaterEqual(f1, 0.60)


if __name__ == "__main__":
    unittest.main()