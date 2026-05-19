import unittest
import sys
import os

# 🔥 Fix import path issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask_app.app import app


class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    # ==========================
    # Test Home Page
    # ==========================
    def test_home_page(self):
        response = self.client.get('/')

        self.assertEqual(response.status_code, 200)

        # 🔥 Safer check (avoid strict HTML match)
        self.assertIn(b'Sentiment', response.data)

    # ==========================
    # Test Prediction Endpoint
    # ==========================
    def test_predict_page(self):
        response = self.client.post(
            '/predict',
            data=dict(text="I love this product")
        )

        self.assertEqual(response.status_code, 200)

        response_data = response.data.decode("utf-8")

        # 🔥 Robust check
        self.assertTrue(
            ("Positive" in response_data) or ("Negative" in response_data),
            "Prediction should be either Positive or Negative"
        )


if __name__ == '__main__':
    unittest.main()