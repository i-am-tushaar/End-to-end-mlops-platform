from flask import Flask, render_template, request
import mlflow
import pickle
import os
import time
import re
import string

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import dagshub
import warnings
warnings.filterwarnings("ignore")

# =========================
# TEXT PREPROCESSING
# =========================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = str(text)

    # remove urls
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # remove numbers
    text = ''.join([char for char in text if not char.isdigit()])

    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text).strip()

    # remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    # lemmatization
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/tushar.dataexpert/End-to-end-mlops-platform.mlflow')
dagshub.init(repo_owner='tushar.dataexpert', repo_name='End-to-end-mlops-platform', mlflow=True)
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "tushar.dataexpert"
# repo_name = "End-to-end-mlops-platform"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Initialize Flask app
app = Flask(__name__)

# from prometheus_client import CollectorRegistry

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup
model_name = "sentiment_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)

# 🔥 IMPORTANT
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()

    response = render_template("index.html", result=None)

    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]

    # 🔥 FIX 1: correct preprocessing
    processed_text = preprocess_text(text)

    # 🔥 FIX 2: convert to vector
    text_vector = vectorizer.transform([processed_text])

    # 🔥 FIX 3: prediction
    result = model.predict(text_vector)[0]

    # 🔥 FIX 4: human readable output
    prediction = "Positive" if result == 1 else "Negative"

    PREDICTION_COUNT.labels(prediction=prediction).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)