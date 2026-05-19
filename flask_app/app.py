from flask import Flask, render_template, request
import mlflow
import pickle
import os
import time
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import dagshub

# Below code block is for production use
# -------------------------------------------------------------------------------------
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

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    return sorted(versions, key=lambda x: int(x.version))[-1].version

model_name = "sentiment_model"
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"

# =========================
# LAZY LOAD
# =========================
model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    if model is None:
        model = mlflow.pyfunc.load_model(model_uri)
    if vectorizer is None:
        vectorizer = pickle.load(open(os.path.join("models", "vectorizer.pkl"), "rb"))

# =========================
# APP
# =========================
app = Flask(__name__)

registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Predictions", ["prediction"], registry=registry)

@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start = time.time()
    res = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start)
    return res

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start = time.time()

    text = request.form["text"]

    if not text.strip():
        return render_template("index.html", result="Please enter valid text")

    load_artifacts()

    # ✅ NO preprocessing here
    vec = vectorizer.transform([text])

    try:
        pred = model.predict(vec)[0]
        result = "Positive" if pred == 1 else "Negative"
    except Exception as e:
        result = str(e) 

    PREDICTION_COUNT.labels(prediction=result).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

    return render_template("index.html", result=result)

@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker