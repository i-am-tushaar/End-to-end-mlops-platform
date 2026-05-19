
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY flask_app/requirements.txt ./requirements.txt

# Install dependencies (NOW includes correct sklearn version)
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY flask_app/ /app/

# Create models folder
RUN mkdir -p /app/models

# Copy vectorizer
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

EXPOSE 5000

#local
# CMD ["python", "app.py"]  

# Production server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]