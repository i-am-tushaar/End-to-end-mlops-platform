import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data: %s', e)
        raise


def train_model(X_train, y_train):
    try:
        clf = LogisticRegression(
            C=1,
            solver='liblinear',
            penalty='l2',
            class_weight='balanced',
            max_iter=1000
        )

        clf.fit(X_train, y_train)

        logging.info('Model training completed')
        return clf

    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving model: %s', e)
        raise


def main():
    try:
        # 🔥 Use TF-IDF processed data
        train_data = load_data('./data/processed/train_tfidf.csv')

        # Features & target
        X = train_data.drop(columns=['sentiment'])
        y = train_data['sentiment']

        # Train-Test split (still useful for sanity, but no evaluation here)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        clf = train_model(X_train, y_train)

        # Save model
        save_model(clf, 'models/model.pkl')

        print("✅ Model training completed and saved")

    except Exception as e:
        logging.error('Model building failed: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()