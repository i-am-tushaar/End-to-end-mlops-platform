# feature engineering

import numpy as np
import pandas as pd
import os
import yaml
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logging.error('Error loading params: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    try:
        logging.info("Applying TF-IDF...")

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2)
        )

        X_train = train_data['review'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['review'].values
        y_test = test_data['sentiment'].values

        # ✅ FIT ONLY ON TRAIN
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # ✅ TRANSFORM TEST
        X_test_tfidf = vectorizer.transform(X_test)

        # ✅ Convert to DataFrame
        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['sentiment'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['sentiment'] = y_test

        # ✅ Save vectorizer (AFTER fitting)
        os.makedirs("models", exist_ok=True)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        # ✅ DEBUG CHECK (temporary)
        with open('models/vectorizer.pkl', 'rb') as f:
            v = pickle.load(f)
        print("Vectorizer fitted:", hasattr(v, "idf_"))

        logging.info('TF-IDF applied successfully')

        return train_df, test_df

    except Exception as e:
        logging.error('Error during TF-IDF transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving data: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']

        train_data = load_data('./data/interim/train_processed_data.csv')
        test_data = load_data('./data/interim/test_processed_data.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, "./data/processed/train_tfidf.csv")
        save_data(test_df, "./data/processed/test_tfidf.csv")

    except Exception as e:
        logging.error('Feature engineering failed: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()