import numpy as np
import pandas as pd
import os
import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logging

nltk.download('wordnet')
nltk.download('stopwords')


# ==========================
# 🔥 GLOBAL OBJECTS (IMPORTANT FIX)
# ==========================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ==========================
# Text Preprocessing
# ==========================
def preprocess_text(text):
    """Clean a single text string."""
    text = str(text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()

    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    # Lemmatization
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    return text


def preprocess_data(df, text_col='review'):
    """Apply preprocessing to entire dataframe."""
    try:
        df[text_col] = df[text_col].apply(preprocess_text)

        df = df.dropna(subset=[text_col])

        logging.info("Data preprocessing completed")
        return df

    except Exception as e:
        logging.error("Error in preprocessing: %s", e)
        raise


# ==========================
# Main Pipeline
# ==========================
def main():
    try:
        # Load raw data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        logging.info('Raw data loaded')

        # Preprocess
        train_processed = preprocess_data(train_data, 'review')
        test_processed = preprocess_data(test_data, 'review')

        # Save processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed.to_csv(
            os.path.join(data_path, "train_processed_data.csv"),
            index=False
        )

        test_processed.to_csv(
            os.path.join(data_path, "test_processed_data.csv"),
            index=False
        )

        logging.info('Processed data saved to %s', data_path)

    except Exception as e:
        logging.error('Failed preprocessing pipeline: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()