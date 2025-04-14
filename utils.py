import json
import pandas as pd

from config import REVIEW_CSV_PATH, PRODUCT_CSV_PATH


def load_product_data() -> pd.DataFrame:
    df = pd.read_csv(PRODUCT_CSV_PATH)
    return df


def load_review_data() -> pd.DataFrame:
    df = pd.read_csv(REVIEW_CSV_PATH)
    return df


def load_json(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding {f}: {e}")
    return data
