from typing import Any

import emoji
import pandas as pd

from utils import load_review_data, load_product_data

REVIEW_COLUMNS = [
    'author_id', 'rating', 'submission_time', 'review_text',
    'skin_tone', 'eye_color', 'skin_type', 'hair_color',
    'product_id',
]
PRODUCT_COLUMNS = [
    'product_id', 'product_name', 'brand_name', 'loves_count',
    'rating', 'reviews', 'size', 'ingredients', 'price_usd',
    'primary_category', 'secondary_category', 'tertiary_category'
]
MIN_TEXT_LEN = 50
MAX_TEXT_LEN = 2000
MIN_YEAR = 2023


def filter_data(review_df_raw: pd.DataFrame, product_df_raw: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    print("Filtering data...")
    review_df = review_df_raw[REVIEW_COLUMNS].copy()
    product_df = product_df_raw[PRODUCT_COLUMNS].copy()

    review_df = review_df.dropna()
    review_df["author_id"] = review_df["author_id"].astype(str)
    review_df['submission_time'] = pd.to_datetime(review_df['submission_time'])
    review_df['year'] = review_df['submission_time'].dt.year
    review_df['review_text_len'] = review_df['review_text'].astype(str).str.len()
    review_df = review_df[review_df['review_text_len'] >= MIN_TEXT_LEN & (review_df['review_text_len'] <= MAX_TEXT_LEN)]
    review_df = review_df[review_df['year'] >= MIN_YEAR]
    review_df = review_df[REVIEW_COLUMNS]

    review_df['review_text'] = review_df['review_text'].apply(clean_text)
    review_df = review_df.drop_duplicates(subset=['review_text'])

    product_df = product_df[product_df['product_id'].isin(review_df['product_id'])]
    return review_df, product_df


def clean_text(text: str) -> str:
    text = emoji.replace_emoji(text, replace='')
    return text


if __name__ == "__main__":
    review_df = load_review_data()  # shape (1094411, 18)
    product_df = load_product_data()  # shape (8494, 27)

    review_df, product_df = filter_data(review_df, product_df)

    print(review_df.shape)  # (21278, 10)
    print(product_df.shape)  # (1545, 12)

    review_df.to_csv("data/review.csv", index=False)
    product_df.to_csv("data/product.csv", index=False)
