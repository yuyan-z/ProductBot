import json
import os

import kagglehub
import pandas as pd

os.environ["KAGGLEHUB_CACHE"] = "."
DATASET_DIR = 'datasets/nadyinky/sephora-products-and-skincare-reviews/versions/2'


def download_dataset() -> None:
    path = kagglehub.dataset_download("nadyinky/sephora-products-and-skincare-reviews")
    print("Downloaded dataset:", path)


def load_review_data() -> pd.DataFrame:
    print("Loading review data...")
    filenames = [
        filename for filename in os.listdir(DATASET_DIR)
        if filename.startswith('reviews') and filename.endswith('.csv')
    ]
    dfs = [
        pd.read_csv(os.path.join(DATASET_DIR, file), low_memory=False, index_col=0)
        for file in filenames
    ]
    df = pd.concat(dfs, ignore_index=True)
    return df


def load_product_data() -> pd.DataFrame:
    print("Loading product data...")
    df = pd.read_csv(os.path.join(DATASET_DIR, "product_info.csv"))
    return df


def load_json(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding {f}: {e}")
    return data


if __name__ == "__main__":
    download_dataset()

    review_df = load_review_data()
    # print("review_df.shape:", review_df.shape)
    product_df = load_product_data()
    # print("product_df.shape:", df.shape)




