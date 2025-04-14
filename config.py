import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
REVIEW_CSV_PATH = os.path.join(DATA_DIR, "review.csv")
PRODUCT_CSV_PATH = os.path.join(DATA_DIR, "product.csv")

DATABASE_DIR = os.path.join(BASE_DIR, "database")
VECTORDB_PATH = os.path.join(DATABASE_DIR, "vectordb")

LORA_MODEL_PATH = os.path.join(BASE_DIR, "train", "models", "lora_strong")

PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
