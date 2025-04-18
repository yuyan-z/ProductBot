import os

import chromadb
import pandas as pd
from chromadb.api.models import Collection
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm

from config import VECTORDB_PATH
from utils import load_review_data, load_product_data

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)


def load_collection() -> Collection:
    print("Loading collection...")
    chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = chroma_client.get_or_create_collection("review_collection")

    # Initialize collection if no data
    if collection.count() == 0:
        init_collection(collection)

    return collection


def init_collection(collection: Collection) -> None:
    review_df = load_review_data()
    print("Initializing collection...")
    # Add data in batches
    batch_size = 200
    doc_id_counter = 0
    for i in tqdm(range(0, len(review_df), batch_size)):
        batch = review_df.iloc[i:i + batch_size]

        # split document to chunks
        documents = [
            Document(
                page_content=row["review_text"],
                metadata={
                    "product_id": row["product_id"],
                    "review_id": row["review_id"],
                })
            for _, row in batch.iterrows()
        ]
        chunks = text_splitter.split_documents(documents)

        collection.add(
            documents=[doc.page_content for doc in chunks],
            ids=[f"{doc_id_counter + j}" for j in range(len(chunks))],
            metadatas=[doc.metadata for doc in chunks]
        )

        doc_id_counter += len(chunks)


def print_query_results(review_df: pd.DataFrame, product_df: pd.DataFrame, query_text: str, results: dict) -> None:
    print(f"-- Query --")
    print(query_text)

    print(f"-- Results --")
    print(results)
    print()

    n_results = len(results["ids"][0])
    for i in range(n_results):
        chunk_id = results["ids"][0][i]
        distance = results["distances"][0][i]
        chunk = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        product_id = metadata["product_id"]
        review_id = metadata["review_id"]
        product = product_df[product_df["product_id"] == product_id].iloc[0].to_dict()
        review = review_df[review_df["review_id"] == review_id].iloc[0].to_dict()

        print(f"chunk_id: {chunk_id}, distance: {distance:.4f}, review_id: {review_id}, product_id: {product_id}")
        print("\tchunk: ", chunk)
        print("\treview:", review)
        print("\tproduct: ", product)
        print()


def format_query_results(product_df: pd.DataFrame, query_text: str, results: dict) -> dict:
    results_formatted = {
        "user_query": query_text,
        "products": [],
        "reviews": []
    }

    n_results = len(results["ids"][0])
    for i in range(n_results):
        review = {}
        review["review_text"] = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        product_id = metadata["product_id"]
        product = product_df[product_df["product_id"] == product_id].iloc[0].to_dict()
        results_formatted["products"].append(product)
        review["product_id"] = product_id
        review["product_name"] = product["product_name"]
        results_formatted["reviews"].append(review)

    return results_formatted


def do_query(collection: Collection, query_text: str, n_results: int) -> dict:
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results


if __name__ == "__main__":
    review_df = load_review_data()
    product_df = load_product_data()
    collection = load_collection()
    query_text = "What is the best makeup remover for oily skin under $30?"
    results = do_query(collection, query_text, 2)
    print_query_results(review_df, product_df, query_text, results)
