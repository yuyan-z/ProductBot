import chromadb
import pandas as pd
from chromadb.api.models import Collection
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm

DATABASE_PATH = "database"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)

review_df = pd.read_csv("data/review.csv", dtype={0: str})
review_df["review_id"] = review_df.index.astype(str)
product_df = pd.read_csv("data/product.csv", dtype={0: str})


def load_collection() -> Collection:
    print("Loading collection...")
    chroma_client = chromadb.PersistentClient(path=DATABASE_PATH)
    collection = chroma_client.get_or_create_collection("review_collection")

    # Initialize collection if no data
    if collection.count() == 0:
        init_collection(collection)

    return collection


def init_collection(collection: Collection) -> None:
    print("Initializing collection...")
    # Add data in batches
    batch_size = 200
    doc_id_counter = 0
    for i in tqdm(range(0, len(review_df)  // 100, batch_size)):
        batch = review_df.iloc[i:i+batch_size]

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


def print_query_result(query_text: str, result: dict) -> None:
    print(f"-- Query --")
    print(query_text)

    print(f"-- Results --")
    print(result)
    n_results = len(result["ids"][0])
    for i in range(n_results):
        chunk_id = result["ids"][0][i]
        distance = result["distances"][0][i]
        chunk = result["documents"][0][i]
        metadata = result["metadatas"][0][i]
        product_id = metadata["product_id"]
        review_id = metadata["review_id"]
        product = product_df[product_df["product_id"] == product_id].iloc[0].to_dict()
        review = review_df[review_df["review_id"] == review_id].iloc[0].to_dict()

        print(f"chunk_id: {chunk_id}, distance: {distance:.4f}, review_id: {review_id}, product_id: {product_id}")
        print("\tchunk: ", chunk)
        print("\treview:" , review)
        print("\tproduct: ", product)
        print()


if __name__ == "__main__":
    collection = load_collection()
    query_text = "Can you recommend any makeup removers for oily skin?"

    results = collection.query(
        query_texts=[query_text],
        n_results=5
    )

    print_query_result(query_text, results)

