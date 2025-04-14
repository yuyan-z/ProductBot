import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from config import PROMPTS_DIR
from vectordb import load_collection, do_query, format_query_results
from utils import load_json, load_review_data, load_product_data

PROMPT_PATH = os.path.join(PROMPTS_DIR, "review_agent.json")


class ReviewAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.prompt_data = load_json(PROMPT_PATH)
        self.select_model()

    def select_model(self):
        if self.model_name == "llama3":
            self.model = OllamaLLM(model="llama3")
        else:
            raise NotImplementedError

    def generate_response(self, query_results: dict, product_response: str = ''):
        if self.model_name == "llama3":
            response = self._generate_response_llama(query_results, product_response)
        else:
            raise NotImplementedError

        return response

    def _generate_response_llama(self, query_results: dict, product_response: str):
        print(f"Generating response with {self.model_name}...")

        output_parser = StrOutputParser()

        messages = [
            ("system", self.prompt_data[0]["system"]),
            ("user", "User Question: {user_query}"),
        ]
        reviews = query_results["reviews"]
        for i, review in enumerate(reviews):
            review_str = "\n".join([f"{key}: {value}" for key, value in review.items()])
            messages.append(("user", f"Review:\n{review_str}"))

        if product_response != '':
            messages.append(("user",
                             f"Please prioritize the reviews for the following recommended products: \n{product_response}"))

        messages.append(("user", self.prompt_data[0]["user"]))
        template = ChatPromptTemplate.from_messages(messages)

        chain = (
                {"user_query": lambda x: x["user_query"]}
                | template
                | self.model
                | output_parser
        )

        response = chain.invoke({
            "user_query": query_results["user_query"],
            "reviews": query_results["reviews"]
        })

        return response


if __name__ == "__main__":
    review_df = load_review_data()
    product_df = load_product_data()

    collection = load_collection()
    print("collection count:", collection.count())

    query_text = "What are the best makeup removers for oily skin under $30?"

    results = do_query(collection, query_text, 10)
    results_formatted = format_query_results(product_df, query_text, results)
    print(results_formatted["reviews"])

    review_agent = ReviewAgent("llama3")
    response = review_agent.generate_response(results_formatted)
    print("-- Response --")
    print(response)
