import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from query import load_collection, do_query, format_query_result
from utils import load_json

PROMPT_DATA_PATH = "../prompts/review_agent.json"


class ReviewAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.prompt_data = load_json(PROMPT_DATA_PATH)
        self.select_model()

    def select_model(self):
        if self.model_name == "llama3":
            self.model = OllamaLLM(model="llama3")
        else:
            raise NotImplementedError

    def generate_response(self, query_result, product_response: str = ''):
        if self.model_name == "llama3":
            response = self._generate_response_llama(query_result)
        else:
            raise NotImplementedError

        return response

    def _generate_response_llama(self, query_result, response_products: str = ''):
        print("Generating response with LLama...")

        output_parser = StrOutputParser()

        messages = [
            ("system", self.prompt_data[0]["system"]),
            ("user", "User Question: {user_query}"),
        ]
        reviews = query_result["reviews"]
        for i, review in enumerate(reviews):
            review_text = "\n".join([f"{key}: {value}" for key, value in review.items()])
            messages.append(("user", f"Review:\n{review_text}"))

        if response_products != '':
            messages.append(("user", f"Please prioritize the reviews for the following recommended products: \n{response_products }"))

        messages.append(("user", self.prompt_data[0]["user"]))
        template = ChatPromptTemplate.from_messages(messages)

        chain = (
                {"user_query": lambda x: x["user_query"]}
                | template
                | self.model
                | output_parser
        )

        response = chain.invoke({
            "user_query": query_result["user_query"],
            "reviews": query_result["reviews"]
        })

        return response


if __name__ == "__main__":
    review_df = pd.read_csv("../data/review.csv")
    product_df = pd.read_csv("../data/product.csv")

    collection = load_collection()
    print("collection count:", collection.count())

    query_text = "What are the best makeup removers for oily skin under $30?"

    result = do_query(collection, query_text, 10)
    result_formatted = format_query_result(product_df, query_text, result)
    print(result_formatted["reviews"])

    review_agent = ReviewAgent("llama3")
    response = review_agent.generate_response(result_formatted)
    print("-- Response --")
    print(response)