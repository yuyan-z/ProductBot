import re

import pandas as pd
from langgraph.graph import StateGraph
from typing import TypedDict

from agents.analyser import SentimentAnalyser
from agents.product_agent import ProductAgent
from agents.review_agent import ReviewAgent
from query import load_collection, do_query, format_query_result


class AgentState(TypedDict):
    query_text: str
    query_result: dict
    product_response: str
    review_response: str
    final_response: str


# 1. ProductAgent Node
def run_product_agent(state: AgentState) -> AgentState:
    agent = ProductAgent("llama3")
    response = agent.generate_response(state["query_result"])
    return {**state, "product_response": response}


def extract_product_ids(response_text: str) -> str:
    ids = re.findall(r"\bP\d+\b", response_text)
    return ", ".join(ids) if ids else ""

# 2. ReviewAgent Node
def run_review_agent(state: AgentState) -> AgentState:
    agent = ReviewAgent("llama3")

    product_response = extract_product_ids(state['product_response'])
    print("Product IDs in response", product_response)

    response = agent.generate_response(
        query_result=state["query_result"],
        product_response=product_response
    )
    return {**state, "review_response": response}


# 3. Merge Results
def merge_results(state: AgentState) -> str:
    final_response = (
        f"{state['product_response']}\n\n"
        f"{state['review_response']}"
    )
    return {**state, "final_response": final_response}

builder = StateGraph(AgentState)
builder.add_node("product_agent", run_product_agent)
builder.add_node("review_agent", run_review_agent)
builder.add_node("merge", merge_results)

builder.set_entry_point("product_agent")
builder.add_edge("product_agent", "review_agent")
builder.add_edge("review_agent", "merge")
builder.set_finish_point("merge")

graph = builder.compile()


def sentiment_analyse(query_result: dict) -> dict:
    analyser = SentimentAnalyser()
    review_texts = [r["review_text"] for r in query_result["reviews"]]
    sentiments = analyser.analyze(review_texts)
    for r, s in zip(query_result["reviews"], sentiments):
        r["sentiment"] = s
    return query_result


if __name__ == "__main__":
    product_df = pd.read_csv("data/product.csv")
    collection = load_collection()

    query_text = "What are the best makeup removers for oily skin under $30?"

    result = do_query(collection, query_text, 10)
    result_formatted = format_query_result(product_df, query_text, result)
    print("Retrieved results", result_formatted)

    useAnalyser = True
    if useAnalyser:
        result_formatted = sentiment_analyse(result_formatted)
        print("Reviews with sentiments", result_formatted["reviews"])

    output = graph.invoke({
        "query_text": query_text,
        "query_result": result_formatted
    })

    print("-- Final response ====")
    print(output["final_response"])
