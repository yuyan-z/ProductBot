import re
from langgraph.graph import StateGraph
from typing import TypedDict

from agents.analyser import SentimentAnalyser
from agents.product_agent import ProductAgent
from agents.review_agent import ReviewAgent
from utils import load_product_data
from vectordb import load_collection, do_query, format_query_results


class AgentState(TypedDict):
    query_text: str
    query_results: dict
    product_response: str
    review_response: str
    final_response: str
    product_model_name: str
    review_model_name: str


# 1. ProductAgent Node
def run_product_agent(state: AgentState) -> AgentState:
    agent = ProductAgent(state["product_model_name"])
    response = agent.generate_response(state["query_results"])
    return {**state, "product_response": response}


def extract_product_ids(response_text: str) -> str:
    ids = re.findall(r"\bP\d+\b", response_text)
    return ", ".join(ids) if ids else ""

# 2. ReviewAgent Node
def run_review_agent(state: AgentState) -> AgentState:
    agent = ReviewAgent(state["review_model_name"])

    product_response = extract_product_ids(state['product_response'])
    print("-- Product IDs in response --\n", product_response)

    response = agent.generate_response(
        query_results=state["query_results"],
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


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("product_agent", run_product_agent)
    builder.add_node("review_agent", run_review_agent)
    builder.add_node("merge", merge_results)

    builder.set_entry_point("product_agent")
    builder.add_edge("product_agent", "review_agent")
    builder.add_edge("review_agent", "merge")
    builder.set_finish_point("merge")

    graph = builder.compile()
    # print(graph.get_graph().draw_mermaid())
    return graph

def sentiment_analyse(query_result: dict) -> dict:
    analyser = SentimentAnalyser()
    review_texts = [r["review_text"] for r in query_result["reviews"]]
    sentiments = analyser.analyze(review_texts)
    for r, s in zip(query_result["reviews"], sentiments):
        r["sentiment"] = s
    return query_result


if __name__ == "__main__":
    product_df = load_product_data()
    collection = load_collection()

    query_text = "What are the best makeup removers for oily skin under $30?"

    results = do_query(collection, query_text, 10)
    results_formatted = format_query_results(product_df, query_text, results)
    print("-- Retrieved results --\n", results_formatted)

    useAnalyser = True
    if useAnalyser:
        results_formatted = sentiment_analyse(results_formatted)
        print("-- Reviews with sentiments --\n", results_formatted["reviews"])

    graph = build_graph()
    output = graph.invoke({
        "query_text": query_text,
        "query_results": results_formatted,
        "product_model_name": "llama3",
        "review_model_name": "llama3"
    })

    print("-- Final response --")
    print(output["final_response"])
