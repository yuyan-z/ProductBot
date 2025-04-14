from flask import Flask, request, jsonify, render_template

from graph import build_graph, sentiment_analyse
from utils import load_product_data
from vectordb import load_collection, do_query, format_query_results

app = Flask(__name__)

# Load resources
product_df = load_product_data()
collection = load_collection()
graph = build_graph()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()

    query_text = data.get("query_text", "")
    product_model = data.get("product_model", "llama3")
    review_model = data.get("review_model", "llama3")
    top_k = data.get("top_k", 10)
    use_analyser = data.get("use_analyser", True)

    # 1. Retrieve relevant products
    results = do_query(collection, query_text, top_k)
    results_formatted = format_query_results(product_df, query_text, results)

    # 2. Sentiment analysis if enabled
    if use_analyser:
        results_formatted = sentiment_analyse(results_formatted)

    # 3. Run the LangGraph pipeline
    state = {
        "query_text": query_text,
        "query_results": results_formatted,
        "product_model_name": product_model,
        "review_model_name": review_model
    }

    output = graph.invoke(state)
    return jsonify({
        "product_response": output["product_response"],
        "review_response": output["review_response"],
        "final_response": output["final_response"]
    })


if __name__ == "__main__":
    app.run(debug=True)
