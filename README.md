# ProductBot

## Workflow
User Input → Embeddings  
→ Search for the most similar review_text from ChromaDB  
→ Pass the retrieved results to product_agent  
→ Pass the retrieved results and the filtered products from product_agent to review_agent  
→ Merge and process final results  

## Getting Started
1. `docker build -t product-app`
2. `docker run -p 5000:5000 product-app`

## Example

