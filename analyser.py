import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


MODEL_NAME = "distilbert-base-uncased"
LORA_MODEL_PATH = "models/lora_strong"

class SentimentAnalyser:
    def __init__(self):
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        self.model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
        self.model.eval()

    def analyze(self, docs: list[str]) -> list[str]:
        sentiments = []

        for doc in docs:
            if not isinstance(doc, str):
                doc = str(doc)

            inputs = self.tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            pred = outputs.logits.argmax(dim=-1).item()
            sentiments.append(self.label_map.get(pred))

        return sentiments


if __name__ == "__main__":
    review_df = pd.read_csv("data/review.csv")

    analyser = SentimentAnalyser()
    reviews_sample = review_df.sample(10)
    review_texts = reviews_sample["review_text"].tolist()
    sentiments = analyser.analyze(review_texts)
    reviews_sample["sentiment"] = sentiments
    print(reviews_sample.head())

