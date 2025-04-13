import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DistilBertForSequenceClassification, EarlyStoppingCallback
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = '../data/review.csv'


def rating_to_sentiment(r: int) -> int:
    if r in [1, 2]:
        return 0  # negative
    elif r == 3:
        return 1  # neutral
    else:
        return 2  # positive


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["review_text"], padding="max_length", truncation=True, max_length=256)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df["labels"] = df["rating"].apply(rating_to_sentiment)
    df = df[["review_text", "labels"]].dropna()

    dataset = Dataset.from_pandas(df)
    train_valtest = dataset.train_test_split(test_size=0.3, seed=42)
    val_test = train_valtest['test'].train_test_split(test_size=0.5, seed=42)
    dataset_dict = DatasetDict({
        'train': train_valtest['train'],
        'valid': val_test['train'],
        'test': val_test['test']
    })
    return dataset_dict


def train_lora_model(config, encoded_dataset, base_model):
    print(f"Training LoRA config: {config['name']}")

    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["alpha"],
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(base_model, lora_config)

    output_dir = f"./results/{config['name']}"
    logging_dir = f"./logs/{config['name']}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="accuracy",
        report_to="tensorboard",
        logging_dir=logging_dir,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    trainer.save_model(f"./models/{config['name']}")


if __name__ == "__main__":
    with open("train_config.json", "r") as f:
        config_list = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset_dict = load_dataset()
    encoded_dataset = dataset_dict.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    for config in config_list:
        train_lora_model(config, encoded_dataset, base_model)
