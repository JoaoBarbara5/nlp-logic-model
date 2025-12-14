import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
)


# ---------- 1. Data loading ----------

def load_reclor_style_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def to_hf_dataset(reclor_list: List[Dict[str, Any]]) -> Dataset:
    """
    Convert list of ReClor-style dicts to a HuggingFace Dataset
    with columns: id, context, question, choices, label
    """
    records = []
    for d in reclor_list:
        records.append(
            {
                "id": d["id_string"],
                "context": d["context"],
                "question": d["question"],
                "choices": d["answers"],
                "label": d.get("label", 0),  # test set may lack labels
            }
        )
    return Dataset.from_list(records)


# ---------- 2. Tokenization / preprocessing ----------

@dataclass
class MCPreprocessor:
    tokenizer: Any
    max_length: int = 256

    def __call__(self, examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # examples["context"] is a list of strings
        # examples["choices"] is a list of list-of-4 strings
        context_list = examples["context"]
        question_list = examples["question"]
        choices_list = examples["choices"]

        # Flatten: for each example, create 4 (context, question+choice) pairs
        batch_input_ids = []
        batch_attention_mask = []

        for context, question, choices in zip(context_list, question_list, choices_list):
            pair_input_ids = []
            pair_attention_mask = []

            for choice in choices:
                # text_a = context, text_b = question + " " + choice
                encoded = self.tokenizer(
                    context,
                    question + " " + choice,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
                pair_input_ids.append(encoded["input_ids"])
                pair_attention_mask.append(encoded["attention_mask"])

            batch_input_ids.append(pair_input_ids)
            batch_attention_mask.append(pair_attention_mask)

        result = {
            "input_ids": batch_input_ids,            # shape: (batch, num_choices, seq_len)
            "attention_mask": batch_attention_mask,  # shape: (batch, num_choices, seq_len)
        }

        if "label" in examples:
            result["labels"] = examples["label"]

        return result


# ---------- 3. Metric ----------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits shape: (batch_size, num_choices)
    preds = logits.argmax(axis=-1)
    accuracy = (preds == labels).mean().item()
    return {"accuracy": accuracy}


# ---------- 4. Main ----------

def main():
    # --- paths: point these to reclor data 
    train_path = "reclor_data/train.json"  
    val_path = "reclor_data/val.json"     

    # --- model name: change as needed "roberta-large", "bert-base-uncased", etc.
    model_name = "roberta-base" 

    # 4.1 Load data
    train_raw = load_reclor_style_json(train_path)
    val_raw = load_reclor_style_json(val_path)

    train_ds = to_hf_dataset(train_raw)
    val_ds = to_hf_dataset(val_raw)

    # 4.2 Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)

    preprocessor = MCPreprocessor(tokenizer=tokenizer, max_length=256)

    train_tokenized = train_ds.map(
        preprocessor,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tokenized = val_ds.map(
        preprocessor,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    # Set format for PyTorch
    train_tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    val_tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # 4.3 Training arguments
    training_args = TrainingArguments(
        output_dir="./mc_checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.06,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # 4.4 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 4.5 Train & evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    print("Validation results:", eval_results)




