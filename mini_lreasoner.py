import json
import argparse
import os
import ast
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, BertConfig, BertModel, get_linear_schedule_with_warmup 
from sklearn.model_selection import StratifiedKFold


@dataclass
class ReclorExample:
    """Single ReClor / LReasoner-style example."""
    qid: str
    context: str
    question: str
    options: List[str]
    label: Optional[int] = None


def load_reclor_json(path: str) -> List[ReclorExample]:
    """Load ReClor-style JSON (same schema as HF reclor + LReasoner reclor-data)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[ReclorExample] = []
    for item in data:
        # LReasoner / ReClor keys: context, question, answers, label, id_string
        context = item["context"]
        question = item["question"]
        options = item.get("answers") or item.get("options")
        if not options:
            raise ValueError("Expected key 'answers' or 'options' in JSON.")

        label = item.get("label")
        # Different repos use "id", "id_string", or "idx"
        qid = str(item.get("id_string", item.get("id", item.get("idx", len(examples)))))

        examples.append(
            ReclorExample(
                qid=qid,
                context=context,
                question=question,
                options=list(options),
                label=label,
            )
        )
    return examples


def load_reclor_csv(path: str) -> List[ReclorExample]:
    """Load a CSV like test.csv (id, context, question, answers, [label])."""
    import pandas as pd

    df = pd.read_csv(path)
    examples: List[ReclorExample] = []
    has_label = "label" in df.columns

    for _, row in df.iterrows():
        raw_answers = row["answers"]
        if isinstance(raw_answers, str):
            # In baseline files answers is a stringified Python list
            options = ast.literal_eval(raw_answers)
        else:
            options = list(raw_answers)

        examples.append(
            ReclorExample(
                qid=str(row["id"]),
                context=str(row["context"]),
                question=str(row["question"]),
                options=list(options),
                label=int(row["label"]) if has_label and not pd.isna(row["label"]) else None,
            )
        )
    return examples


class ReclorDataset(Dataset):
    """Multiple-choice dataset for ReClor/LReasoner-style JSON/CSV."""

    def __init__(self, examples: List[ReclorExample], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]

        num_choices = len(ex.options)
        # Following the typical encoding used in ReClor/LReasoner:
        # [CLS] context [SEP] question [SEP] option [SEP]
        texts = [ex.context] * num_choices
        text_pairs = [ex.question + " " + self.tokenizer.sep_token + " " + opt for opt in ex.options]

        encoded = self.tokenizer(
            texts,
            text_pairs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"],  # (num_choices, L)
            "attention_mask": encoded["attention_mask"],
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"]
        else:
            # Some tokenizers (RoBERTa) don't use token_type_ids
            item["token_type_ids"] = torch.zeros_like(encoded["input_ids"])

        item["labels"] = torch.tensor(-1 if ex.label is None else ex.label, dtype=torch.long)
        item["qid"] = ex.qid
        return item


class MiniLReasoner(torch.nn.Module):
    """
    A very simple LReasoner-style model:
    - backbone: BERT-style encoder (randomly initialized by default)
    - head: linear layer on [CLS] for each option
    - trained only with multiple-choice cross-entropy (no contrastive loss, no symbolic module)
    """

    def __init__(self, backbone_name: str = "bert-base-uncased", init_from_pretrained: bool = False):
        super().__init__()
        if init_from_pretrained:
            self.config = BertConfig.from_pretrained(backbone_name)
            self.bert = BertModel.from_pretrained(backbone_name, config=self.config)
        else:
            # Use the config of a standard BERT but *randomly initialize* weights
            self.config = BertConfig.from_pretrained(backbone_name)
            self.bert = BertModel(self.config)

        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.config.hidden_size, 1)  # score per option

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        # input_ids: (batch_size, num_choices, seq_len)
        bsz, num_choices, seq_len = input_ids.size()

        flat_input_ids = input_ids.view(bsz * num_choices, seq_len)
        flat_attention_mask = attention_mask.view(bsz * num_choices, seq_len)
        flat_token_type_ids = token_type_ids.view(bsz * num_choices, seq_len)

        outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
        )
        pooled = outputs.pooler_output  # (bsz*num_choices, hidden)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (bsz*num_choices, 1)
        logits = logits.view(bsz, num_choices)  # (bsz, num_choices)

        loss = None
        if labels is not None and (labels >= 0).any():
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return logits, loss


def collate_fn(batch):
    # batch is list of dicts from ReclorDataset.__getitem__
    qids = [item["qid"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])  # (B,)

    input_ids = torch.stack([item["input_ids"] for item in batch])  # (B, C, L)
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    token_type_ids = torch.stack([item["token_type_ids"] for item in batch])

    return {
        "qid": qids,
        "labels": labels,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_logits = []
    all_qids = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            logits, _ = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                labels=None,
            )
            preds = logits.argmax(dim=-1)  # (B,)
            mask = labels >= 0
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
            all_logits.append(logits.cpu())
            all_qids.extend(batch["qid"])

    acc = correct / total if total > 0 else 0.0
    return acc, all_qids, torch.cat(all_logits, dim=0)


def train(
    model,
    train_loader,
    dev_loader,
    device,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 5,
    warmup_ratio: float = 0.1,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    output_dir: str = "./mini_lreasoner_ckpt",
):
    os.makedirs(output_dir, exist_ok=True)

    t_total = num_epochs * len(train_loader) // grad_accum_steps
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(warmup_ratio * t_total), num_training_steps=t_total
    )

    model.to(device)
    global_step = 0
    best_dev_acc = 0.0
    best_ckpt_path = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            logits, loss = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = loss / grad_accum_steps
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        dev_acc, _, _ = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} | train_loss={running_loss:.4f} | dev_acc={dev_acc:.4f}")

        # Save best checkpoint
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model.config.to_dict(),
                },
                best_ckpt_path,
            )
            print(f"  -> New best dev_acc={best_dev_acc:.4f}, saving to {best_ckpt_path}")

    print(f"Training finished. Best dev_acc={best_dev_acc:.4f} ({best_ckpt_path})")
    return best_dev_acc, best_ckpt_path


def predict_on_test(model, test_loader, device, output_csv: str):
    model.to(device)
    model.eval()
    all_qids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            logits, _ = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                labels=None,
            )
            preds = logits.argmax(dim=-1)
            all_qids.extend(batch["qid"])
            all_preds.extend(preds.cpu().tolist())

    import pandas as pd

    df = pd.DataFrame({"id": all_qids, "label": all_preds})
    df.to_csv(output_csv, index=False)
    print(f"Wrote predictions to {output_csv}")


def predict_logits(model, dataloader, device):
    """Return logits and qids for a dataloader (used for fold ensembling)."""
    model.to(device)
    model.eval()
    all_qids = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            logits, _ = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                labels=None,
            )
            all_qids.extend(batch["qid"])
            all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0), all_qids


def main():
    parser = argparse.ArgumentParser(description="Mini LReasoner for ReClor (simplified)")
    default_train = "aml-2025-read-between-the-lines/train.csv"
    default_test = "test.csv"
    parser.add_argument(
        "--train_file",
        type=str,
        default=default_train,
        help=f"Path to training file (csv or json). Default: {default_train}",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default=None,
        help="Optional dev/validation file. If omitted, K-fold CV is used.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=default_test,
        help=f"Optional test file for prediction. Default: {default_test}",
    )
    parser.add_argument(
        "--file_format",
        type=str,
        default="csv",
        choices=["json", "csv"],
        help="Input file format (json for LReasoner reclor-data, csv for Kaggle-style).",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="bert-base-uncased",
        help="Backbone name (used for tokenizer + config, like in LReasoner).",
    )
    parser.add_argument(
        "--init_from_pretrained",
        action="store_true",
        help="If set, initialize BERT weights from a pretrained checkpoint instead of random.",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./mini_lreasoner_ckpt")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for CV when dev_file is not provided")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for CV shuffling")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_name, use_fast=True)

    if args.file_format == "json":
        train_examples = load_reclor_json(args.train_file)
        dev_examples = load_reclor_json(args.dev_file) if args.dev_file else None
        test_examples = load_reclor_json(args.test_file) if args.test_file else None
    else:
        train_examples = load_reclor_csv(args.train_file)
        dev_examples = load_reclor_csv(args.dev_file) if args.dev_file else None
        test_examples = load_reclor_csv(args.test_file) if args.test_file else None

    test_dataset = ReclorDataset(test_examples, tokenizer, max_length=args.max_length) if test_examples else None
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
        if test_dataset
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dev_examples is not None:
        # Standard train/dev training when an explicit dev file is supplied.
        train_dataset = ReclorDataset(train_examples, tokenizer, max_length=args.max_length)
        dev_dataset = ReclorDataset(dev_examples, tokenizer, max_length=args.max_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

        model = MiniLReasoner(backbone_name=args.backbone_name, init_from_pretrained=args.init_from_pretrained)
        best_dev_acc, best_ckpt_path = train(
            model,
            train_loader,
            dev_loader,
            device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            warmup_ratio=args.warmup_ratio,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            output_dir=args.output_dir,
        )

        # Reload best checkpoint for test-time prediction (if requested)
        if best_ckpt_path and test_loader is not None:
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            predict_on_test(model, test_loader, device, os.path.join(args.output_dir, "submission.csv"))
    else:
        # K-fold cross-validation when no explicit dev file is provided.
        labels = [ex.label for ex in train_examples]
        if any(label is None for label in labels):
            raise ValueError("Training data must include labels for cross-validation.")

        skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        fold_accs = []
        test_logits_sum = None
        cached_test_qids = None

        for fold_id, (train_idx, dev_idx) in enumerate(skf.split(range(len(train_examples)), labels)):
            print(f"\n===== Fold {fold_id + 1}/{args.num_folds} =====")
            fold_train_examples = [train_examples[i] for i in train_idx]
            fold_dev_examples = [train_examples[i] for i in dev_idx]

            train_dataset = ReclorDataset(fold_train_examples, tokenizer, max_length=args.max_length)
            dev_dataset = ReclorDataset(fold_dev_examples, tokenizer, max_length=args.max_length)

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=collate_fn,
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_fn,
            )

            model = MiniLReasoner(backbone_name=args.backbone_name, init_from_pretrained=args.init_from_pretrained)
            fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_id}")
            best_dev_acc, best_ckpt_path = train(
                model,
                train_loader,
                dev_loader,
                device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_epochs=args.num_epochs,
                warmup_ratio=args.warmup_ratio,
                grad_accum_steps=args.grad_accum_steps,
                max_grad_norm=args.max_grad_norm,
                output_dir=fold_output_dir,
            )

            fold_accs.append(best_dev_acc)

            if best_ckpt_path and test_loader is not None:
                ckpt = torch.load(best_ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                logits, qids = predict_logits(model, test_loader, device)
                cached_test_qids = cached_test_qids or qids
                test_logits_sum = logits if test_logits_sum is None else test_logits_sum + logits

        if fold_accs:
            mean_acc = sum(fold_accs) / len(fold_accs)
            print(f"\nCV finished | mean_dev_acc={mean_acc:.4f} | per_fold={fold_accs}")

        if test_loader is not None and test_logits_sum is not None:
            import pandas as pd

            avg_logits = test_logits_sum / len(fold_accs)
            preds = avg_logits.argmax(dim=-1)
            submission_path = os.path.join(args.output_dir, "submission_cv.csv")
            os.makedirs(args.output_dir, exist_ok=True)
            df = pd.DataFrame({"id": cached_test_qids, "label": preds.tolist()})
            df.to_csv(submission_path, index=False)
            print(f"Wrote CV-averaged predictions to {submission_path}")


if __name__ == "__main__":
    main()
