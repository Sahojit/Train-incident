import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertForTokenClassification, BertTokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm

from nlp.dataset import (
    NERDataset, NERSample,
    NER_LABEL2ID, NER_ID2LABEL,
    generate_ner_sample,
)


MODEL_NAME = "bert-base-uncased"


class TrafficNERModel:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = len(NER_LABEL2ID),
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=NER_ID2LABEL,
            label2id=NER_LABEL2ID,
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded NER checkpoint from {checkpoint_path}")

    def train(
        self,
        samples: List[NERSample],
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
        val_split: float = 0.15,
        save_path: str = "models/ner_model.pt",
    ) -> Dict:
        dataset = NERDataset(samples, self.tokenizer)
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        best_val_loss = float("inf")
        history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"NER Train E{epoch}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            correct, total = 0, 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    val_loss += outputs.loss.item()
                    preds = outputs.logits.argmax(-1)
                    mask = batch["labels"] != -100
                    correct += (preds[mask] == batch["labels"][mask]).sum().item()
                    total += mask.sum().item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            acc = correct / total if total > 0 else 0

            print(f"  Epoch {epoch}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  val_acc={acc:.4f}")
            history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val, "val_acc": acc})

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": self.model.state_dict()}, save_path)
                print(f"  Best NER model saved")

        return {"history": history, "best_val_loss": best_val_loss}

    def predict(self, text: str) -> List[Dict]:
        words = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze(0).tolist()

        word_ids = encoding.word_ids(batch_index=0)
        entities = []
        current_entity = None

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            label = NER_ID2LABEL.get(predictions[token_idx], "O")

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"text": words[word_id], "label": label[2:], "start_word": word_id}
            elif label.startswith("I-") and current_entity:
                current_entity["text"] += " " + words[word_id]
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--predict", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default="models/ner_model.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n_samples", type=int, default=400)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ner = TrafficNERModel(checkpoint_path=args.checkpoint if not args.train else None)

    if args.train:
        samples = [generate_ner_sample() for _ in range(args.n_samples)]
        ner.train(samples, epochs=args.epochs, save_path=args.checkpoint)

    if args.predict:
        entities = ner.predict(args.predict)
        print(f"\nText: {args.predict}")
        print("Entities:")
        for ent in entities:
            print(f"  [{ent['label']}] {ent['text']}")
