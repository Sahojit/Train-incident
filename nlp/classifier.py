import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertForSequenceClassification, BertTokenizerFast, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm

from nlp.dataset import (
    TextClassificationDataset,
    CLS_LABEL2ID, CLS_ID2LABEL,
    generate_cls_samples,
)


MODEL_NAME = "bert-base-uncased"


class TrafficTextClassifier:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = len(CLS_LABEL2ID),
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=CLS_ID2LABEL,
            label2id=CLS_LABEL2ID,
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded classifier checkpoint from {checkpoint_path}")

    def train(
        self,
        texts: List[str],
        labels: List[int],
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
        val_split: float = 0.2,
        save_path: str = "models/cls_model.pt",
    ) -> Dict:
        dataset = TextClassificationDataset(texts, labels, self.tokenizer)
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

        best_val_acc = 0.0
        history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"CLS Train E{epoch}"):
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
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    val_loss += outputs.loss.item()
                    preds = outputs.logits.argmax(-1).cpu().tolist()
                    all_preds.extend(preds)
                    all_labels.extend(batch["labels"].cpu().tolist())

            acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)

            print(f"  Epoch {epoch}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  val_acc={acc:.4f}")
            history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val, "val_acc": acc})

            if acc > best_val_acc:
                best_val_acc = acc
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": self.model.state_dict()}, save_path)
                print(f"  Best classifier saved")

        return {"history": history, "best_val_acc": best_val_acc}

    def predict(self, text: str) -> Dict:
        self.model.eval()
        encoding = self.tokenizer(
            text, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoding).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        class_probs = {CLS_ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)}
        predicted_label = max(class_probs, key=class_probs.get)

        return {
            "text": text,
            "predicted_class": predicted_label,
            "probabilities": class_probs,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict(t) for t in texts]

    def evaluate(self, texts: List[str], labels: List[int]) -> str:
        preds = [CLS_LABEL2ID[self.predict(t)["predicted_class"]] for t in texts]
        return classification_report(
            labels, preds,
            target_names=list(CLS_LABEL2ID.keys()),
            zero_division=0,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--predict", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default="models/cls_model.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n_per_class", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clf = TrafficTextClassifier(checkpoint_path=args.checkpoint if not args.train else None)

    if args.train:
        texts, labels = generate_cls_samples(n_per_class=args.n_per_class)
        clf.train(texts, labels, epochs=args.epochs, save_path=args.checkpoint)

    if args.predict:
        result = clf.predict(args.predict)
        print(f"\nText: {result['text']}")
        print(f"Predicted class: {result['predicted_class']}")
        print("Probabilities:")
        for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            print(f"  {cls:<15}: {prob:.4f}")
