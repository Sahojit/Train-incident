"""
Datasets for:
  1. BERT-based NER fine-tuning (token classification)
  2. Text classification (accident / jam / road_closure / normal)

Includes synthetic data generators for both tasks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


# -----------------------------------------------------------------------
# Label maps
# -----------------------------------------------------------------------

NER_LABEL2ID: Dict[str, int] = {
    "O": 0,
    "B-LOCATION": 1,
    "I-LOCATION": 2,
    "B-INCIDENT_TYPE": 3,
    "I-INCIDENT_TYPE": 4,
    "B-SEVERITY": 5,
    "I-SEVERITY": 6,
}
NER_ID2LABEL = {v: k for k, v in NER_LABEL2ID.items()}

CLS_LABEL2ID: Dict[str, int] = {
    "accident": 0,
    "jam": 1,
    "road_closure": 2,
    "normal": 3,
}
CLS_ID2LABEL = {v: k for k, v in CLS_LABEL2ID.items()}


# -----------------------------------------------------------------------
# NER dataset (token classification)
# -----------------------------------------------------------------------

@dataclass
class NERSample:
    tokens: List[str]
    ner_tags: List[str]


class NERDataset(Dataset):
    def __init__(
        self,
        samples: List[NERSample],
        tokenizer: BertTokenizerFast,
        max_length: int = 128,
        label2id: Dict[str, int] = NER_LABEL2ID,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample.tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Special tokens → ignored in loss
            elif word_id != prev_word_id:
                label_ids.append(self.label2id.get(sample.ner_tags[word_id], 0))
            else:
                # Continuation sub-token: use I- tag if B- was used, else -100
                tag = sample.ner_tags[word_id]
                if tag.startswith("B-"):
                    label_ids.append(self.label2id.get("I-" + tag[2:], 0))
                else:
                    label_ids.append(self.label2id.get(tag, 0))
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


# -----------------------------------------------------------------------
# Classification dataset
# -----------------------------------------------------------------------

class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizerFast,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# -----------------------------------------------------------------------
# Synthetic data for testing
# -----------------------------------------------------------------------

LOCATIONS = ["Sector 62", "NH-8", "MG Road", "Ring Road", "Highway 48", "Outer Ring Road", "Nehru Place"]
INCIDENT_TYPES = ["accident", "collision", "pile-up", "breakdown", "road closure"]
SEVERITIES = ["minor", "major", "severe", "heavy", "light"]

CLS_TEMPLATES = {
    "accident": [
        "A {sev} {inc} occurred near {loc}.",
        "Vehicle {inc} reported at {loc} causing traffic disruption.",
        "{sev} {inc} on {loc} involving multiple vehicles.",
    ],
    "jam": [
        "Heavy traffic congestion on {loc} due to peak hours.",
        "Long queues reported near {loc}, expect delays.",
        "Slow moving traffic on {loc} stretch.",
    ],
    "road_closure": [
        "{loc} closed for maintenance until further notice.",
        "Road closure reported on {loc} due to construction.",
        "Complete shutdown of {loc} due to security operations.",
    ],
    "normal": [
        "Traffic flowing smoothly near {loc}.",
        "No incidents reported on {loc}.",
        "Normal traffic conditions on {loc}.",
    ],
}


def generate_ner_sample(label: str = "accident") -> NERSample:
    loc = random.choice(LOCATIONS)
    inc = random.choice(INCIDENT_TYPES)
    sev = random.choice(SEVERITIES)

    tokens = ["A", sev, inc, "occurred", "near"] + loc.split() + ["."]
    tags = (
        ["O"]
        + ["B-SEVERITY"] + (["I-SEVERITY"] * (len(sev.split()) - 1))
        + ["B-INCIDENT_TYPE"] + (["I-INCIDENT_TYPE"] * (len(inc.split()) - 1))
        + ["O", "O"]
        + ["B-LOCATION"] + ["I-LOCATION"] * (len(loc.split()) - 1)
        + ["O"]
    )
    return NERSample(tokens=tokens, ner_tags=tags)


def generate_cls_samples(n_per_class: int = 50) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    for label, templates in CLS_TEMPLATES.items():
        for _ in range(n_per_class):
            tmpl = random.choice(templates)
            text = tmpl.format(
                loc=random.choice(LOCATIONS),
                inc=random.choice(INCIDENT_TYPES),
                sev=random.choice(SEVERITIES),
            )
            texts.append(text)
            labels.append(CLS_LABEL2ID[label])
    return texts, labels


if __name__ == "__main__":
    sample = generate_ner_sample()
    print("NER sample:", list(zip(sample.tokens, sample.ner_tags)))

    texts, labels = generate_cls_samples(n_per_class=2)
    for t, l in zip(texts, labels):
        print(f"[{CLS_ID2LABEL[l]}] {t}")
