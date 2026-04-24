"""
NLP preprocessing pipeline using spaCy.
Handles tokenization, POS tagging, dependency parsing, and feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import spacy
from spacy.tokens import Doc, Token


# Load once at module level; download with: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )


@dataclass
class TokenInfo:
    text: str
    lemma: str
    pos: str
    tag: str
    dep: str
    head: str
    is_stop: bool
    is_alpha: bool


@dataclass
class ParsedDocument:
    original_text: str
    cleaned_text: str
    tokens: List[TokenInfo]
    sentences: List[str]
    noun_chunks: List[str]
    named_entities: List[Dict[str, str]]


def clean_text(text: str) -> str:
    """Basic text normalisation before spaCy processing."""
    import re
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?'-]", "", text)
    return text


def extract_token_info(token: Token) -> TokenInfo:
    return TokenInfo(
        text=token.text,
        lemma=token.lemma_,
        pos=token.pos_,
        tag=token.tag_,
        dep=token.dep_,
        head=token.head.text,
        is_stop=token.is_stop,
        is_alpha=token.is_alpha,
    )


def parse_document(text: str) -> ParsedDocument:
    """Run full spaCy pipeline and extract structured information."""
    cleaned = clean_text(text)
    doc: Doc = nlp(cleaned)

    tokens = [extract_token_info(t) for t in doc]

    sentences = [sent.text.strip() for sent in doc.sents]

    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    named_entities = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]

    return ParsedDocument(
        original_text=text,
        cleaned_text=cleaned,
        tokens=tokens,
        sentences=sentences,
        noun_chunks=noun_chunks,
        named_entities=named_entities,
    )


def extract_keywords(text: str, pos_filter: Optional[List[str]] = None) -> List[str]:
    """Extract content words (nouns, verbs, adjectives by default)."""
    if pos_filter is None:
        pos_filter = ["NOUN", "VERB", "ADJ", "PROPN"]

    doc = nlp(clean_text(text))
    return [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in pos_filter and not token.is_stop and token.is_alpha and len(token.text) > 1
    ]


def get_dependency_triples(text: str) -> List[Dict[str, str]]:
    """Extract subject-verb-object triples via dependency parsing."""
    doc = nlp(clean_text(text))
    triples = []
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            subject = next((t.text for t in token.lefts if t.dep_ in ("nsubj", "nsubjpass")), None)
            obj = next((t.text for t in token.rights if t.dep_ in ("dobj", "pobj", "attr")), None)
            if subject or obj:
                triples.append({"subject": subject, "verb": token.text, "object": obj})
    return triples


def batch_preprocess(texts: List[str]) -> List[ParsedDocument]:
    return [parse_document(t) for t in texts]


def document_to_dict(doc: ParsedDocument) -> Dict:
    return {
        "original_text": doc.original_text,
        "cleaned_text": doc.cleaned_text,
        "sentences": doc.sentences,
        "noun_chunks": doc.noun_chunks,
        "named_entities": doc.named_entities,
        "tokens": [
            {"text": t.text, "lemma": t.lemma, "pos": t.pos, "dep": t.dep}
            for t in doc.tokens
        ],
    }


if __name__ == "__main__":
    sample = "A major accident occurred on NH-8 near Sector 62 causing heavy traffic congestion."
    parsed = parse_document(sample)
    print("Sentences:", parsed.sentences)
    print("Noun chunks:", parsed.noun_chunks)
    print("Named entities:", parsed.named_entities)
    print("Keywords:", extract_keywords(sample))
    print("Dep triples:", get_dependency_triples(sample))
