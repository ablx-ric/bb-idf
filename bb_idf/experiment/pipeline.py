"""Corpus loading, preprocessing, and shared term-document representation.

All three algorithms (TF-IDF, BB-IDF, TextRank) consume the SAME token lists
and the SAME vocabulary, guaranteeing a fair, controlled comparison.

The expensive spaCy preprocessing is materialized ONCE and cached to disk
(``data/processed/preprocessed.pkl``); the algorithms and the analysis consume
that artifact via ``preprocess_corpus`` instead of re-tokenizing.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import spacy
import fitz

# English function words that leak into the (mostly Spanish) corpus through
# bilingual abstracts and the English article. spaCy's Spanish stop list does
# not cover them. Applied identically to documents and gold keywords.
_ENGLISH_STOPWORDS = {
    "the", "and", "for", "was", "were", "with", "this", "that", "these",
    "those", "from", "are", "but", "not", "which", "their", "have", "has",
    "been", "they", "them", "into", "than", "then", "also", "can", "will",
    "would", "should", "could", "about", "after", "before", "between",
}


@functools.lru_cache(maxsize=1)
def _nlp():
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    nlp.max_length = 20_000_000
    return nlp


def load_documents(corpus_dir: str | Path) -> list[tuple[str, str]]:
    """Return (filename, text) for every top-level PDF, in sorted order."""
    path = Path(corpus_dir)
    # Sort case-sensitively by string path so the order is deterministic and
    # independent of platform (Path comparison is case-insensitive on Windows).
    files = sorted(path.glob("*.pdf"), key=str)
    docs: list[tuple[str, str]] = []
    for f in files:
        try:
            doc = fitz.open(f)
            text = "".join(page.get_text() for page in doc)
            doc.close()
        except Exception as exc:  # pragma: no cover
            print(f"WARN: no se pudo leer {f.name} ({exc})")
            continue
        docs.append((f.name, text))
    return docs


def preprocess(text: str, nlp) -> list[str]:
    """Lemmatize and filter a document into content tokens (lowercased).

    Filter: no punctuation, no whitespace, no stopwords, no numeric-like
    tokens, alphabetic lemma only (drops digits/symbols/URLs/ISSN fragments),
    and original token length >= 3. Mirrors the project's Preprocessor plus an
    alphabetic filter to remove journal-metadata noise (page ranges, DOIs,
    currency symbols). Applied identically to documents and gold keywords.
    """
    doc = nlp(text)
    return [
        t.lemma_.lower() for t in doc
        if not t.is_punct and not t.is_space and not t.is_stop
        and not t.like_num and len(t.text) >= 3 and t.lemma_.isalpha()
        and t.lemma_.lower() not in _ENGLISH_STOPWORDS
    ]


def preprocess_all(documents: list[tuple[str, str]]) -> list[list[str]]:
    nlp = _nlp()
    return [preprocess(text, nlp) for _, text in documents]


def build_vocabulary(docs_tokens: list[list[str]]) -> list[str]:
    """Deterministic, sorted vocabulary (union of tokens across documents)."""
    vocab: set[str] = set()
    for tokens in docs_tokens:
        vocab.update(tokens)
    return sorted(vocab)


def count_matrix(docs_tokens: list[list[str]], vocab: list[str]) -> np.ndarray:
    """Raw term-frequency matrix (N_docs x V)."""
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(docs_tokens), len(vocab)), dtype=np.int64)
    for d, tokens in enumerate(docs_tokens):
        for t in tokens:
            X[d, idx[t]] += 1
    return X


def normalize_gold(keyword_phrases: list[str], nlp) -> set[str]:
    """Normalize author keyword phrases into a set of content unigrams."""
    unigrams: set[str] = set()
    for phrase in keyword_phrases:
        for token in preprocess(phrase, nlp):
            if len(token) >= 3:
                unigrams.add(token)
    return unigrams


def build_gold(documents: list[tuple[str, str]], nlp) -> tuple[list[set[str]], list[int]]:
    """Return (gold_sets aligned to ``documents`` order, excluded indices)."""
    from bb_idf.experiment import gold as gold_module

    gb = gold_module.gold_by_file()
    gold_sets: list[set[str]] = []
    excluded: list[int] = []
    for i, (fname, _text) in enumerate(documents):
        entry = gb.get(fname)
        if entry is None or fname in gold_module.EXCLUDED_FILES or not entry["keywords"]:
            gold_sets.append(set())
            excluded.append(i)
        else:
            gold_sets.append(normalize_gold(entry["keywords"], nlp))
    return gold_sets, excluded


@dataclass
class PreprocessedCorpus:
    """Materialized preprocessing: everything downstream needs, no spaCy."""
    documents: list[tuple[str, str]]
    docs_tokens: list[list[str]]
    vocab: list[str]
    X: np.ndarray
    gold_sets: list[set[str]] = field(default_factory=list)
    excluded: list[int] = field(default_factory=list)


def _preprocess_signature() -> str:
    """Fingerprint of the preprocessing + gold logic, used to invalidate the
    on-disk cache when the code (not just the corpus) changes."""
    from bb_idf.experiment import gold as gold_module

    parts = [
        inspect.getsource(preprocess),
        inspect.getsource(normalize_gold),
        inspect.getsource(gold_module),
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def preprocess_corpus(corpus_dir: str | Path,
                      cache_dir: str | Path = "data/processed") -> PreprocessedCorpus:
    """Load (or compute + cache) the preprocessed corpus.

    spaCy runs only when the cache is missing or the corpus changed; otherwise
    the artifact is read from ``cache_dir/preprocessed.pkl``. The cache is
    keyed by both the sorted filenames and a signature of the preprocessing and
    gold code, so editing ``preprocess``, ``normalize_gold`` or ``gold.py``
    forces a re-preprocessing.
    """
    documents = load_documents(corpus_dir)
    filenames = [f for f, _ in documents]
    signature = _preprocess_signature()

    cache_path = Path(cache_dir) / "preprocessed.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                cached = pickle.load(fh)
            if isinstance(cached, tuple) and len(cached) == 2:
                cached_sig, cached_corpus = cached
            else:
                # Legacy cache without signature: recompute once.
                cached_sig, cached_corpus = None, cached
            cached_names = [f for f, _ in cached_corpus.documents]
            if cached_sig == signature and cached_names == filenames:
                return cached_corpus
            print("Cache obsoleta (lógica o corpus cambiado), re-preprocesando...")
        except Exception:
            print("Cache ilegible, re-preprocesando...")

    nlp = _nlp()
    docs_tokens = preprocess_all(documents)
    vocab = build_vocabulary(docs_tokens)
    X = count_matrix(docs_tokens, vocab)
    gold_sets, excluded = build_gold(documents, nlp)
    result = PreprocessedCorpus(
        documents=documents, docs_tokens=docs_tokens, vocab=vocab,
        X=X, gold_sets=gold_sets, excluded=excluded,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump((signature, result), fh)
    return result
