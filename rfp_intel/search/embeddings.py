"""Embedding backends.

* ``sentence-transformers`` (default, local, BAAI/bge-small-en-v1.5): strong
  retrieval quality at 384 dims and CPU speed.
* ``openai``: text-embedding-3-small via API.
* ``spacy``: averaged static word vectors (en_core_web_md), offline semantic baseline.
* ``hash``: dependency-free, deterministic fallback (hashed word + char n-grams
  projected to 384 dims). Used when model weights cannot be downloaded (offline
  CI). It is lexical in nature; the README reports results for both.
"""
from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

from ..config import Settings, get_settings

log = logging.getLogger("rfp_intel.search")


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class Embedder:
    name = "base"
    dim = 384

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(Embedder):
    name = "hash"

    def __init__(self, dim: int = 384, n_features: int = 2 ** 18, seed: int = 13) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.word = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), alternate_sign=False,
                                      norm="l2", token_pattern=r"(?u)\b[\w][\w\-./#]*\b")
        self.char = HashingVectorizer(n_features=n_features, analyzer="char_wb", ngram_range=(3, 5),
                                      alternate_sign=False, norm="l2")
        from scipy import sparse

        rng = np.random.default_rng(seed)
        # sparse random projection: each hashed feature maps to 4 dims with random signs.
        # Fixed seed => stable vector space => incremental indexing stays valid.
        k = 4
        rows = np.repeat(np.arange(n_features), k)
        cols = rng.integers(0, dim, size=n_features * k)
        vals = rng.choice([-1.0, 1.0], size=n_features * k).astype(np.float32)
        self.proj = sparse.csr_matrix((vals, (rows, cols)), shape=(n_features, dim))

    def encode(self, texts, is_query=False):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        x = self.word.transform(texts) + 0.5 * self.char.transform(texts)
        return _normalize(np.asarray((x @ self.proj).todense()))


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = f"st:{model}"

    def encode(self, texts, is_query=False):
        if is_query and "bge" in self.model_name.lower():
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        return _normalize(self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True))


class SpacyVectorsEmbedder(Embedder):
    """Mean of static word vectors (spaCy ``en_core_web_md``), stopwords/punctuation removed.

    A light, fully offline *semantic* baseline (knows "deadline" ~ "due date")
    for environments that cannot download transformer weights.
    """

    def __init__(self, model: str = "en_core_web_md") -> None:
        import spacy

        self.nlp = spacy.load(model, exclude=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler", "senter"])
        self.dim = self.nlp.vocab.vectors_length
        self.name = f"spacy:{model}"

    def encode(self, texts, is_query=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=64)):
            vecs = [t.vector for t in doc if t.has_vector and not (t.is_stop or t.is_punct or t.is_space)]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return _normalize(out)


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model if model.startswith("text-embedding") else "text-embedding-3-small"
        self.dim = 1536
        self.name = f"openai:{self.model_name}"

    def encode(self, texts, is_query=False):
        out = []
        for i in range(0, len(texts), 256):
            resp = self.client.embeddings.create(model=self.model_name, input=texts[i:i + 256])
            out.extend(d.embedding for d in resp.data)
        return _normalize(np.array(out, dtype=np.float32))


_CACHE: dict[str, Embedder] = {}


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    key = f"{settings.embedding_backend}:{settings.embedding_model}"
    if key in _CACHE:
        return _CACHE[key]
    backend = settings.embedding_backend.lower()
    emb: Embedder
    try:
        if backend in ("sentence-transformers", "st"):
            emb = SentenceTransformerEmbedder(settings.embedding_model)
        elif backend == "spacy":
            emb = SpacyVectorsEmbedder(settings.embedding_model if settings.embedding_model.startswith("en_") else "en_core_web_md")
        elif backend == "openai":
            emb = OpenAIEmbedder(settings.embedding_model)
        else:
            emb = HashEmbedder()
    except Exception as exc:
        # offline fallback chain: static word vectors if installed, else hashed n-grams
        try:
            emb = SpacyVectorsEmbedder()
        except Exception:
            emb = HashEmbedder()
        log.warning("embedding backend %s unavailable (%s); falling back to %s", backend, exc, emb.name)
    _CACHE[key] = emb
    return emb


_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-./#]*[a-z0-9]|[a-z0-9]")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who whom when where how does do did shall should may must any all can".split())


def tokenize(text: str) -> list[str]:
    """BM25 tokenizer that keeps identifiers (JA-207652, 210-BLYZ, WD22TB4) intact
    and also indexes their parts so partial matches still score."""
    toks: list[str] = []
    for t in _TOKEN.findall(text.lower()):
        if t in STOPWORDS:
            continue
        toks.append(t)
        if any(c in t for c in "-./#"):
            toks.extend(p for p in re.split(r"[-./#]", t) if p and p not in STOPWORDS)
    return toks


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
