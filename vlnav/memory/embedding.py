"""Cached label similarity on top of the shared `TextEmbedder`."""
from __future__ import annotations

import threading

import numpy as np

from vlnav.acquisition.label_matcher import TextEmbedder


class LabelSimilarity:
    """Cosine similarity between short texts, clipped to [0, 1]; embeddings cached per text."""

    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            missing = sorted({t for t in texts if t not in self._cache})
            if missing:
                for t, e in zip(missing, np.asarray(self.embedder.encode(missing), dtype=float)):
                    self._cache[t] = e
            return np.stack([self._cache[t] for t in texts]) if texts else np.empty((0, 0))

    def sims(self, text: str, labels: list[str]) -> np.ndarray:
        if not labels:
            return np.empty(0)
        e = self.embed([text, *labels])
        return np.clip(e[1:] @ e[0], 0.0, 1.0)

    def __call__(self, a: str, b: str) -> float:
        return float(self.sims(a, [b])[0])
