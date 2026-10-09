"""Target text -> matching node labels (acquisition spec §7.1), and the shared `TextEmbedder`.

Instantiate one embedder in `runtime/` and inject it into memory (association, find_by_text),
acquisition (`LabelMatcher`) and the planner serializer.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from vlnav.common.config import REPO_ROOT


@runtime_checkable
class TextEmbedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float array of L2-normalized embeddings, one row per text."""
        ...


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class OpenAIClipTextEmbedder:
    """CLIP text tower from the openai `clip` package (e.g. ViT-B-32.pt, also YOLO-World's)."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        import clip
        import torch
        path = Path(checkpoint)
        path = path if path.is_absolute() else REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"CLIP checkpoint missing: {path}")
        self.model, _ = clip.load(str(path), device=device)
        self.model.eval()
        self._clip, self._torch, self.device = clip, torch, device
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock, self._torch.no_grad():
            tok = self._clip.tokenize(texts, truncate=True).to(self.device)
            return _normalize(self.model.encode_text(tok).float().cpu().numpy())


class HFTextEmbedder:
    """Any transformers model with `get_text_features` (CLIP, SigLIP)."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self._torch, self.device = torch, device
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock, self._torch.no_grad():
            batch = self.tok(texts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            out = self.model.get_text_features(**batch)
            out = out if isinstance(out, self._torch.Tensor) else out.pooler_output
            return _normalize(out.float().cpu().numpy())


def make_text_embedder(kind: str, path: str, device: str) -> TextEmbedder:
    if kind == "clip_vitb32":
        return OpenAIClipTextEmbedder(path, device)
    if kind == "hf":
        return HFTextEmbedder(path, device)
    raise ValueError(f"unknown text encoder {kind!r}")


def norm_label(text: str) -> str:
    return " ".join(text.lower().strip().split())


class LabelMatcher:
    """Cosine similarity in [0, 1] between a target and detector labels (prompt template on both
    sides, cached embeddings); a synonym-table hit scores 1.0."""

    def __init__(self, embedder: TextEmbedder | None, template: str, threshold: float,
                 synonyms: dict[str, list[str]] | None = None, rel_margin: float | None = None):
        self.embedder = embedder
        self.template = template
        self.threshold = threshold
        self.rel_margin = rel_margin       # also require sim >= best - rel_margin (CLIP band is narrow)
        self.synonyms = {norm_label(k): {norm_label(v) for v in vs} for k, vs in (synonyms or {}).items()}
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            missing = sorted({t for t in texts if t not in self._cache})
            if missing:
                vecs = self.embedder.encode([self.template.format(t) for t in missing])
                self._cache.update(zip(missing, np.asarray(vecs, dtype=float)))
            return np.stack([self._cache[t] for t in texts])

    def score(self, target: str, labels: list[str]) -> dict[str, float]:
        t = norm_label(target)
        uniq = sorted({norm_label(lbl) for lbl in labels})
        if not uniq:
            return {}
        if self.embedder is not None:
            e = self._embed([t, *uniq])
            sims = dict(zip(uniq, np.clip(e[1:] @ e[0], 0.0, 1.0)))
        else:
            sims = {lbl: float(lbl == t) for lbl in uniq}
        syn = self.synonyms.get(t, set())
        out = {}
        for lbl in labels:
            n = norm_label(lbl)
            out[lbl] = 1.0 if (n == t or n in syn) else float(sims[n])
        return out

    def matches(self, target: str, labels: list[str]) -> dict[str, float]:
        scores = self.score(target, labels)
        floor = self.threshold
        if self.rel_margin is not None and scores:
            floor = max(floor, max(scores.values()) - self.rel_margin)
        return {lbl: s for lbl, s in scores.items() if s >= floor}

    def sim(self, a: str, b: str) -> float:
        """Symmetric pairwise similarity (memory association / serializer relevance)."""
        return self.score(a, [b])[b]
