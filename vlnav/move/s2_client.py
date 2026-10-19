"""Clients for the DualVLN S2 server (TASK_move_subtask.md §6.3).

`HttpS2Client` talks to `scripts/serve_dualvln_s2.py`; `CachedS2Client` stores replies on disk by a
hash of the request (instruction, flags, image bytes) so bag replays are reproducible;
`MockS2Client` replays scripted texts for tests. Calls are synchronous; `MoveCore` runs them in a
worker thread.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from pydantic import BaseModel


@dataclass
class S2Request:
    instruction: str
    image: np.ndarray                                  # HxWx3 RGB, original resolution
    history: list[np.ndarray] = field(default_factory=list)   # RGB, already resized to the model input
    look_down: bool = False
    prev_output: str | None = None
    look_down_image: np.ndarray | None = None          # RGB, original resolution


class S2Response(BaseModel):
    text: str
    input_size: tuple[int, int]                        # (w, h) of the image the pixel refers to
    processed_size: tuple[int, int]                    # (w, h) after the Qwen processor's smart resize
    n_history: int
    latency_ms: float
    cached: bool = False


class S2Client(Protocol):
    def query(self, req: S2Request) -> S2Response: ...


def _png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def _files(req: S2Request) -> dict[str, bytes]:
    files = {"image": _png(req.image)}
    for i, h in enumerate(req.history):
        files[f"history_{i}"] = _png(h)
    if req.look_down:
        files["look_down_image"] = _png(req.look_down_image if req.look_down_image is not None else req.image)
    return files


def _meta(req: S2Request) -> dict:
    return {"instruction": req.instruction, "look_down": req.look_down, "prev_output": req.prev_output}


def request_key(req: S2Request) -> str:
    h = hashlib.sha256(json.dumps(_meta(req), sort_keys=True).encode())
    for name, data in sorted(_files(req).items()):
        h.update(name.encode())
        h.update(data)
    return h.hexdigest()


class HttpS2Client:
    def __init__(self, base_url: str, timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict:
        import requests
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def query(self, req: S2Request) -> S2Response:
        import requests
        files = {k: (f"{k}.png", v, "image/png") for k, v in _files(req).items()}
        r = requests.post(f"{self.base_url}/s2", files=files, data={"json": json.dumps(_meta(req))},
                          timeout=self.timeout_s)
        r.raise_for_status()
        return S2Response.model_validate(r.json())


class CachedS2Client:
    def __init__(self, inner: S2Client, cache_dir: str | Path):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def query(self, req: S2Request) -> S2Response:
        path = self.dir / f"{request_key(req)}.json"
        if path.exists():
            return S2Response.model_validate({**json.loads(path.read_text()), "cached": True})
        res = self.inner.query(req)
        path.write_text(res.model_dump_json())
        return res


class MockS2Client:
    """Replies from a list of texts (in order, last one repeated) or a function of the request."""

    def __init__(self, script: Sequence[str] | Callable[[S2Request], str],
                 input_size: tuple[int, int] | None = None, processed_size: tuple[int, int] | None = None):
        self.script = script
        self.input_size = input_size
        self.processed_size = processed_size
        self.requests: list[S2Request] = []

    def query(self, req: S2Request) -> S2Response:
        self.requests.append(req)
        if callable(self.script):
            text = self.script(req)
        else:
            text = self.script[min(len(self.requests), len(self.script)) - 1]
        img = req.look_down_image if req.look_down and req.look_down_image is not None else req.image
        size = self.input_size or (img.shape[1], img.shape[0])
        return S2Response(text=text, input_size=size, processed_size=self.processed_size or size,
                          n_history=len(req.history), latency_ms=0.0)
