"""VLM clients (acquisition spec §8.2). One client is shared by acquisition and the planner.

Messages use the OpenAI chat format; images are `{"type": "image_url", "image_url": {"url":
"data:image/jpeg;base64,..."}}` parts (see `image_part`).

- `OpenAIVLMClient`: any OpenAI-compatible endpoint (vLLM, or a cloud API with `api_key_env`).
  Structured output via `response_format` (OpenAI json_schema) or `guided_json` (older vLLM) — AUDIT 23.
  The key is read from the environment variable named by `api_key_env`, never from a file.
- `LocalQwenVLMClient`: Qwen3-VL loaded in-process with transformers on one or several GPUs. No
  constrained decoding: the schema is appended to the prompt and the reply is parsed robustly.
- `MockVLMClient`: scripted replies for tests (timeouts and malformed JSON included).
Every reply is cached by prompt hash when `cache_dir` is set (AUDIT 27).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from vlnav.common.config import CONFIG_DIR, REPO_ROOT, deep_merge, load_yaml


class VLMClient(Protocol):
    def chat(self, messages: list[dict], json_schema: dict | None, timeout_s: float) -> str: ...


class VLMError(RuntimeError):
    """Transport / server failure (distinct from a bad reply)."""


# ---------------------------------------------------------------------------------- config
class LocalVLMCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    devices: list[int]
    max_memory_gb: float
    dtype: str
    attn_implementation: str


class VLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    backend: Literal["local", "openai", "mock"]
    model: str
    base_url: str
    api_key_env: str | None           # backend openai: env variable holding the key; None -> local server
    structured_output: Literal["response_format", "guided_json", "none"]
    temperature: float
    max_tokens: int
    cache_dir: str | None
    local: LocalVLMCfg


def load_vlm_config(path: str | Path = CONFIG_DIR / "vlm.yaml", overrides: dict | None = None) -> VLMConfig:
    return VLMConfig.model_validate(deep_merge(load_yaml(path), overrides))


# ------------------------------------------------------------------------------- utilities
def encode_image(rgb: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def image_part(rgb: np.ndarray) -> dict:
    return {"type": "image_url", "image_url": {"url": encode_image(rgb)}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def strip_images(messages: list[dict]) -> list[dict]:
    """Copy of `messages` with image data replaced by a short hash (logs, cache keys)."""
    out = copy.deepcopy(messages)
    for m in out:
        if isinstance(m.get("content"), list):
            for p in m["content"]:
                if p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    p["image_url"] = {"url": f"<image sha1={hashlib.sha1(url.encode()).hexdigest()[:12]}>"}
    return out


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model reply: code fences, leading prose and reasoning blocks are
    tolerated. Raises ValueError if no object can be parsed."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object in reply: {text[:200]!r}")


class ResponseCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(model: str, messages: list[dict], schema: dict | None, params: dict) -> str:
        blob = json.dumps([model, strip_images(messages), schema, params], sort_keys=True)
        # image hashes in strip_images keep the key image-sensitive
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        p = self.root / f"{key}.json"
        return json.loads(p.read_text())["reply"] if p.exists() else None

    def put(self, key: str, reply: str) -> None:
        (self.root / f"{key}.json").write_text(json.dumps({"reply": reply}))


def _schema_hint(schema: dict) -> str:
    return ("\n\nAnswer with a single JSON object that matches this JSON schema, no other text:\n"
            + json.dumps(schema))


# ---------------------------------------------------------------------------------- clients
class _CachedClient:
    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self.cache = ResponseCache(_resolve(cfg.cache_dir)) if cfg.cache_dir else None
        self.n_calls = 0
        self.n_cache_hits = 0

    def chat(self, messages: list[dict], json_schema: dict | None, timeout_s: float) -> str:
        params = {"temperature": self.cfg.temperature, "max_tokens": self.cfg.max_tokens,
                  "structured_output": self.cfg.structured_output}
        key = ResponseCache.key(self.cfg.model, messages, json_schema, params) if self.cache else None
        if key is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.n_cache_hits += 1
                return hit
        self.n_calls += 1
        reply = self._chat(messages, json_schema, timeout_s)
        if key is not None:
            self.cache.put(key, reply)
        return reply

    def _chat(self, messages, json_schema, timeout_s) -> str:
        raise NotImplementedError


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


class OpenAIVLMClient(_CachedClient):
    def __init__(self, cfg: VLMConfig):
        super().__init__(cfg)
        from openai import OpenAI
        api_key = "EMPTY"
        if cfg.api_key_env:
            api_key = os.environ.get(cfg.api_key_env, "")
            if not api_key:
                raise VLMError(f"backend openai: environment variable {cfg.api_key_env} is not set")
        self.client = OpenAI(base_url=cfg.base_url, api_key=api_key)

    def _chat(self, messages, json_schema, timeout_s) -> str:
        kw: dict[str, Any] = {}
        if json_schema is not None and self.cfg.structured_output == "response_format":
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "reply", "schema": json_schema}}
        elif json_schema is not None and self.cfg.structured_output == "guided_json":
            kw["extra_body"] = {"guided_json": json_schema}
        try:
            r = self.client.chat.completions.create(
                model=self.cfg.model, messages=messages, temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens, timeout=timeout_s, **kw)
        except Exception as e:                       # openai.APITimeoutError, connection errors
            if "timeout" in type(e).__name__.lower():
                raise TimeoutError(str(e)) from e
            raise VLMError(str(e)) from e
        return r.choices[0].message.content or ""


class LocalQwenVLMClient(_CachedClient):
    """Qwen3-VL in-process. `local.devices: [1]` -> one GPU; `[0, 1]` -> split by accelerate."""

    def __init__(self, cfg: VLMConfig):
        super().__init__(cfg)
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        path = _resolve(cfg.model)
        if not path.exists():
            raise FileNotFoundError(f"VLM weights not found: {path}")
        lc = cfg.local
        dtype = getattr(torch, lc.dtype)
        if len(lc.devices) == 1:
            device_map: Any = {"": f"cuda:{lc.devices[0]}"}
            max_memory = None
        else:
            device_map = "auto"
            max_memory = {d: f"{lc.max_memory_gb}GiB" for d in lc.devices}
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=dtype, device_map=device_map, max_memory=max_memory,
            attn_implementation=lc.attn_implementation).eval()
        self._lock = threading.Lock()
        self._torch = torch

    @staticmethod
    def _to_hf(messages: list[dict]) -> list[dict]:
        from PIL import Image
        import io
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
                continue
            parts = []
            for p in content:
                if p["type"] == "text":
                    parts.append({"type": "text", "text": p["text"]})
                else:
                    b64 = p["image_url"]["url"].split(",", 1)[1]
                    parts.append({"type": "image",
                                  "image": Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")})
            out.append({"role": m["role"], "content": parts})
        return out

    def _chat(self, messages, json_schema, timeout_s) -> str:
        msgs = copy.deepcopy(messages)
        if json_schema is not None and self.cfg.structured_output != "none":
            last = msgs[-1]
            if isinstance(last["content"], str):
                last["content"] += _schema_hint(json_schema)
            else:
                last["content"].append(text_part(_schema_hint(json_schema)))
        inputs = self.processor.apply_chat_template(
            self._to_hf(msgs), tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt").to(self.model.device)
        gen = {"max_new_tokens": self.cfg.max_tokens}
        if self.cfg.temperature > 0:
            gen.update(do_sample=True, temperature=self.cfg.temperature)
        else:
            gen.update(do_sample=False)
        with self._lock, self._torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0]


class MockVLMClient:
    """Scripted replies: a string, a list (consumed in order) or a function of the messages.
    A reply that is an Exception instance is raised (e.g. TimeoutError())."""

    def __init__(self, reply: str | list | Callable[[list[dict]], str | Exception]):
        self._reply = reply
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], json_schema: dict | None, timeout_s: float) -> str:
        self.calls.append({"messages": messages, "schema": json_schema})
        if callable(self._reply):
            r = self._reply(messages)
        elif isinstance(self._reply, list):
            r = self._reply[len(self.calls) - 1]
        else:
            r = self._reply
        if isinstance(r, Exception):
            raise r
        return r


def make_vlm_client(cfg: VLMConfig) -> VLMClient:
    if cfg.backend == "openai":
        return OpenAIVLMClient(cfg)
    if cfg.backend == "local":
        return LocalQwenVLMClient(cfg)
    raise ValueError("backend 'mock' is created directly with MockVLMClient(...)")
