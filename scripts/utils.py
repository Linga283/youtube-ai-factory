"""
Shared utilities for the YouTube AI Factory pipeline.
Handles: config loading, prompt templating, AI provider calls (Gemini/OpenAI),
JSON-safe parsing, history tracking, and simple logging.
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

import yaml

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def load_config() -> dict:
    with open(ROOT / "config" / "channel.yml", "r") as f:
        return yaml.safe_load(f)


def load_prompts() -> dict:
    with open(ROOT / "config" / "prompts.yml", "r") as f:
        return yaml.safe_load(f)


def get_env(name: str, required: bool = True) -> str:
    val = os.environ.get(name, "")
    if required and not val:
        log(f"Missing required environment variable: {name}", "ERROR")
        sys.exit(1)
    return val


# ------------------------------------------------------------------
# JSON extraction (models sometimes wrap JSON in prose or ```json fences)
# ------------------------------------------------------------------
def extract_json(text: str):
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first { ... } or [ ... ] block via brace matching
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not extract valid JSON from model response:\n{text[:500]}")


# ------------------------------------------------------------------
# AI provider abstraction
# ------------------------------------------------------------------
def call_ai_text(prompt: str, provider: str, model: str, max_tokens: int = 4096) -> str:
    """Call a text-generation model (Gemini or OpenAI) and return raw text.
    Retries with exponential backoff on transient errors (503, 429, timeouts)."""
    import time as _time
    import requests as _requests

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            if provider == "gemini":
                return _call_gemini_text(prompt, model, max_tokens)
            elif provider == "openai":
                return _call_openai_text(prompt, model, max_tokens)
            else:
                raise ValueError(f"Unknown text provider: {provider}")
        except _requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt < max_attempts:
                wait = min(60, 2 ** attempt)
                log(f"Transient error ({status}) from {provider}, retrying in {wait}s "
                    f"(attempt {attempt}/{max_attempts})...", "WARN")
                _time.sleep(wait)
                continue
            raise
        except (_requests.exceptions.ConnectionError, _requests.exceptions.Timeout) as e:
            if attempt < max_attempts:
                wait = min(60, 2 ** attempt)
                log(f"Network error ({e}), retrying in {wait}s "
                    f"(attempt {attempt}/{max_attempts})...", "WARN")
                _time.sleep(wait)
                continue
            raise


def _call_gemini_text(prompt: str, model: str, max_tokens: int) -> str:
    import requests
    import time
    api_key = get_env("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.9},
    }

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = min(60, 2 ** attempt)
            log(f"Gemini returned {resp.status_code}, retrying in {wait}s "
                f"(attempt {attempt}/{max_retries})...", "WARN")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise ValueError(f"Unexpected Gemini response shape: {json.dumps(data)[:500]}")

    raise RuntimeError(f"Gemini API still unavailable after {max_retries} retries "
                        f"(last status: {resp.status_code}). This is usually a temporary "
                        f"issue on Google's side - try again in a few minutes.")


def _call_openai_text(prompt: str, model: str, max_tokens: int) -> str:
    import requests
    api_key = get_env("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.9,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_ai_image(prompt: str, provider: str, cfg: dict, out_path: str):
    """Generate a single image and save to out_path.
    Retries with exponential backoff on transient errors (503, 429, timeouts)."""
    import time as _time
    import requests as _requests

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            if provider == "gemini":
                _call_gemini_image(prompt, cfg, out_path)
            elif provider == "openai":
                _call_openai_image(prompt, cfg, out_path)
            else:
                raise ValueError(f"Unknown image provider: {provider}")
            return
        except _requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt < max_attempts:
                wait = min(60, 2 ** attempt)
                log(f"Transient error ({status}) from {provider} image call, retrying in {wait}s "
                    f"(attempt {attempt}/{max_attempts})...", "WARN")
                _time.sleep(wait)
                continue
            raise
        except (_requests.exceptions.ConnectionError, _requests.exceptions.Timeout) as e:
            if attempt < max_attempts:
                wait = min(60, 2 ** attempt)
                log(f"Network error ({e}), retrying in {wait}s "
                    f"(attempt {attempt}/{max_attempts})...", "WARN")
                _time.sleep(wait)
                continue
            raise


def _call_gemini_image(prompt: str, cfg: dict, out_path: str):
    import requests
    import base64
    import time
    api_key = get_env("GEMINI_API_KEY")
    model = cfg["image_generation"]["gemini"]["model"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        resp = requests.post(url, json=payload, timeout=180)
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = min(60, 2 ** attempt)
            log(f"Gemini image API returned {resp.status_code}, retrying in {wait}s "
                f"(attempt {attempt}/{max_retries})...", "WARN")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        for part in parts:
            if "inlineData" in part:
                img_bytes = base64.b64decode(part["inlineData"]["data"])
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                return
        raise ValueError("No image data returned by Gemini")

    raise RuntimeError(f"Gemini image API still unavailable after {max_retries} retries "
                        f"(last status: {resp.status_code}).")


def _call_openai_image(prompt: str, cfg: dict, out_path: str):
    import requests
    import base64
    api_key = get_env("OPENAI_API_KEY")
    model = cfg["image_generation"]["openai"]["model"]
    size = cfg["image_generation"]["openai"]["size"]
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "size": size, "n": 1}
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    b64 = data["data"][0]["b64_json"]
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))


# ------------------------------------------------------------------
# History tracking (avoid repeating topics)
# ------------------------------------------------------------------
def load_history(cfg: dict) -> list:
    path = ROOT / cfg["history"]["db_file"]
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_history_entry(cfg: dict, entry: dict):
    path = ROOT / cfg["history"]["db_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    history = load_history(cfg)
    history.append(entry)
    max_items = cfg["history"]["max_history_items"]
    history = history[-max_items:]
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def data_dir() -> Path:
    d = ROOT / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d
