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
            candidate = data["candidates"][0]
            finish_reason = candidate.get("finishReason", "")
            text = candidate["content"]["parts"][0]["text"]
            if finish_reason == "MAX_TOKENS":
                log(f"Gemini response was truncated (finishReason=MAX_TOKENS). "
                    f"Got {len(text)} chars before cutoff. Consider raising max_tokens.", "WARN")
            return text
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
            if provider == "replicate":
                _call_replicate_image(prompt, cfg, out_path)
            elif provider == "pollinations":
                _call_pollinations_image(prompt, cfg, out_path)
            elif provider == "gemini":
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


def _call_replicate_image(prompt: str, cfg: dict, out_path: str):
    """Generate image via Replicate using FLUX or Stable Diffusion.
    Free tier: $25/month credit (more than enough for several videos).
    Get a free API key at https://replicate.com and add it as REPLICATE_API_KEY secret.
    Docs: https://replicate.com/docs/
    """
    import requests
    import time
    
    api_key = get_env("REPLICATE_API_KEY")
    
    # Use FLUX Schnell (fast, free tier, works well for illustration style)
    model_config = cfg["image_generation"].get("replicate", {})
    model_name = model_config.get("model", "black-forest-labs/flux-schnell")
    width, height = cfg["video"]["resolution"]
    
    # Step 1: Get the latest model version ID from Replicate API
    log(f"Fetching latest version for {model_name}...", "INFO")
    version_id = _get_replicate_model_version(model_name, api_key)
    
    # Step 2: Submit the job
    url = "https://api.replicate.com/v1/predictions"
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    
    payload = {
        "version": version_id,
        "input": {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": 4,  # FLUX Schnell uses only 4 steps
            "guidance_scale": 3.5,
        }
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    prediction = resp.json()
    prediction_id = prediction["id"]
    
    log(f"Replicate job submitted: {prediction_id}", "INFO")
    
    # Step 3: Poll for completion
    max_polls = 360  # ~30 min max wait
    poll_interval = 2
    
    for poll_num in range(max_polls):
        time.sleep(poll_interval)
        
        status_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
        resp = requests.get(status_url, headers=headers, timeout=30)
        resp.raise_for_status()
        prediction = resp.json()
        
        status = prediction.get("status")
        if status == "succeeded":
            output = prediction.get("output")
            if output and isinstance(output, list) and len(output) > 0:
                image_url = output[0]
                log(f"Image generation complete, downloading...", "INFO")
                
                # Step 4: Download the image
                img_resp = requests.get(image_url, timeout=30)
                img_resp.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(img_resp.content)
                return
            else:
                raise ValueError(f"Unexpected output format: {output}")
        
        elif status == "failed":
            error = prediction.get("error", "Unknown error")
            raise RuntimeError(f"Replicate prediction failed: {error}")
        
        elif status in ("starting", "processing"):
            elapsed = (poll_num + 1) * poll_interval
            if (poll_num + 1) % 5 == 0:  # Log every 10 seconds
                log(f"Waiting for image generation... ({elapsed}s elapsed)", "INFO")
            continue
        
        else:
            raise RuntimeError(f"Unexpected prediction status: {status}")
    
    raise RuntimeError(f"Image generation timed out after {max_polls * poll_interval}s")


def _get_replicate_model_version(model_name: str, api_key: str) -> str:
    """Fetch the latest version ID for a Replicate model.
    model_name should be in format: owner/model
    Uses the correct endpoint: https://api.replicate.com/v1/{owner}/{model}/versions
    """
    import requests
    
    # Split model_name into owner and model
    if "/" not in model_name:
        raise ValueError(f"Model name must be in format 'owner/model', got: {model_name}")
    
    owner, model = model_name.split("/", 1)
    
    # Correct Replicate API endpoint for model versions
    url = f"https://api.replicate.com/v1/{owner}/{model}/versions"
    headers = {"Authorization": f"Token {api_key}"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        # If model not found, provide helpful error
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Model '{model_name}' not found on Replicate or you don't have access. "
                f"Check https://replicate.com/{owner}/{model} or explore at https://replicate.com/explore. "
                f"Make sure your REPLICATE_API_KEY is correct."
            )
        raise
    
    if isinstance(data, dict) and "results" in data:
        versions = data["results"]
    elif isinstance(data, list):
        versions = data
    else:
        raise ValueError(f"Unexpected response format from Replicate: {type(data)}")
    
    if not versions:
        raise RuntimeError(f"No versions found for model {model_name}")
    
    # Get the first (latest) version
    latest_version = versions[0]
    version_id = latest_version.get("id")
    
    if not version_id:
        raise RuntimeError(f"Could not extract version ID from Replicate response: {latest_version}")
    
    log(f"Using model version: {version_id[:12]}...", "INFO")
    return version_id


def _call_pollinations_image(prompt: str, cfg: dict, out_path: str):
    """Free image generation via Pollinations.ai (Flux model by default).
    Get a free API key (no credit card) at https://enter.pollinations.ai
    and add it as the POLLINATIONS_API_KEY secret. Works without a key too,
    but a key gives more reliable daily quota.
    Docs: https://github.com/pollinations/pollinations/blob/main/APIDOCS.md
    """
    import requests
    import urllib.parse

    model = cfg["image_generation"]["pollinations"].get("model", "flux")
    width, height = cfg["video"]["resolution"]
    api_key = get_env("POLLINATIONS_API_KEY", required=False)

    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://gen.pollinations.ai/image/{encoded_prompt}"
    params = {"width": width, "height": height, "model": model, "nologo": "true"}
    if api_key:
        params["key"] = api_key

    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "image" not in content_type:
        raise ValueError(f"Pollinations did not return an image (content-type: {content_type})")

    with open(out_path, "wb") as f:
        f.write(resp.content)


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
