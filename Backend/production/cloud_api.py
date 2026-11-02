"""Lightweight Gemini + Ollama API used by both Render and local installs.

The hosted Render instance uses Gemini. A local install can also connect to the
user's Ollama server, keep chat context, and expose the same response contract
to the React frontend. Heavy research/training dependencies are deliberately
not loaded by this serving entry point.
"""

from collections import OrderedDict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
import queue
import re
import threading
import time

import requests
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
app.logger.setLevel(logging.INFO)

GEMINI_MODELS = {
    "gemini-pro": "gemini-2.5-pro",
    "gemini-flash": "gemini-2.5-flash",
    "gemini": "gemini-2.5-flash",
}
OLLAMA_MODELS = {
    "tinyllama": ("tinyllama:latest", "tinyllama"),
    "phi-3": ("phi3:latest", "phi3", "phi:latest"),
    "codellama": ("codellama:7b", "codellama:latest"),
    "qwen-coder": ("qwen2.5-coder:3b", "qwen2.5-coder:latest"),
    "qwen-math": ("qwen2-math:latest", "qwen2-math:7b"),
    "deepseek-math": ("t1c/deepseek-math-7b-rl:latest", "deepseek-math:latest"),
}
STRATEGY_TARGETS = {
    "conservative": {"target_reduction": 15, "min_similarity": 0.90},
    "balanced": {"target_reduction": 30, "min_similarity": 0.90},
    "aggressive": {"target_reduction": 35, "min_similarity": 0.90},
}
REPORT_BENCHMARKS = {
    "token_reduction_percent": 30,
    "cost_savings_percent_min": 25,
    "cost_savings_percent_max": 35,
    "semantic_similarity_minimum": 0.90,
    "semantic_similarity_reported": 0.92,
    "multi_turn_coherence_improvement_percent": 22,
    "single_turn_prompts": 300,
    "multi_turn_conversations": 80,
}
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_EMBEDDING_USD_PER_MILLION_TOKENS = 0.15
MAX_SEMANTIC_SCORE_CHARS = 6000
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").lower() in ("1", "true", "yes", "on")
CACHE_LIMIT = 128
CONVERSATION_TURNS = 12
_CACHE = OrderedDict()
_CACHE_LOCK = threading.RLock()
_POLICY_LOCK = threading.RLock()
# Runtime feedback is intentionally process-local on the free hosted service.
# Local users can run multiple turns and have them update this contextual policy.
_POLICY_FEEDBACK = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_history(history):
    """Accept only recent user/model text and cap retained context size."""
    if not isinstance(history, list):
        return []
    normalized = []
    for item in history[-CONVERSATION_TURNS * 2:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in ("user", "assistant", "model"):
            continue
        content = str(item.get("content", item.get("text", ""))).strip()
        if content:
            normalized.append({"role": "model" if role in ("assistant", "model") else "user", "content": content[:8000]})
    # Keep the newest context when a conversation has very long turns.
    total = 0
    bounded = []
    for item in reversed(normalized):
        if total >= 24000:
            break
        remaining = 24000 - total
        content = item["content"][-remaining:]
        bounded.append({"role": item["role"], "content": content})
        total += len(content)
    return list(reversed(bounded))


def classify(prompt):
    """Classify explicit tasks and code, not ordinary mentions of a function."""
    text = prompt.strip().lower()
    if "```" in text or re.search(r"(?:^|\n)\s*(?:def\s+\w+\s*\(|class\s+\w+\s*[:({]|(?:const|let|var)\s+\w+\s*=|import\s+\w+|#include\s*<)", text):
        return "coding"

    asks_for_code = re.search(r"\b(?:write|create|implement|build|debug|fix|refactor|review|generate|complete|optimize)\b", text)
    names_code = re.search(r"\b(?:code|coding|programming|script|program|api endpoint|sql query|software)\b", text)
    asks_for_callable = asks_for_code and re.search(r"\b(?:function|method|class)\b", text)
    if (asks_for_code and names_code) or asks_for_callable:
        return "coding"

    if re.search(r"\b(?:calculate|solve|compute|derive|prove|integrate|differentiate|equation|math|mathematics|integral|derivative|theorem)\b", text):
        return "math"
    if re.search(r"\b\d+(?:\.\d+)?\s*[+*/^=<>-]\s*\d+(?:\.\d+)?\b", text):
        return "math"
    return "generic"


def optimize_prompt(prompt, strategy, category):
    """Apply safe cleanup; preserve code and avoid meaning-changing summaries."""
    if category == "coding":
        return prompt.strip(), "code-preserved"

    text = prompt.strip()
    if strategy == "conservative":
        # Keep line and paragraph boundaries; normalize only excess horizontal space.
        cleaned = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
        return cleaned, "conservative-cleanup"

    replacements = (
        (r"\bI would like you to\s+", ""),
        (r"\bI want you to\s+", ""),
        (r"\bCould you please\s+", ""),
        (r"\bCan you please\s+", ""),
        (r"\bplease\b[,:]?\s*", ""),
        (r"\bkindly\b\s*", ""),
    )
    optimized = text
    for pattern, replacement in replacements:
        optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)

    # Compact recurring verbose instructions without deleting the requested tasks.
    compact_phrases = (
        (r"\bthe most important\b", "key"),
        (r"\b(?:please\s+)?be exhaustive, verbose, detailed, explicit, comprehensive, and educational\b", "Be comprehensive, explicit, and educational"),
        (r"\bWhen something can reasonably be explained further, provide the additional explanation rather than assuming it is obvious\b", "Explain further when useful instead of assuming it is obvious"),
        (r"\bAt the same time, maintain logical organization so that the response remains understandable despite its length\b", "Organize the response clearly, even when lengthy"),
        (r"\bTen questions that someone should ask themselves after reading the explanation\b", "Ten reflection questions for after the explanation"),
        (r"\bTen additional topics that would be useful to study next\b", "Ten useful follow-up topics to study"),
        (r"\bA beginner-friendly action plan\b", "An action plan for beginners"),
        (r"\bAn intermediate-level action plan\b", "An intermediate action plan"),
        (r"\bAn advanced-level action plan\b", "An advanced action plan"),
    )
    for pattern, replacement in compact_phrases:
        optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)

    if strategy == "aggressive":
        for pattern, replacement in (
            (r"\bin order to\b", "to"),
            (r"\bdue to the fact that\b", "because"),
            (r"\bat this point in time\b", "now"),
            (r"\bfor the purpose of\b", "for"),
        ):
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
    optimized = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in optimized.splitlines())

    # Remove only repeated, identical sentences. Never paraphrase requirements.
    if strategy in ("balanced", "aggressive"):
        seen = set()
        sentences = re.split(r"(?<=[.!?])\s+", optimized)
        kept = []
        for sentence in sentences:
            key = re.sub(r"\W+", " ", sentence.lower()).strip()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            kept.append(sentence)
        optimized = " ".join(kept)

    optimized = re.sub(r"\s+([,.;:!?])", r"\1", optimized).strip()
    return optimized or text, "redundancy-pruning" if optimized != text else "no-safe-reduction"


def estimate_tokens(text):
    """Rough tokenizer-independent estimate, marked as an estimate in the API."""
    return max(0, math.ceil(len(text) / 4))


def count_prompt_tokens(prompt, history, provider, model_alias):
    """Count the full prompt context with the provider tokenizer when possible."""
    if provider == "gemini":
        try:
            client = gemini_client()
            contents = [{"role": item["role"], "parts": [{"text": item["content"]}]} for item in history]
            contents.append({"role": "user", "parts": [{"text": prompt}]})
            response = client.models.count_tokens(model=GEMINI_MODELS[model_alias], contents=contents)
            total = getattr(response, "total_tokens", None)
            if total is not None:
                return int(total), "Gemini model tokenizer"
        except Exception as exc:
            app.logger.info("Gemini token counting unavailable; using a documented estimate (%s)", type(exc).__name__)

    total_chars = len(prompt) + sum(len(item["content"]) for item in history)
    return math.ceil(total_chars / 4), "estimated at approximately 4 characters per token"


def cosine_similarity(first, second):
    if not first or not second or len(first) != len(second):
        return None
    dot = sum(left * right for left, right in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return None
    return round(max(0.0, min(1.0, dot / (first_norm * second_norm))), 4)


def local_embedding_model():
    configured = os.getenv("OLLAMA_EMBEDDING_MODEL", "").strip()
    tags = ollama_tags()
    if configured:
        return configured
    preferred = ("nomic-embed-text", "mxbai-embed-large", "all-minilm")
    return next((tag for tag in tags if any(name in tag.lower() for name in preferred)), None)


def semantic_similarity(original, optimized, provider):
    """Compare the original and optimized prompts using embedding cosine similarity."""
    if original == optimized:
        return 1.0, "exact text match"
    if max(len(original), len(optimized)) > MAX_SEMANTIC_SCORE_CHARS:
        return None, "not measured: prompt exceeds the live embedding limit"

    try:
        if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
            from google.genai import types
            response = gemini_client().models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=[original, optimized],
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=768,
                ),
            )
            embeddings = getattr(response, "embeddings", None) or []
            values = [getattr(item, "values", None) for item in embeddings]
            if len(values) == 2 and values[0] and values[1]:
                return cosine_similarity(values[0], values[1]), "Gemini text embedding cosine similarity"
        elif provider == "ollama":
            model = local_embedding_model()
            if model:
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": model, "input": [original, optimized]},
                    timeout=(4, 30),
                )
                response.raise_for_status()
                vectors = response.json().get("embeddings", [])
                if len(vectors) == 2:
                    return cosine_similarity(vectors[0], vectors[1]), f"Ollama embedding cosine similarity ({model})"
    except Exception as exc:
        app.logger.info("Semantic embedding unavailable; the request will continue without this score (%s)", type(exc).__name__)

    if provider == "ollama":
        return None, "not measured: install an Ollama embedding model (for example nomic-embed-text)"
    return None, "not measured: semantic embedding service unavailable"


def ollama_tags(force=False):
    if not OLLAMA_ENABLED:
        return []
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5)
        if response.ok:
            return [str(item.get("name", "")) for item in response.json().get("models", []) if item.get("name")]
    except requests.RequestException:
        pass
    except (ValueError, TypeError):
        pass
    return []


def ollama_aliases(tags=None):
    tags = tags if tags is not None else ollama_tags()
    available = {}
    lowered = {tag.lower(): tag for tag in tags}
    for alias, candidates in OLLAMA_MODELS.items():
        for candidate in candidates:
            if candidate.lower() in lowered:
                available[alias] = lowered[candidate.lower()]
                break
        if alias not in available:
            for tag in tags:
                if any(candidate.split(":")[0].lower() in tag.lower() for candidate in candidates):
                    available[alias] = tag
                    break
    return available


def model_catalog():
    tags = ollama_tags()
    available_aliases = ollama_aliases(tags)
    models = []
    if os.getenv("GEMINI_API_KEY"):
        models.extend([
            {"id": "gemini-flash", "name": "Gemini 2.5 Flash", "provider": "gemini", "available": True},
            {"id": "gemini-pro", "name": "Gemini 2.5 Pro", "provider": "gemini", "available": True},
        ])
    for alias, model_name in available_aliases.items():
        models.append({"id": alias, "name": model_name, "provider": "ollama", "available": True})
    return models, tags


@lru_cache(maxsize=1)
def gemini_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Gemini is not configured. Set GEMINI_API_KEY in the backend environment.")
    from google import genai
    return genai.Client(api_key=key)


def conversation_digest(history):
    serialized = "\n".join(f"{item['role']}:{item['content']}" for item in history)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def cache_lookup(prompt, category, model, history):
    # Only cache standalone questions. Reusing an answer with conversational
    # context or a time-sensitive request can change its meaning.
    if history or re.search(r"\b(today|current|latest|now|weather|news|price|stock|score)\b", prompt.lower()):
        return None
    key = (category, model)
    normalized = re.sub(r"\W+", " ", prompt.lower()).strip()
    with _CACHE_LOCK:
        for cache_key, entry in reversed(_CACHE.items()):
            if cache_key[:2] != key:
                continue
            ratio = SequenceMatcher(None, normalized, entry["normalized"]).ratio()
            if ratio >= 0.975:
                _CACHE.move_to_end(cache_key)
                return entry["response"], round(ratio, 4)
    return None


def cache_store(prompt, category, model, response_text):
    if not response_text or re.search(r"\b(today|current|latest|now|weather|news|price|stock|score)\b", prompt.lower()):
        return
    normalized = re.sub(r"\W+", " ", prompt.lower()).strip()
    key = (category, model, hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16])
    with _CACHE_LOCK:
        _CACHE[key] = {"normalized": normalized, "response": response_text}
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_LIMIT:
            _CACHE.popitem(last=False)


def complexity_score(prompt, category, history):
    words = len(prompt.split())
    score = min(1.0, words / 220)
    score += 0.23 if category in ("coding", "math") else 0
    score += 0.12 if re.search(r"\b(compare|trade.?off|analyze|architecture|step by step|reason|prove|derive)\b", prompt.lower()) else 0
    score += 0.08 if len(history) >= 6 else 0
    return min(1.0, score)


def feedback_adjustment(category, model):
    entry = _POLICY_FEEDBACK.get(f"{category}:{model}", {"count": 0, "reward_sum": 0})
    if not entry["count"]:
        return 0.0
    average = entry["reward_sum"] / entry["count"]
    return (average - 0.5) * min(0.3, 0.08 * math.sqrt(entry["count"]))


def choose_model(prompt, category, preference, selected, history):
    models, tags = model_catalog()
    cloud_available = bool(os.getenv("GEMINI_API_KEY"))
    local_available = ollama_aliases(tags)
    available_ids = {model["id"] for model in models}

    if preference == "manual":
        if selected in GEMINI_MODELS and cloud_available:
            return selected, "manual model selection", "gemini"
        if selected in local_available:
            return selected, "manual local Ollama model", "ollama"
        if selected in GEMINI_MODELS:
            raise ValueError("Gemini is not configured on this backend. Set GEMINI_API_KEY or choose an installed Ollama model.")
        if selected in OLLAMA_MODELS:
            raise ValueError(f"Ollama model '{selected}' is not installed or Ollama is not running. Run `ollama pull {OLLAMA_MODELS[selected][0]}` and retry.")
        raise ValueError("Select an available Gemini or Ollama model.")

    candidates = []
    score = complexity_score(prompt, category, history)
    if cloud_available:
        candidates.extend(["gemini-flash", "gemini-pro"])
    if category == "coding":
        candidates.extend(alias for alias in ("qwen-coder", "codellama", "phi-3", "tinyllama") if alias in local_available)
    elif category == "math":
        candidates.extend(alias for alias in ("qwen-math", "deepseek-math", "phi-3", "tinyllama") if alias in local_available)
    else:
        candidates.extend(alias for alias in ("phi-3", "tinyllama", "qwen-coder") if alias in local_available)
    candidates = list(dict.fromkeys(model for model in candidates if model in available_ids))
    if not candidates:
        raise ValueError("No model is available. Configure GEMINI_API_KEY or start Ollama and install a model.")

    # Lightweight contextual bandit: task category/complexity sets the prior;
    # explicit thumbs-up/down feedback adjusts the model's future score.
    best_model = candidates[0]
    best_score = float("-inf")
    for model in candidates:
        is_local = model in local_available
        prior = 0.0
        if model == "gemini-pro":
            prior = (0.40 + score * 0.75) - (0.10 if category == "generic" else 0)
        elif model == "gemini-flash":
            prior = 0.78 - score * 0.25
        elif model in ("qwen-coder", "codellama"):
            prior = 0.83 if category == "coding" else 0.40
        elif model in ("qwen-math", "deepseek-math"):
            prior = 0.82 if category == "math" else 0.40
        elif model == "phi-3":
            prior = 0.78 if category == "generic" else 0.62
        elif model == "tinyllama":
            prior = 0.64
        if is_local:
            prior += 0.07  # Local inference has no Gemini API charge.
        prior += feedback_adjustment(category, model)
        if prior > best_score:
            best_score, best_model = prior, model

    reasons = {
        "gemini-pro": "higher-complexity task routed to the stronger reasoning model",
        "gemini-flash": "fast, low-cost cloud route",
        "qwen-coder": "coding task routed to the installed local coder model",
        "codellama": "coding task routed to an installed local code model",
        "qwen-math": "math task routed to the installed local math model",
        "deepseek-math": "math task routed to the installed local math model",
        "phi-3": "general task routed to a local model",
        "tinyllama": "general task routed to the available lightweight local model",
    }
    return best_model, reasons.get(best_model, "adaptive model selection"), "ollama" if best_model in local_available else "gemini"


def clean_provider_error(error, provider):
    message = str(error).lower()
    if "resource_exhausted" in message or "429" in message or "quota" in message:
        return "Gemini quota is currently unavailable. Try Gemini Flash, wait for quota to reset, or use an installed local Ollama model.", 429
    if "not configured" in message or "api key" in message or "permission_denied" in message or "401" in message:
        return "Gemini could not authenticate. Check GEMINI_API_KEY in the backend environment.", 503
    if provider == "ollama":
        return f"Could not reach Ollama at {OLLAMA_BASE_URL}. Start Ollama and confirm the selected model is installed.", 503
    if "403" in message or "forbidden" in message:
        return "Gemini denied access to this model. Check that the API key has access to the selected model.", 503
    if "404" in message or "not found" in message:
        return "The selected Gemini model is unavailable. Choose another model or try again after the model service recovers.", 503
    if any(term in message for term in ("unavailable", "503", "deadline_exceeded", "timed out", "timeout")):
        return "Gemini is temporarily unavailable or took too long to respond. Retry in a moment or choose Gemini Flash.", 503
    if any(term in message for term in ("blocked", "safety", "prompt_feedback")):
        return "Gemini blocked this request. Rephrase it and try again.", 400
    return "The language model could not generate a response. Please retry in a moment.", 502


def generate_gemini(model_alias, prompt, history, on_delta=None):
    client = gemini_client()
    contents = [{"role": item["role"], "parts": [{"text": item["content"]}]} for item in history]
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    if on_delta:
        from google.genai import types

        response_text = []
        usage = None
        stream = client.models.generate_content_stream(
            model=GEMINI_MODELS[model_alias],
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=8192),
        )
        for chunk in stream:
            piece = chunk.text or ""
            if piece:
                response_text.append(piece)
                on_delta(piece)
            usage = getattr(chunk, "usage_metadata", None) or usage
        response_text = "".join(response_text).strip()
    else:
        response = client.models.generate_content(model=GEMINI_MODELS[model_alias], contents=contents)
        response_text = (response.text or "").strip()
        usage = getattr(response, "usage_metadata", None)
    usage_data = {
        "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "completion_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
        "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
        "source": "provider usage metadata" if usage else "unavailable",
    }
    return response_text, usage_data


def generate_ollama(model_alias, prompt, history, on_delta=None):
    models = ollama_aliases()
    actual_model = models.get(model_alias)
    if not actual_model:
        raise RuntimeError("Selected Ollama model is not installed or Ollama is not running.")
    messages = [{"role": item["role"], "content": item["content"]} for item in history]
    messages.append({"role": "user", "content": prompt})
    if on_delta:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": actual_model, "messages": messages, "stream": True, "options": {"num_predict": 4096}},
            timeout=(4, 600),
            stream=True,
        )
        response.raise_for_status()
        response_parts = []
        last_data = {}
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            piece = str(data.get("message", {}).get("content", ""))
            if piece:
                response_parts.append(piece)
                on_delta(piece)
            last_data = data
        response_text = "".join(response_parts).strip()
        usage = {
            "prompt_tokens": last_data.get("prompt_eval_count"),
            "completion_tokens": last_data.get("eval_count"),
            "total_tokens": (last_data.get("prompt_eval_count") or 0) + (last_data.get("eval_count") or 0),
            "source": "Ollama usage metadata",
        }
    else:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": actual_model, "messages": messages, "stream": False, "options": {"num_predict": 1024}},
            timeout=(4, 180),
        )
        response.raise_for_status()
        data = response.json()
        response_text = str(data.get("message", {}).get("content", "")).strip()
        usage = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "total_tokens": (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0),
            "source": "Ollama usage metadata",
        }
    return response_text, usage, actual_model


def gemini_cost(model_alias, prompt_tokens, output_tokens):
    # Standard paid-tier text rates, USD per million tokens. This is a list-price
    # estimate; free-tier use, discounts, long-context pricing and billing can differ.
    rates = {
        "gemini-flash": (0.30, 2.50),
        "gemini": (0.30, 2.50),
        "gemini-pro": (1.25, 10.00),
    }
    input_rate, output_rate = rates[model_alias]
    return (prompt_tokens * input_rate + output_tokens * output_rate) / 1_000_000


@app.get("/api/health")
def health():
    return jsonify({"status": "healthy", "framework_initialized": True, "timestamp": utc_now()})


@app.get("/api/status")
def status():
    models, tags = model_catalog()
    return jsonify({
        "initialized": True,
        "mode": "hybrid-gemini-ollama",
        "optimizer": "conservative-and-redundancy-pruning",
        "routing_policy": "contextual-bandit-with-user-feedback",
        "gemini_available": bool(os.getenv("GEMINI_API_KEY")),
        "ollama_available": bool(tags),
        "ollama_base_url": OLLAMA_BASE_URL,
        "installed_ollama_models": tags,
        "available_models": models,
        "semantic_metrics": {
            "gemini": "Gemini text embeddings" if os.getenv("GEMINI_API_KEY") else None,
            "ollama": local_embedding_model() if tags else None,
            "report_similarity_threshold": REPORT_BENCHMARKS["semantic_similarity_minimum"],
        },
        "report_benchmarks": REPORT_BENCHMARKS,
        "semantic_cache_entries": len(_CACHE),
        "feedback_observations": sum(value["count"] for value in _POLICY_FEEDBACK.values()),
        "timestamp": utc_now(),
    })


@app.get("/api/models")
def models_endpoint():
    models, tags = model_catalog()
    return jsonify({"models": models, "ollama_available": bool(tags), "gemini_available": bool(os.getenv("GEMINI_API_KEY"))})


@app.get("/api/strategies")
def strategies():
    models, tags = model_catalog()
    return jsonify({
        "strategies": {
            "conservative": {"description": "Safe cleanup; prioritizes preserving the prompt", **STRATEGY_TARGETS["conservative"]},
            "balanced": {"description": "Safe cleanup plus duplicate-sentence pruning", **STRATEGY_TARGETS["balanced"]},
            "aggressive": {"description": "The same meaning-safe rules with duplicate pruning", **STRATEGY_TARGETS["aggressive"]},
        },
        "categories": ["coding", "math", "generic"],
        "models": models,
        "gemini_available": bool(os.getenv("GEMINI_API_KEY")),
        "ollama_available": bool(tags),
        "report_benchmarks": REPORT_BENCHMARKS,
    })


@app.post("/api/feedback")
def feedback():
    data = request.get_json(silent=True) or {}
    model = str(data.get("model", "")).strip()
    category = data.get("category") if data.get("category") in ("coding", "math", "generic") else "generic"
    rating = data.get("rating")
    if not model or rating not in (1, -1):
        return jsonify({"success": False, "error": "A model and a +1 or -1 rating are required."}), 400
    key = f"{category}:{model}"
    with _POLICY_LOCK:
        entry = _POLICY_FEEDBACK.setdefault(key, {"count": 0, "reward_sum": 0})
        entry["count"] += 1
        entry["reward_sum"] += 1 if rating > 0 else 0
    return jsonify({"success": True, "message": "Feedback recorded for adaptive routing.", "timestamp": utc_now()})


def _process_payload(data, started, on_delta=None, on_progress=None):
    def progress(message):
        if on_progress:
            on_progress(message)

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return {"success": False, "error": "Prompt is required", "timestamp": utc_now()}, 400
    if len(prompt) > 50000:
        return {"success": False, "error": "Prompts are limited to 50,000 characters."}, 413

    progress("Checking the prompt and conversation context…")
    history = normalize_history(data.get("history", []))
    category = classify(prompt)
    strategy = str(data.get("strategy", "balanced")).lower()
    if strategy not in STRATEGY_TARGETS:
        strategy = "balanced"
    optimized, strategy_used = optimize_prompt(prompt, strategy, category)

    preference = "manual" if data.get("model_preference") == "manual" else "auto"
    selected = str(data.get("selected_model") or "gemini-flash")
    try:
        chosen, route_reason, provider = choose_model(prompt, category, preference, selected, history)
    except ValueError as exc:
        return {"success": False, "error": str(exc), "timestamp": utc_now()}, 400

    progress("Checking semantic similarity against the 0.90 report threshold…")
    candidate_similarity, candidate_similarity_method = semantic_similarity(prompt, optimized, provider)
    progress("Measuring report token counts with the selected model…")
    original_tokens, original_token_method = count_prompt_tokens(prompt, history, provider, chosen)
    if optimized == prompt:
        candidate_tokens, candidate_token_method = original_tokens, original_token_method
    else:
        candidate_tokens, candidate_token_method = count_prompt_tokens(optimized, history, provider, chosen)
    candidate_reduction = ((original_tokens - candidate_tokens) / original_tokens * 100) if original_tokens else 0
    similarity_guard_triggered = optimized != prompt and (
        candidate_similarity is None or candidate_similarity < REPORT_BENCHMARKS["semantic_similarity_minimum"]
    )
    if similarity_guard_triggered:
        optimized = prompt
        strategy_used = "original prompt retained by semantic-similarity guard"
        optimized_tokens, optimized_token_method = original_tokens, original_token_method
        similarity, similarity_method = 1.0, "exact text match (candidate optimization rejected)"
    else:
        optimized_tokens, optimized_token_method = candidate_tokens, candidate_token_method
        similarity, similarity_method = candidate_similarity, candidate_similarity_method
    tokens_saved = original_tokens - optimized_tokens
    reduction = (tokens_saved / original_tokens * 100) if original_tokens else 0
    minimum_similarity = STRATEGY_TARGETS[strategy]["min_similarity"]
    target_reduction = STRATEGY_TARGETS[strategy]["target_reduction"]

    result = {
        "original_prompt": prompt,
        "optimized_prompt": optimized,
        "strategy_used": strategy,
        "optimization_method": strategy_used,
        "token_reduction_percent": round(reduction, 1),
        "similarity": similarity,
        "similarity_method": similarity_method,
        "optimization_candidate_similarity": candidate_similarity,
        "optimization_candidate_similarity_method": candidate_similarity_method,
        "optimization_candidate_token_reduction_percent": round(candidate_reduction, 1),
        "semantic_guard_triggered": similarity_guard_triggered,
        "target_achieved": None if similarity is None else reduction >= target_reduction and similarity >= minimum_similarity,
        "report_benchmarks": REPORT_BENCHMARKS,
        "selected_llm": chosen,
        "model_preference": preference,
        "routing_policy": "contextual-bandit-with-user-feedback" if preference == "auto" else "manual",
        "routing_reason": route_reason,
        "complexity_score": round(complexity_score(prompt, category, history), 2),
        "history_turns": sum(item["role"] == "user" for item in history),
        "category": category,
        "metrics": {
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "token_count_method": original_token_method if original_token_method == optimized_token_method else f"original: {original_token_method}; optimized: {optimized_token_method}",
            "target_reduction_percent": target_reduction,
            "minimum_similarity": minimum_similarity,
            "similarity_method": similarity_method,
            "optimization_candidate_similarity": candidate_similarity,
            "semantic_guard_triggered": similarity_guard_triggered,
        },
        "actual_usage": None,
        "cost": None,
        "cache_hit": False,
    }

    if data.get("include_response", True):
        cache_entry = cache_lookup(prompt, category, chosen, history)
        if cache_entry:
            result["response"] = cache_entry[0]
            result["cache_hit"] = True
            result["cache_similarity"] = cache_entry[1]
            result["response_complete"] = True
            result["actual_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "source": "similarity cache"}
            if on_delta:
                on_delta(cache_entry[0])
        else:
            try:
                progress(f"Generating a response with {chosen}…")
                streamed_parts = []

                def send_delta(piece):
                    if piece:
                        streamed_parts.append(piece)
                        if on_delta:
                            on_delta(piece)

                if provider == "ollama":
                    generated_text, usage, actual_model = generate_ollama(
                        chosen, optimized, history, on_delta=send_delta if on_delta else None
                    )
                    result["provider_model"] = actual_model
                else:
                    try:
                        generated_text, usage = generate_gemini(
                            chosen, optimized, history, on_delta=send_delta if on_delta else None
                        )
                    except Exception as exc:
                        provider_message = str(exc).lower()
                        can_fallback = chosen == "gemini-pro" and not streamed_parts and (
                            "resource_exhausted" in provider_message or "429" in provider_message
                        )
                        if not can_fallback:
                            raise
                        app.logger.info("Gemini Pro quota unavailable; retrying with Flash")
                        chosen, route_reason, provider = "gemini-flash", "Gemini Pro quota unavailable; fell back to Gemini Flash", "gemini"
                        generated_text, usage = generate_gemini(
                            chosen, optimized, history, on_delta=send_delta if on_delta else None
                        )
                        result["model_fallback_notice"] = "Gemini Pro quota was unavailable, so this reply was generated with Gemini Flash."
                        result["selected_llm"] = chosen
                        result["routing_reason"] = route_reason

                if not generated_text:
                    return {"success": False, "error": "The model returned an empty response. Please rephrase your prompt and retry."}, 502
                result["response"] = generated_text
                result["response_complete"] = True
                result["actual_usage"] = usage
                cache_store(prompt, category, chosen, generated_text)

            except Exception as exc:
                app.logger.exception("Model generation failed (%s)", provider)
                partial_response = "".join(streamed_parts).strip()
                if partial_response:
                    result["response"] = partial_response
                    result["response_complete"] = False
                    result["generation_warning"] = (
                        "The model connection stopped before completion was confirmed. "
                        "This partial answer is shown; retry your request to get a complete response."
                    )
                    result["actual_usage"] = {
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "source": "unavailable after interrupted stream",
                    }
                else:
                    message, status_code = clean_provider_error(exc, provider)
                    return {"success": False, "error": message, "timestamp": utc_now()}, status_code

        if provider == "gemini" and result.get("response"):
            usage = result.get("actual_usage") or {}
            output_tokens = usage.get("completion_tokens") or estimate_tokens(result.get("response", ""))
            embedding_was_used = similarity is not None and similarity_method.startswith("Gemini text embedding")
            embedding_tokens = estimate_tokens(prompt) + estimate_tokens(optimized) if embedding_was_used else 0
            embedding_cost = embedding_tokens * GEMINI_EMBEDDING_USD_PER_MILLION_TOKENS / 1_000_000
            original_cost = gemini_cost(chosen, original_tokens, output_tokens)
            current_generation_cost = 0 if result.get("cache_hit") else gemini_cost(chosen, optimized_tokens, output_tokens)
            optimized_cost = current_generation_cost + embedding_cost
            estimated_savings = original_cost - optimized_cost
            result["cost"] = {
                "original_cost_usd": round(original_cost, 8),
                "optimized_cost_usd": round(optimized_cost, 8),
                "estimated_savings_usd": round(estimated_savings, 8),
                "estimated_savings_percent": round((estimated_savings / original_cost) * 100, 2) if original_cost else 0,
                "estimated_current_request_usd": round(optimized_cost, 8),
                "embedding_cost_usd": round(embedding_cost, 8),
                "cost_basis": "Same selected Gemini model; raw versus optimized full prompt context. Includes the paid-tier embedding estimate and cache savings when applicable.",
                "model_pricing": "Estimated standard paid-tier Gemini list prices; free-tier use and actual billing may differ",
                "input_usd_per_million_tokens": 1.25 if chosen == "gemini-pro" else 0.30,
                "output_usd_per_million_tokens": 10.0 if chosen == "gemini-pro" else 2.5,
            }

    return {"success": True, "data": result, "processing_time": round(time.time() - started, 3), "timestamp": utc_now()}, 200


@app.post("/api/process")
def process():
    started = time.time()
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON", "timestamp": utc_now()}), 400
    payload, status_code = _process_payload(request.get_json(silent=True) or {}, started)
    return jsonify(payload), status_code


@app.post("/api/process/stream")
def process_stream():
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON", "timestamp": utc_now()}), 400
    data = request.get_json(silent=True) or {}
    if not str(data.get("prompt", "")).strip():
        return jsonify({"success": False, "error": "Prompt is required", "timestamp": utc_now()}), 400

    events = queue.Queue()
    started = time.time()

    def run_request():
        try:
            payload, status_code = _process_payload(
                data,
                started,
                on_delta=lambda value: events.put({"type": "delta", "text": value}),
                on_progress=lambda message: events.put({"type": "progress", "message": message}),
            )
            events.put({"type": "result", "payload": payload, "status_code": status_code})
        except Exception as exc:
            app.logger.exception("Streamed model request failed")
            events.put({"type": "error", "message": "The model could not complete this request. Please retry."})

    def encode_event(value):
        return f"data: {json.dumps(value, ensure_ascii=False)}\n\n"

    def generate_events():
        yield encode_event({"type": "progress", "message": "Waking the model service and preparing your request…"})
        threading.Thread(target=run_request, daemon=True).start()
        while True:
            try:
                event = events.get(timeout=10)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if event["type"] in ("delta", "progress"):
                yield encode_event(event)
                continue
            if event["type"] == "result":
                payload = event["payload"]
                if event["status_code"] >= 400 or not payload.get("success"):
                    yield encode_event({"type": "error", "message": payload.get("error", "The request could not be completed.")})
                else:
                    yield encode_event({"type": "complete", "response": payload})
                return
            yield encode_event(event)
            return

    return Response(
        stream_with_context(generate_events()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
