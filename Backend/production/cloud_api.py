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
import logging
import math
import os
import re
import threading
import time

import requests
from flask import Flask, jsonify, request
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
    "balanced": {"target_reduction": 30, "min_similarity": 0.85},
    "aggressive": {"target_reduction": 35, "min_similarity": 0.80},
}
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


def similarity_proxy(original, optimized):
    """Text-overlap proxy only; it is not represented as an embedding score."""
    if original == optimized:
        return 1.0
    return round(SequenceMatcher(None, original.lower(), optimized.lower()).ratio(), 4)


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
    return "The language model could not generate a response. Please retry in a moment.", 502


def generate_gemini(model_alias, prompt, history):
    client = gemini_client()
    contents = [{"role": item["role"], "parts": [{"text": item["content"]}]} for item in history]
    contents.append({"role": "user", "parts": [{"text": prompt}]})
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


def generate_ollama(model_alias, prompt, history):
    models = ollama_aliases()
    actual_model = models.get(model_alias)
    if not actual_model:
        raise RuntimeError("Selected Ollama model is not installed or Ollama is not running.")
    messages = [{"role": item["role"], "content": item["content"]} for item in history]
    messages.append({"role": "user", "content": prompt})
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


@app.post("/api/process")
def process():
    started = time.time()
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON", "timestamp": utc_now()}), 400

    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"success": False, "error": "Prompt is required", "timestamp": utc_now()}), 400
    if len(prompt) > 50000:
        return jsonify({"success": False, "error": "Prompts are limited to 50,000 characters."}), 413

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
        return jsonify({"success": False, "error": str(exc), "timestamp": utc_now()}), 400

    original_tokens = estimate_tokens(prompt)
    optimized_tokens = estimate_tokens(optimized)
    tokens_saved = max(0, original_tokens - optimized_tokens)
    reduction = (tokens_saved / original_tokens * 100) if original_tokens else 0
    similarity = similarity_proxy(prompt, optimized)
    minimum_similarity = STRATEGY_TARGETS[strategy]["min_similarity"]
    target_reduction = STRATEGY_TARGETS[strategy]["target_reduction"]

    result = {
        "original_prompt": prompt,
        "optimized_prompt": optimized,
        "strategy_used": strategy,
        "optimization_method": strategy_used,
        "token_reduction_percent": round(reduction, 1),
        "similarity": similarity,
        "similarity_method": "character-overlap proxy; not a transformer embedding",
        "target_achieved": reduction >= target_reduction and similarity >= minimum_similarity,
        "selected_llm": chosen,
        "model_preference": preference,
        "routing_policy": "contextual-bandit-with-user-feedback" if preference == "auto" else "manual",
        "routing_reason": route_reason,
        "complexity_score": round(complexity_score(prompt, category, history), 2),
        "category": category,
        "metrics": {
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "token_count_method": "estimated at approximately 4 characters per token",
            "target_reduction_percent": target_reduction,
            "minimum_similarity": minimum_similarity,
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
            result["actual_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "source": "similarity cache"}
        else:
            try:
                if provider == "ollama":
                    generated_text, usage, actual_model = generate_ollama(chosen, optimized, history)
                    result["provider_model"] = actual_model
                else:
                    try:
                        generated_text, usage = generate_gemini(chosen, optimized, history)
                    except Exception as exc:
                        provider_message = str(exc).lower()
                        if chosen != "gemini-pro" or not ("resource_exhausted" in provider_message or "429" in provider_message):
                            raise
                        app.logger.info("Gemini Pro quota unavailable; retrying with Flash")
                        chosen, route_reason, provider = "gemini-flash", "Gemini Pro quota unavailable; fell back to Gemini Flash", "gemini"
                        generated_text, usage = generate_gemini(chosen, optimized, history)
                        result["model_fallback_notice"] = "Gemini Pro quota was unavailable, so this reply was generated with Gemini Flash."
                        result["selected_llm"] = chosen
                        result["routing_reason"] = route_reason

                if not generated_text:
                    return jsonify({"success": False, "error": "The model returned an empty response. Please rephrase your prompt and retry."}), 502
                result["response"] = generated_text
                result["actual_usage"] = usage
                cache_store(prompt, category, chosen, generated_text)

                if provider == "gemini":
                    output_tokens = usage.get("completion_tokens") or estimate_tokens(generated_text)
                    context_tokens = sum(estimate_tokens(item["content"]) for item in history)
                    base_input = estimate_tokens(prompt) + context_tokens
                    optimized_input = estimate_tokens(optimized) + context_tokens
                    original_cost = gemini_cost(chosen, base_input, output_tokens)
                    optimized_cost = gemini_cost(chosen, optimized_input, output_tokens)
                    result["cost"] = {
                        "original_cost_usd": round(original_cost, 8),
                        "optimized_cost_usd": round(optimized_cost, 8),
                        "estimated_savings_usd": round(max(0, original_cost - optimized_cost), 8),
                        "estimated_current_request_usd": round(optimized_cost, 8),
                        "model_pricing": "Estimated standard paid-tier text list price; actual billing/free tier may differ",
                        "input_usd_per_million_tokens": 1.25 if chosen == "gemini-pro" else 0.30,
                        "output_usd_per_million_tokens": 10.0 if chosen == "gemini-pro" else 2.5,
                    }
            except Exception as exc:
                app.logger.exception("Model generation failed (%s)", provider)
                message, status_code = clean_provider_error(exc, provider)
                return jsonify({"success": False, "error": message, "timestamp": utc_now()}), status_code

        if result.get("cache_hit") and provider == "gemini":
            result["cost"] = {
                "original_cost_usd": 0,
                "optimized_cost_usd": 0,
                "estimated_savings_usd": 0,
                "estimated_current_request_usd": 0,
                "model_pricing": "No model call was made; response reused from the in-memory similarity cache",
            }

    return jsonify({"success": True, "data": result, "processing_time": round(time.time() - started, 3), "timestamp": utc_now()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
