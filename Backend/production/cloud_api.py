"""Small Gemini API for the hosted UI.

The full RL/NLP stack remains available in api.py for local use with Ollama.
This entry point keeps the free hosted service within its memory and build
limits while preserving the frontend's API contract.
"""

from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
import os
import re
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GEMINI_MODELS = {
    "gemini-pro": "gemini-2.5-pro",
    "gemini-flash": "gemini-2.5-flash",
    "gemini": "gemini-2.5-flash",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def compact_prompt(prompt):
    """Apply conservative, meaning-preserving cleanup before model inference."""
    cleaned = re.sub(r"\s+", " ", prompt).strip()
    replacements = (
        (r"\bI would like you to\b", ""),
        (r"\bI want you to\b", ""),
        (r"\bCould you please\b", ""),
        (r"\bCan you please\b", ""),
        (r"\bplease\b", ""),
        (r"\bkindly\b", ""),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def classify(prompt):
    text = prompt.strip().lower()

    # Avoid substring matches such as "functionality" and only classify
    # ambiguous terms like "function" when the prompt asks for code work.
    if "```" in text or re.search(
        r"\b(?:python|javascript|typescript|java|c\+\+|c#|rust|golang|sql|bash|powershell)\b",
        text,
    ):
        return "coding"

    asks_for_code = re.search(
        r"\b(?:write|create|implement|build|debug|fix|refactor|review|generate|complete|optimize)\b",
        text,
    )
    names_code = re.search(
        r"\b(?:code|coding|programming|script|program|api endpoint|sql query)\b",
        text,
    )
    asks_for_callable = re.search(
        r"\b(?:write|create|implement|build|debug|fix|refactor|review|generate|complete)\b",
        text,
    ) and re.search(r"\b(?:function|method|class)\b", text)
    if (asks_for_code and names_code) or asks_for_callable:
        return "coding"

    if re.search(r"\b(?:calculate|solve|compute|derive|prove|integrate|differentiate)\b", text) or re.search(
        r"\b(?:equation|math|mathematics|integral|derivative)\b", text
    ):
        return "math"
    return "generic"


@lru_cache(maxsize=1)
def gemini_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Gemini is not configured. Add GEMINI_API_KEY to the backend service environment.")
    from google import genai
    return genai.Client(api_key=key)


@app.get("/api/health")
def health():
    return jsonify({"status": "healthy", "framework_initialized": True, "timestamp": utc_now()})


@app.get("/api/status")
def status():
    return jsonify({
        "initialized": True,
        "mode": "gemini-cloud",
        "optimizer": "lightweight",
        "gemini_available": bool(os.getenv("GEMINI_API_KEY")),
        "ollama_available": False,
        "timestamp": utc_now(),
    })


@app.get("/api/strategies")
def strategies():
    return jsonify({
        "strategies": {
            "conservative": {"description": "Conservative prompt cleanup", "target_reduction": 15, "min_similarity": 0.90},
            "balanced": {"description": "Balanced prompt cleanup", "target_reduction": 30, "min_similarity": 0.85},
            "aggressive": {"description": "More compact prompt cleanup", "target_reduction": 35, "min_similarity": 0.85},
        },
        "categories": ["coding", "math", "generic"],
        "llms": ({"gemini_coding": "gemini-pro", "gemini_math": "gemini-pro", "gemini_generic": "gemini-flash"}
                 if os.getenv("GEMINI_API_KEY") else {}),
        "gemini_available": bool(os.getenv("GEMINI_API_KEY")),
        "ollama_available": False,
    })


@app.post("/api/process")
def process():
    started = time.time()
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON", "timestamp": utc_now()}), 400

    data = request.get_json() or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"success": False, "error": "Prompt is required", "timestamp": utc_now()}), 400

    preference = data.get("model_preference", "auto")
    selected = data.get("selected_model") or "gemini-flash"
    if preference == "manual" and selected not in GEMINI_MODELS:
        return jsonify({
            "success": False,
            "error": "Ollama models run on your own computer. Start the local API to use them.",
            "timestamp": utc_now(),
        }), 400

    optimized = compact_prompt(prompt)
    original_count = len(prompt.split())
    optimized_count = len(optimized.split())
    reduction = (original_count - optimized_count) / original_count * 100 if original_count else 0
    similarity = SequenceMatcher(None, prompt.lower(), optimized.lower()).ratio()
    strategy = "conservative" if reduction < 15 else "balanced"
    chosen = selected if preference == "manual" else "gemini-flash"
    category = classify(prompt)

    result = {
        "original_prompt": prompt,
        "optimized_prompt": optimized,
        "strategy_used": strategy,
        "token_reduction_percent": round(reduction, 1),
        "similarity": round(similarity, 3),
        "target_achieved": reduction >= 15 and similarity >= 0.85,
        "selected_llm": chosen,
        "model_preference": preference,
        "category": category,
        "metrics": {"original_tokens": original_count, "optimized_tokens": optimized_count, "tokens_saved": original_count - optimized_count},
        # Pricing is deliberately omitted until an up-to-date model rate is configured.
        "cost": None,
    }

    if data.get("include_response", True):
        try:
            # Keep a strong reference to the SDK client for the full request.
            # Recent google-genai versions can close an inline temporary client
            # before the HTTP request completes.
            client = gemini_client()
            response = client.models.generate_content(
                model=GEMINI_MODELS[chosen],
                contents=optimized,
            )
            generated_text = (response.text or "").strip()
            if not generated_text:
                app.logger.error("Gemini returned an empty response for model %s", GEMINI_MODELS[chosen])
                return jsonify({
                    "success": False,
                    "error": "Gemini returned an empty response. Please try rephrasing your prompt.",
                }), 502
            result["response"] = generated_text
        except Exception as exc:
            app.logger.exception("Gemini generation failed")
            message = str(exc)
            if "not configured" in message.lower():
                return jsonify({"success": False, "error": message}), 503
            return jsonify({
                "success": False,
                "error": "Gemini could not generate a response. Please retry in a moment.",
                "detail": message,
            }), 502

    return jsonify({
        "success": True,
        "data": result,
        "processing_time": round(time.time() - started, 2),
        "timestamp": utc_now(),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
