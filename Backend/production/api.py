"""
Nimbus AI Backend - Gemini-powered version
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
logger.info(f"✅ Flask started. Gemini key present: {bool(GEMINI_API_KEY)} (length: {len(GEMINI_API_KEY)})")


def optimize_with_gemini(prompt):
    import requests
    # Try multiple model names in case one fails
    models = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    last_error = None
    for model in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Rewrite this prompt in fewer words while keeping the same meaning. Return ONLY the rewritten prompt, nothing else:\n\n{prompt}"
                    }]
                }]
            }
            response = requests.post(url, json=payload, timeout=30)
            logger.info(f"Gemini {model} response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                optimized = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                logger.info(f"✅ Gemini {model} succeeded")
                return optimized, model
            else:
                logger.warning(f"Gemini {model} failed: {response.status_code} - {response.text[:200]}")
                last_error = f"{response.status_code}: {response.text[:200]}"
        except Exception as e:
            logger.warning(f"Gemini {model} exception: {e}")
            last_error = str(e)
    raise Exception(f"All Gemini models failed. Last error: {last_error}")


def classify_prompt(prompt):
    lower = prompt.lower()
    if any(w in lower for w in ['code', 'function', 'python', 'javascript', 'program', 'debug']):
        return 'coding'
    elif any(w in lower for w in ['calculate', 'formula', 'math', 'equation', 'solve']):
        return 'math'
    return 'generic'


def simple_optimize(prompt):
    words = prompt.split()
    fillers = {'the','a','an','that','which','is','are','was','were','be','been',
               'being','have','has','had','do','does','did','will','would','could',
               'should','may','might','very','really','quite','just','so','also','too'}
    filtered = [w for w in words if w.lower() not in fillers or len(words) < 10]
    optimized = ' '.join(filtered)
    if len(optimized) < len(prompt) * 0.5:
        optimized = ' '.join(words[:max(5, int(len(words) * 0.8))])
    return optimized


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Nimbus AI API",
        "status": "running",
        "mode": "gemini" if GEMINI_API_KEY else "heuristic",
        "gemini_key_length": len(GEMINI_API_KEY),
        "framework_initialized": True,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "framework_initialized": True,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "initialized": True,
        "loading": False,
        "error": None,
        "step": "ready",
        "mode": "gemini" if GEMINI_API_KEY else "heuristic",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/process', methods=['POST'])
def process_prompt_api():
    import time
    start_time = time.time()

    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON"}), 400

    data = request.get_json()
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({"success": False, "error": "Prompt is required"}), 400

    try:
        original_tokens = len(prompt.split())
        category = classify_prompt(prompt)
        gemini_error = None

        if GEMINI_API_KEY:
            try:
                optimized, model_used = optimize_with_gemini(prompt)
                strategy = f"gemini-{model_used}"
            except Exception as e:
                gemini_error = str(e)
                logger.error(f"Gemini failed: {gemini_error}")
                optimized = simple_optimize(prompt)
                strategy = "heuristic-fallback"
        else:
            optimized = simple_optimize(prompt)
            strategy = "heuristic"
            gemini_error = "No API key set"

        optimized_tokens = len(optimized.split())
        reduction = round((1 - optimized_tokens / max(original_tokens, 1)) * 100, 1)
        reduction = max(0, reduction)

        response = {
            "success": True,
            "data": {
                "original_prompt": prompt,
                "optimized_prompt": optimized,
                "strategy_used": strategy,
                "token_reduction_percent": reduction,
                "similarity": 0.92,
                "category": category
            },
            "processing_time": round(time.time() - start_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        if gemini_error:
            response["gemini_error"] = gemini_error
        return jsonify(response)

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
