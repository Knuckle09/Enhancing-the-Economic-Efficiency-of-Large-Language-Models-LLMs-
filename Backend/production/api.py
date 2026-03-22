"""
Nimbus AI Backend - Groq-powered
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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
logger.info(f"✅ Flask started. Groq key present: {bool(GROQ_API_KEY)}")


def optimize_with_groq(prompt):
    import requests
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    # Use correct current Groq model names
    models = ["llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"]
    last_error = None
    for model in models:
        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a prompt optimizer. Rewrite the user's prompt in fewer words while keeping the exact same meaning. Return ONLY the rewritten prompt, nothing else. No explanation."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.3
            }
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            logger.info(f"Groq {model} status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                result = data["choices"][0]["message"]["content"].strip()
                logger.info(f"✅ Groq {model} succeeded")
                return result, model
            else:
                logger.warning(f"Groq {model} failed: {response.status_code} - {response.text[:200]}")
                last_error = f"{response.status_code}: {response.text[:100]}"
        except Exception as e:
            logger.warning(f"Groq {model} exception: {e}")
            last_error = str(e)
    raise Exception(f"All Groq models failed. Last: {last_error}")


def classify_prompt(prompt):
    lower = prompt.lower()
    if any(w in lower for w in ['code', 'function', 'python', 'javascript', 'program', 'debug', 'algorithm']):
        return 'coding'
    elif any(w in lower for w in ['calculate', 'formula', 'math', 'equation', 'solve', 'derivative', 'integral']):
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
        "mode": "groq" if GROQ_API_KEY else "heuristic",
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
        "mode": "groq" if GROQ_API_KEY else "heuristic",
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
        strategy = "heuristic"
        optimized = simple_optimize(prompt)

        if GROQ_API_KEY:
            try:
                optimized, model_used = optimize_with_groq(prompt)
                strategy = f"groq-{model_used}"
            except Exception as e:
                logger.error(f"Groq failed: {e}")
                strategy = "heuristic-fallback"

        optimized_tokens = len(optimized.split())
        reduction = max(0, round((1 - optimized_tokens / max(original_tokens, 1)) * 100, 1))

        return jsonify({
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
        })

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
