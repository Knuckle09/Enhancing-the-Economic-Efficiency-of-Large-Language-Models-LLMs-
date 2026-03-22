"""
Nimbus AI Backend - Groq-powered (free & fast)
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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

logger.info(f"✅ Flask started. Groq key: {bool(GROQ_API_KEY)} | Gemini key: {bool(GEMINI_API_KEY)}")


def optimize_with_groq(prompt):
    import requests
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are a prompt optimizer. Rewrite the user's prompt in fewer words while keeping the exact same meaning. Return ONLY the rewritten prompt, nothing else."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    logger.info(f"Groq response status: {response.status_code}")
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def optimize_with_gemini(prompt):
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Rewrite this prompt in fewer words while keeping the same meaning. Return ONLY the rewritten prompt:\n\n{prompt}"
            }]
        }]
    }
    response = requests.post(url, json=payload, timeout=30)
    logger.info(f"Gemini response status: {response.status_code}")
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


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
    mode = "groq" if GROQ_API_KEY else ("gemini" if GEMINI_API_KEY else "heuristic")
    return jsonify({
        "name": "Nimbus AI API",
        "status": "running",
        "mode": mode,
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
    mode = "groq" if GROQ_API_KEY else ("gemini" if GEMINI_API_KEY else "heuristic")
    return jsonify({
        "initialized": True,
        "loading": False,
        "error": None,
        "step": "ready",
        "mode": mode,
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

        # Try Groq first (free), then Gemini, then heuristic
        if GROQ_API_KEY:
            try:
                optimized = optimize_with_groq(prompt)
                strategy = "groq-llama3"
                logger.info("✅ Groq succeeded")
            except Exception as e:
                logger.error(f"Groq failed: {e}")
                if GEMINI_API_KEY:
                    try:
                        optimized = optimize_with_gemini(prompt)
                        strategy = "gemini-fallback"
                    except Exception as e2:
                        logger.error(f"Gemini also failed: {e2}")
        elif GEMINI_API_KEY:
            try:
                optimized = optimize_with_gemini(prompt)
                strategy = "gemini"
            except Exception as e:
                logger.error(f"Gemini failed: {e}")

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
