"""
Flask API - Nimbus AI Backend
Flask binds to port FIRST, then initializes framework in background thread
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import threading
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logger.info("✅ Flask app created, binding to port...")

framework_status = {"initialized": False, "error": None, "loading": False}
rl_optimizer = None
prompt_tester = None


def initialize_framework():
    global rl_optimizer, prompt_tester, framework_status

    if framework_status["initialized"] or framework_status["loading"]:
        return

    framework_status["loading"] = True
    logger.info("🔄 Background: loading modules...")

    try:
        from run import process_prompt
        logger.info("✅ run imported")

        from rl_optimizer import RLOptimizer, get_latest_training_data_file
        logger.info("✅ rl_optimizer imported")

        from prompt_diversity_test import PromptDiversityTester
        logger.info("✅ prompt_diversity_test imported")

        training_data_file = get_latest_training_data_file("./results")
        rl_optimizer = RLOptimizer(training_data_file)
        prompt_tester = PromptDiversityTester()

        model_path = "./text_optimizer_ppo.zip"
        if os.path.exists(model_path):
            from stable_baselines3 import PPO
            rl_optimizer.model = PPO.load(model_path)
            logger.info(f"✅ Loaded RL model: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

        framework_status = {
            "initialized": True,
            "loading": False,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        logger.info("✅ Framework ready")

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        logger.error(f"❌ Init failed:\n{err}")
        framework_status = {
            "initialized": False,
            "loading": False,
            "error": err,
            "timestamp": datetime.now().isoformat()
        }


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Nimbus AI API",
        "status": "running",
        "framework_initialized": framework_status["initialized"],
        "framework_loading": framework_status.get("loading", False),
        "endpoints": ["/api/health", "/api/status", "/api/process"],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "framework_initialized": framework_status["initialized"],
        "framework_loading": framework_status.get("loading", False),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify(framework_status)


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

    if framework_status.get("loading"):
        return jsonify({
            "success": False,
            "error": "Framework is still loading, please try again in a moment."
        }), 503

    if not framework_status["initialized"]:
        return jsonify({
            "success": False,
            "error": framework_status.get("error") or "Framework not initialized."
        }), 503

    try:
        category = prompt_tester.classify_prompt(prompt)
        action, strategy = rl_optimizer.predict_optimal_strategy(prompt, category)
        optimized_prompt, metrics = rl_optimizer.env.apply_optimization_strategy(prompt, action)

        return jsonify({
            "success": True,
            "data": {
                "original_prompt": prompt,
                "optimized_prompt": optimized_prompt,
                "strategy_used": strategy,
                "token_reduction_percent": round(metrics['reduction_percent'], 1),
                "similarity": round(metrics['similarity'], 3),
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
    logger.info(f"🚀 Binding to 0.0.0.0:{port}")

    # Start framework loading in background AFTER Flask binds to port
    t = threading.Thread(target=initialize_framework, daemon=True)
    t.start()

    # Flask starts immediately — port is bound before any imports run
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
