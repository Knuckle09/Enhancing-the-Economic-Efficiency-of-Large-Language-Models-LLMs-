"""
Flask API - Nimbus AI Backend
Debug version - logs after every single import to find the hang
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

framework_status = {"initialized": False, "error": None, "loading": False, "step": "not started"}
rl_optimizer = None
prompt_tester = None


def initialize_framework():
    global rl_optimizer, prompt_tester, framework_status

    framework_status["loading"] = True
    framework_status["step"] = "starting"

    try:
        framework_status["step"] = "importing torch"
        logger.info("🔄 importing torch...")
        import torch
        logger.info("✅ torch done")

        framework_status["step"] = "importing run"
        logger.info("🔄 importing run...")
        from run import process_prompt
        logger.info("✅ run done")

        framework_status["step"] = "importing rl_optimizer"
        logger.info("🔄 importing rl_optimizer...")
        from rl_optimizer import RLOptimizer, get_latest_training_data_file
        logger.info("✅ rl_optimizer done")

        framework_status["step"] = "importing prompt_diversity_test"
        logger.info("🔄 importing prompt_diversity_test...")
        from prompt_diversity_test import PromptDiversityTester
        logger.info("✅ prompt_diversity_test done")

        framework_status["step"] = "loading training data"
        logger.info("🔄 loading training data...")
        training_data_file = get_latest_training_data_file("./results")
        logger.info(f"✅ training data: {training_data_file}")

        framework_status["step"] = "creating RLOptimizer"
        logger.info("🔄 creating RLOptimizer...")
        rl_optimizer = RLOptimizer(training_data_file)
        logger.info("✅ RLOptimizer created")

        framework_status["step"] = "creating PromptDiversityTester"
        logger.info("🔄 creating PromptDiversityTester...")
        prompt_tester = PromptDiversityTester()
        logger.info("✅ PromptDiversityTester created")

        model_path = "./text_optimizer_ppo.zip"
        framework_status["step"] = f"loading model from {model_path}"
        logger.info(f"🔄 loading model from {model_path}...")
        if os.path.exists(model_path):
            from stable_baselines3 import PPO
            rl_optimizer.model = PPO.load(model_path)
            logger.info("✅ model loaded")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

        framework_status = {
            "initialized": True,
            "loading": False,
            "error": None,
            "step": "ready",
            "timestamp": datetime.now().isoformat()
        }
        logger.info("✅ Framework fully ready!")

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        logger.error(f"❌ Init failed at step '{framework_status.get('step')}':\n{err}")
        framework_status = {
            "initialized": False,
            "loading": False,
            "error": err,
            "step": framework_status.get("step", "unknown"),
            "timestamp": datetime.now().isoformat()
        }


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Nimbus AI API",
        "status": "running",
        "framework": framework_status,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "framework_initialized": framework_status["initialized"],
        "step": framework_status.get("step"),
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

    if not framework_status["initialized"]:
        return jsonify({
            "success": False,
            "error": f"Framework not ready. Current step: {framework_status.get('step')}. Loading: {framework_status.get('loading')}",
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

    t = threading.Thread(target=initialize_framework, daemon=True)
    t.start()

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
