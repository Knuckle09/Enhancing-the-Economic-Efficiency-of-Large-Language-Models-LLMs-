"""
Flask API - Lazy loading version for Render deployment
All heavy imports deferred to avoid startup hang
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# No heavy imports at module level — Flask binds to port instantly
logger.info("✅ Flask app created, binding to port...")

framework_status = {"initialized": False, "error": None}
rl_optimizer = None
prompt_tester = None


def initialize_framework():
    """Lazy-load all heavy modules only when first needed"""
    global rl_optimizer, prompt_tester, framework_status

    if framework_status["initialized"]:
        return True

    try:
        logger.info("🔄 Lazy-loading modules...")

        # Import heavy modules here, not at top of file
        from run import process_prompt
        logger.info("✅ run imported")

        from rl_optimizer import RLOptimizer, get_latest_training_data_file
        logger.info("✅ rl_optimizer imported")

        from prompt_diversity_test import PromptDiversityTester
        logger.info("✅ prompt_diversity_test imported")

        training_data_file = get_latest_training_data_file("./results")
        rl_optimizer = RLOptimizer(training_data_file)
        prompt_tester = PromptDiversityTester()

        model_path = "./models/text_optimizer_ppo.zip"
        if os.path.exists(model_path):
            from stable_baselines3 import PPO
            rl_optimizer.model = PPO.load(model_path)
            logger.info(f"✅ Loaded RL model: {model_path}")
        else:
            framework_status = {
                "initialized": False,
                "error": f"No trained model at {model_path}",
                "timestamp": datetime.now().isoformat()
            }
            return False

        framework_status = {
            "initialized": True,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        logger.info("✅ Framework ready")
        return True

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        logger.error(f"❌ Init failed:\n{err}")
        framework_status = {
            "initialized": False,
            "error": err,
            "timestamp": datetime.now().isoformat()
        }
        return False


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Nimbus AI API",
        "status": "running",
        "framework_initialized": framework_status["initialized"],
        "endpoints": ["/api/health", "/api/status", "/api/process", "/api/init"],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "framework_initialized": framework_status["initialized"],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify(framework_status)


@app.route('/api/init', methods=['GET'])
def init():
    """Manually trigger framework initialization"""
    success = initialize_framework()
    return jsonify({
        "success": success,
        "status": framework_status
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

    if not framework_status["initialized"]:
        # Try to initialize on first request
        initialize_framework()
        if not framework_status["initialized"]:
            return jsonify({
                "success": False,
                "error": framework_status.get("error", "Framework not initialized")
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
    logger.info(f"🚀 Binding to 0.0.0.0:{port} — no heavy imports at startup")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
