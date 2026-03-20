"""
Flask API Server for RL-Based Text Optimization Framework

This API provides endpoints to:
1. Process prompts with RL optimization
2. Get framework status and health checks
3. Return structured responses for frontend integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import time
import logging
from datetime import datetime

from run import process_prompt
from rl_optimizer import RLOptimizer, get_latest_training_data_file
from prompt_diversity_test import PromptDiversityTester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

rl_optimizer = None
prompt_tester = None
framework_status = {"initialized": False, "error": None}

def initialize_framework():
    """Initialize framework components once at startup"""
    global rl_optimizer, prompt_tester, framework_status
    
    try:
        logger.info("🚀 Initializing RL-based text optimization framework...")
        
        training_data_file = get_latest_training_data_file("./results")
        rl_optimizer = RLOptimizer(training_data_file)
        prompt_tester = PromptDiversityTester()
        
        model_path = "./models/text_optimizer_ppo.zip"
        if os.path.exists(model_path):
            from stable_baselines3 import PPO
            rl_optimizer.model = PPO.load(model_path)
            logger.info(f"✅ Loaded trained RL model: {model_path}")
        else:
            logger.warning("⚠️ No trained model found at: " + model_path)
            framework_status = {
                "initialized": False,
                "error": "No trained model found. Please upload a trained model.",
                "timestamp": datetime.now().isoformat()
            }
            return False
            
        framework_status = {
            "initialized": True, 
            "error": None,
            "model_path": model_path,
            "training_data": training_data_file,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ Framework initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Framework initialization failed: {e}")
        framework_status = {
            "initialized": False, 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        return False


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Nimbus AI - RL Text Optimization API",
        "status": "running",
        "framework_initialized": framework_status["initialized"],
        "endpoints": ["/api/health", "/api/status", "/api/process"],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "framework_initialized": framework_status["initialized"],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(framework_status)


@app.route('/api/process', methods=['POST'])
def process_prompt_api():
    start_time = time.time()
    
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Request must be JSON"}), 400
        
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        include_response = data.get('include_response', True)

        if not prompt:
            return jsonify({"success": False, "error": "Prompt is required"}), 400

        if not framework_status["initialized"]:
            return jsonify({
                "success": False,
                "error": "Framework not initialized. " + (framework_status.get("error") or "Unknown error.")
            }), 503
        
        category = prompt_tester.classify_prompt(prompt)
        action, strategy = rl_optimizer.predict_optimal_strategy(prompt, category)
        optimized_prompt, metrics = rl_optimizer.env.apply_optimization_strategy(prompt, action)

        response_data = {
            "original_prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "strategy_used": strategy,
            "token_reduction_percent": round(metrics['reduction_percent'], 1),
            "similarity": round(metrics['similarity'], 3),
            "category": category
        }
        
        processing_time = round(time.time() - start_time, 2)
        
        return jsonify({
            "success": True,
            "data": response_data,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ---------------------------------------------------------
# 🔥 RENDER FIX: Server ALWAYS starts, regardless of model
# ---------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))

    logger.info("🚀 Starting Nimbus AI API Server...")
    logger.info(f"🌐 Binding to host=0.0.0.0, port={port}")

    # Initialize framework in background — don't block server startup
    init_success = initialize_framework()
    if init_success:
        logger.info("✅ Framework ready.")
    else:
        logger.warning("⚠️ Framework not initialized — server still starting.")
        logger.warning("   /api/process will return 503 until model is available.")

    # Always start the server
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
