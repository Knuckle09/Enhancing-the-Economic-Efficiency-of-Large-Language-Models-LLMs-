"""
Flask API - Nimbus AI Backend
Minimal version - no lazy loading, instant 503 if model missing
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

logger.info("✅ Flask app created, binding to port...")


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Nimbus AI API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "initialized": False,
        "error": "Model not loaded",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/process', methods=['POST', 'OPTIONS'])
def process_prompt_api():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    return jsonify({
        "success": False,
        "error": "Framework not initialized. The RL model (text_optimizer_ppo.zip) has not been loaded yet.",
        "timestamp": datetime.now().isoformat()
    }), 503


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Binding to 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
