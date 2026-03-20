"""
Minimal diagnostic api.py — strips all custom imports to get Flask running
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import traceback
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------
# Try importing custom modules — log exactly what fails
# -------------------------------------------------------
IMPORT_ERRORS = {}

try:
    from run import process_prompt
    logger.info("✅ run.py imported OK")
except Exception:
    IMPORT_ERRORS["run"] = traceback.format_exc()
    logger.error(f"❌ run.py failed:\n{IMPORT_ERRORS['run']}")

try:
    from rl_optimizer import RLOptimizer, get_latest_training_data_file
    logger.info("✅ rl_optimizer.py imported OK")
except Exception:
    IMPORT_ERRORS["rl_optimizer"] = traceback.format_exc()
    logger.error(f"❌ rl_optimizer.py failed:\n{IMPORT_ERRORS['rl_optimizer']}")

try:
    from prompt_diversity_test import PromptDiversityTester
    logger.info("✅ prompt_diversity_test.py imported OK")
except Exception:
    IMPORT_ERRORS["prompt_diversity_test"] = traceback.format_exc()
    logger.error(f"❌ prompt_diversity_test.py failed:\n{IMPORT_ERRORS['prompt_diversity_test']}")


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "running",
        "import_errors": IMPORT_ERRORS,
        "all_imports_ok": len(IMPORT_ERRORS) == 0,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
