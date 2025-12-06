
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
            logger.warning("⚠️ No trained model found")
            framework_status["error"] = "No trained model found"
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
            return jsonify({
                "success": False,
                "error": "Request must be JSON",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        include_response = data.get('include_response', True)
        model_preference = data.get('model_preference', 'auto')
        selected_model = data.get('selected_model', None)
        
        if not prompt:
            return jsonify({
                "success": False,
                "error": "Prompt is required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        if not framework_status["initialized"]:
            return jsonify({
                "success": False,
                "error": "Framework not initialized",
                "details": framework_status.get("error"),
                "timestamp": datetime.now().isoformat()
            }), 500
        
        logger.info(f"📝 Processing prompt: {prompt[:100]}...")
        logger.info(f"🎯 Model preference: {model_preference}")
        
        category = prompt_tester.classify_prompt(prompt)
        
        action, strategy = rl_optimizer.predict_optimal_strategy(prompt, category)
        optimized_prompt, metrics = rl_optimizer.env.apply_optimization_strategy(prompt, action)
        
        if model_preference == 'manual' and selected_model:
            selected_llm = selected_model
            logger.info(f"👤 User selected model: {selected_llm}")
        else:
            prefer_gemini = framework_status.get('initialized', False) and \
                          hasattr(rl_optimizer.env.llm_tester, 'gemini_available') and \
                          rl_optimizer.env.llm_tester.gemini_available
            selected_llm = prompt_tester.route_prompt_to_llm(optimized_prompt, prefer_gemini=prefer_gemini)
            logger.info(f"🤖 Framework suggested model: {selected_llm}")
        
        gemini_pricing = {
            'gemini-pro': 3.125,
            'gemini-flash': 0.1875,
            'gemini': 0.1875
        }
        
        original_tokens = metrics.get('original_tokens', 0)
        optimized_tokens = metrics.get('optimized_tokens', 0)
        tokens_saved = metrics.get('tokens_saved', 0)
        
        cost_per_million = gemini_pricing.get(selected_llm, 0.1875)
        original_cost = (original_tokens / 1_000_000) * cost_per_million
        optimized_cost = (optimized_tokens / 1_000_000) * cost_per_million
        cost_saved = original_cost - optimized_cost
        
        response_data = {
            "original_prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "strategy_used": strategy,
            "token_reduction_percent": round(metrics['reduction_percent'], 1),
            "similarity": round(metrics['similarity'], 3),
            "target_achieved": metrics.get('target_achieved', False),
            "selected_llm": selected_llm,
            "model_preference": model_preference,
            "category": category,
            "metrics": {
                "original_tokens": original_tokens,
                "optimized_tokens": optimized_tokens,
                "tokens_saved": tokens_saved
            },
            "cost": {
                "original_cost_usd": round(original_cost, 6),
                "optimized_cost_usd": round(optimized_cost, 6),
                "cost_saved_usd": round(cost_saved, 6),
                "model_pricing": f"${cost_per_million:.4f}/1M tokens"
            }
        }
        
        if include_response:
            try:
                if selected_llm in ['gemini-pro', 'gemini-flash', 'gemini']:
                    import google.generativeai as genai
                    
                    model_map = {
                        'gemini-pro': 'gemini-2.5-pro',
                        'gemini-flash': 'gemini-2.5-flash',
                        'gemini': 'gemini-2.5-flash'
                    }
                    actual_model = model_map.get(selected_llm, 'gemini-2.5-flash')
                    
                    model = genai.GenerativeModel(actual_model)
                    response = model.generate_content(optimized_prompt)
                    llm_response = response.text
                    response_data["response"] = llm_response
                    logger.info(f"✅ Generated response using Gemini API: {actual_model}")
                else:
                    import subprocess
                    llm_response = subprocess.check_output(
                        ["ollama", "run", selected_llm, optimized_prompt], 
                        text=True, timeout=120
                    ).strip()
                    response_data["response"] = llm_response
                    logger.info(f"✅ Generated response using Ollama: {selected_llm}")
            except Exception as e:
                logger.warning(f"⚠️ LLM generation failed: {e}")
                response_data["response"] = f"Error generating response: {e}"
                response_data["response_error"] = str(e)
        
        processing_time = round(time.time() - start_time, 2)
        
        logger.info(f"✅ Processed in {processing_time}s - {strategy} strategy, {metrics['reduction_percent']:.1f}% reduction")
        
        return jsonify({
            "success": True,
            "data": response_data,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        logger.error(f"❌ Processing failed: {e}")
        
        return jsonify({
            "success": False,
            "error": str(e),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    gemini_available = False
    if rl_optimizer and hasattr(rl_optimizer.env.llm_tester, 'gemini_available'):
        gemini_available = rl_optimizer.env.llm_tester.gemini_available
    
    llms = {
        "coding": "codellama:7b",
        "math": "qwen2-math:7b",
        "generic": "tinyllama:latest"
    }
    
    if gemini_available:
        llms["gemini_coding"] = "gemini-pro"
        llms["gemini_math"] = "gemini-pro"
        llms["gemini_generic"] = "gemini-pro"
        llms["gemini_flash"] = "gemini-1.5-flash"
    
    return jsonify({
        "strategies": {
            "conservative": {
                "description": "15% token reduction, 90% similarity",
                "target_reduction": 15,
                "min_similarity": 0.90
            },
            "balanced": {
                "description": "30% token reduction, 85% similarity",
                "target_reduction": 30,
                "min_similarity": 0.85
            },
            "aggressive": {
                "description": "35% token reduction, 85% similarity",
                "target_reduction": 35,
                "min_similarity": 0.85
            }
        },
        "categories": ["coding", "math", "generic"],
        "llms": llms,
        "gemini_available": gemini_available
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("🚀 Starting RL-Based Text Optimization API Server...")
    
    if initialize_framework():
        print("✅ Framework initialized successfully")
        print("🌐 Starting Flask server...")
        print("📡 API Endpoints:")
        print("   POST /api/process - Process prompts")
        print("   GET  /api/health  - Health check")
        print("   GET  /api/status  - Framework status")
        print("   GET  /api/strategies - Available strategies")
        print("-" * 50)
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    else:
        print("❌ Framework initialization failed. Check logs for details.")
        sys.exit(1)