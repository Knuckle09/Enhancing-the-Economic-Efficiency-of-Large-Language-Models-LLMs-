# RL-Based Text Optimization Framework - Production

This is a production-ready deployment of the RL-based text optimization framework that automatically reduces prompt tokens while preserving semantic meaning and routes to appropriate LLMs.

> **🆕 NEW: Gemini AI Support!** Now includes Google's Gemini models. [Quick Setup →](GEMINI_QUICKSTART.md) | [Full Guide →](GEMINI_SETUP.md)

## Framework Goal
- **Input**: Any text prompt
- **Process**: 
  1. RL-based token reduction (30%+ reduction while maintaining 80%+ semantic similarity)
  2. Automatic prompt classification and LLM routing
  3. Response generation from selected specialized LLM
- **Output**: Optimized response from the most suitable LLM

## Files Included
- `run.py` - Main framework entry point (CLI)
- `api.py` - Flask REST API server
- `frontend_demo.html` - Web interface demo
- `start_server.sh` - Quick startup script
- `rl_optimizer.py` - RL environment and training components
- `llm_efficiency_test.py` - Token optimization strategies and similarity calculations
- `prompt_diversity_test.py` - LLM routing and prompt classification
- `setup_gemini.py` - Gemini API setup helper
- `test_gemini.py` - Gemini integration tests
- `GEMINI_SETUP.md` - Detailed Gemini setup guide
- `text_optimizer_ppo.zip` - Trained RL model
- `rl_training_data_20250926_232119.json` - Latest training data
- `requirements.txt` - Python dependencies

## Supported LLMs

### Local Models (via Ollama)
- **TinyLlama** - Fast general-purpose model
- **Phi-3** - Microsoft's efficient model
- **CodeLlama** - Meta's coding specialist
- **Qwen2.5-Coder** - Alibaba's coding model
- **Qwen2-Math** - Math specialist
- **DeepSeek-Math** - Advanced math reasoning
- **LLaVA** - Vision-language model
- **Moondream** - Image generation

### Cloud Models
- **Gemini Pro** - Google's powerful AI (coding, math, reasoning)
- **Gemini 1.5 Flash** - Fast general-purpose model

**New!** Gemini integration allows you to use Google's powerful AI models alongside local Ollama models. See [GEMINI_SETUP.md](GEMINI_SETUP.md) for setup instructions.

## Quick Start

### Option 1: API Server (Recommended for Frontend Integration)
```bash
# Quick start with automatic setup
./start_server.sh

# Or manual setup:
pip install -r requirements.txt
python api.py
```

### Option 2: CLI Usage
```bash
# Interactive mode
python run.py

# CLI mode with prompt
python run.py "Your prompt here"
```

## API Endpoints

### POST `/api/process`
Process a prompt with RL optimization

**Request:**
```json
{
  "prompt": "Write a Python function to calculate fibonacci numbers",
  "include_response": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "original_prompt": "Write a Python function to calculate fibonacci numbers",
    "optimized_prompt": "Write Python function calculate fibonacci numbers",
    "strategy_used": "balanced",
    "token_reduction_percent": 25.0,
    "similarity": 0.991,
    "target_achieved": true,
    "selected_llm": "codellama",
    "category": "coding",
    "response": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
  },
  "processing_time": 2.3,
  "timestamp": "2025-09-30T..."
}
```

### GET `/api/health`
Health check endpoint

### GET `/api/status`
Framework initialization status

### GET `/api/strategies`
Available optimization strategies and LLM mappings

## Frontend Integration

1. **Start the API server**: `./start_server.sh`
2. **Open the demo**: Open `frontend_demo.html` in your browser
3. **Or integrate with your app**: Use the REST API endpoints

### Example JavaScript Integration:
```javascript
async function optimizePrompt(prompt) {
    const response = await fetch('http://localhost:5000/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            prompt: prompt,
            include_response: true
        })
    });
    
    const result = await response.json();
    
    if (result.success) {
        console.log('Strategy:', result.data.strategy_used);
        console.log('Reduction:', result.data.token_reduction_percent + '%');
        console.log('LLM:', result.data.selected_llm);
        console.log('Response:', result.data.response);
    }
}
```

## Setup Requirements
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Ensure Ollama is running** with required models:
   - `qwen2-math:7b` (for math prompts)
   - `codellama:7b` (for coding prompts)  
   - `tinyllama:latest` (for generic prompts)
3. **GPU with CUDA** required for optimal performance

## Architecture
The framework uses a PPO-based RL agent trained on token reduction tasks, combined with NLP-based prompt classification for intelligent LLM routing. The API server provides a REST interface for easy integration with web frontends and applications.

### Deployment Options
- **Development**: Direct CLI usage with `python run.py`
- **API Server**: Flask REST API for web integration
- **Frontend**: HTML/JavaScript demo included
- **Production**: Use with reverse proxy (nginx) and process manager (pm2/supervisor)