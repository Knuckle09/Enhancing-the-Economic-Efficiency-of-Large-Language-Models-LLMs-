#!/bin/bash

# Startup script for RL Text Optimization Framework API
# This script sets up and starts the API server

echo "🚀 Starting RL Text Optimization Framework API Server"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ~/test-main/llmbased/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if Ollama is running
echo "🔍 Checking Ollama availability..."
if command -v ollama &> /dev/null; then
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is running"
    else
        echo "⚠️  Ollama is installed but not running. Starting Ollama..."
        ollama serve &
        sleep 5
    fi
else
    echo "⚠️  Ollama not found. Please install Ollama from https://ollama.ai"
    echo "   Required models: codellama:7b, qwen2-math:7b, tinyllama:latest"
fi

# Check required files
echo "🔍 Checking required files..."
required_files=("run.py" "api.py" "rl_optimizer.py" "llm_efficiency_test.py" "prompt_diversity_test.py" "text_optimizer_ppo.zip")

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done

echo "✅ All required files found"

# Start the API server
echo "🌐 Starting API server on http://localhost:5000"
echo "📡 API Endpoints:"
echo "   POST /api/process     - Process prompts"
echo "   GET  /api/health      - Health check"
echo "   GET  /api/status      - Framework status"
echo "   GET  /api/strategies  - Available strategies"
echo ""
echo "🌐 Frontend demo: Open frontend_demo.html in your browser"
echo "📝 Press Ctrl+C to stop the server"
echo "=================================================="

# Start the server
python3 api.py