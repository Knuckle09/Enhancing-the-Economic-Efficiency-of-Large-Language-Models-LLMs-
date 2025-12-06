#!/bin/bash
# RL Text Optimization API Server Startup Script (Linux/Mac)
# This script automatically loads the API key and starts the server

echo "========================================"
echo "RL Text Optimization API Server"
echo "========================================"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with your GEMINI_API_KEY"
    echo "Example: GEMINI_API_KEY=your_api_key_here"
    exit 1
fi

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Check if API key was loaded
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not found in .env file!"
    exit 1
fi

echo "✅ API Key loaded successfully"
echo "🚀 Starting server..."
echo ""
echo "Server will be available at: http://localhost:5000"
echo "Open frontend at: http://localhost:5000 or open frontend_demo.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python api.py
