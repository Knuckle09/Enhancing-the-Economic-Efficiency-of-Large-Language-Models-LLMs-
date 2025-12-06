#!/bin/bash
# Setup script for RL Text Optimization Framework (Linux/Mac)
# Installs all required dependencies

echo "========================================"
echo "RL Text Optimization Framework Setup"
echo "========================================"
echo ""

echo "Step 1: Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Step 2: Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

echo ""
echo "Step 3: Downloading spaCy model..."
python -m spacy download en_core_web_sm
if [ $? -ne 0 ]; then
    echo "WARNING: spaCy model download failed, but will auto-download on first use"
fi

echo ""
echo "Step 4: Checking API key configuration..."
if [ ! -f ".env" ]; then
    echo ""
    echo ".env file not found. Creating template..."
    echo "GEMINI_API_KEY=your_api_key_here" > .env
    echo ""
    echo "⚠️  IMPORTANT: Edit the .env file and add your Gemini API key!"
    echo "Get your key from: https://makersuite.google.com/app/apikey"
    echo ""
else
    echo "✅ .env file exists"
fi

echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Make sure your .env file has your GEMINI_API_KEY"
echo "2. Run: ./start_server.sh (or bash start_server.sh)"
echo "3. Open frontend_demo.html in your browser"
echo ""

# Make start_server.sh executable
chmod +x start_server.sh
echo "✅ Made start_server.sh executable"
echo ""
