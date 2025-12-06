@echo off
REM Setup script for RL Text Optimization Framework
REM Installs all required dependencies

echo ========================================
echo RL Text Optimization Framework Setup
echo ========================================
echo.

echo Step 1: Installing Python dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 2: Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

echo.
echo Step 3: Downloading spaCy model...
python -m spacy download en_core_web_sm
if %ERRORLEVEL% neq 0 (
    echo WARNING: spaCy model download failed, but will auto-download on first use
)

echo.
echo Step 4: Checking API key configuration...
if not exist ".env" (
    echo.
    echo .env file not found. Creating template...
    echo GEMINI_API_KEY=your_api_key_here > .env
    echo.
    echo IMPORTANT: Edit the .env file and add your Gemini API key!
    echo Get your key from: https://makersuite.google.com/app/apikey
    echo.
) else (
    echo .env file exists ✓
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure your .env file has your GEMINI_API_KEY
echo 2. Run: start_server.bat
echo 3. Open frontend_demo.html in your browser
echo.
pause
