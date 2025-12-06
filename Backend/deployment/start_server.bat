@echo off
REM RL Text Optimization API Server Startup Script
REM This script automatically loads the API key and starts the server

echo ========================================
echo RL Text Optimization API Server
echo ========================================
echo.

REM Check if .env file exists
if not exist ".env" (
    echo ERROR: .env file not found!
    echo Please create a .env file with your GEMINI_API_KEY
    echo Example: GEMINI_API_KEY=your_api_key_here
    pause
    exit /b 1
)

REM Load environment variables from .env file
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if "%%a"=="GEMINI_API_KEY" set GEMINI_API_KEY=%%b
)

REM Check if API key was loaded
if "%GEMINI_API_KEY%"=="" (
    echo ERROR: GEMINI_API_KEY not found in .env file!
    pause
    exit /b 1
)

echo API Key loaded successfully
echo Starting server...
echo.
echo Server will be available at: http://localhost:5000
echo Open frontend at: http://localhost:5000 or open frontend_demo.html
echo.
echo Press Ctrl+C to stop the server
echo.

python api.py

pause
