#!/bin/bash

# RL-Based Text Optimization Framework - Production Startup Script
# This script sets up and starts the complete framework

set -e  # Exit on any error

echo "🚀 Starting RL-Based Text Optimization Framework"
echo "================================================"

# Configuration
PYTHON_ENV="venv"
API_PORT=5000
FRONTEND_PORT=8080
LOG_DIR="logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create logs directory
mkdir -p "$LOG_DIR"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is available
port_available() {
    ! nc -z localhost "$1" 2>/dev/null
}

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    print_status "Waiting for $service_name to start..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            print_success "$service_name is running on $host:$port"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to start after $((max_attempts * 2)) seconds"
    return 1
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python $python_version found"
    
    # Check Ollama
    if ! command_exists ollama; then
        print_error "Ollama is not installed. Please install from https://ollama.ai/download"
        exit 1
    fi
    print_success "Ollama found"
    
    # Check CUDA (optional)
    if command_exists nvidia-smi; then
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $gpu_info"
    else
        print_warning "No GPU detected. Framework will run on CPU"
    fi
    
    # Check ports
    if ! port_available "$API_PORT"; then
        print_error "Port $API_PORT is already in use"
        exit 1
    fi
    
    if ! port_available "$FRONTEND_PORT"; then
        print_warning "Port $FRONTEND_PORT is already in use"
    fi
    
    print_success "System requirements check completed"
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$PYTHON_ENV" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv "$PYTHON_ENV"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source "$PYTHON_ENV/bin/activate"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip > "$LOG_DIR/pip_install.log" 2>&1
    
    # Install requirements
    print_status "Installing Python dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt >> "$LOG_DIR/pip_install.log" 2>&1
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Setup Ollama models
setup_ollama() {
    print_status "Setting up Ollama models..."
    
    # Start Ollama service if not running
    if ! pgrep -f "ollama serve" > /dev/null; then
        print_status "Starting Ollama service..."
        ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
        
        # Wait for Ollama to start
        if ! wait_for_service "localhost" "11434" "Ollama"; then
            print_error "Failed to start Ollama service"
            exit 1
        fi
    else
        print_success "Ollama service is already running"
    fi
    
    # Required models
    models=("qwen2-math:7b" "codellama:7b" "tinyllama:latest")
    
    for model in "${models[@]}"; do
        print_status "Checking model: $model"
        if ollama list | grep -q "$model"; then
            print_success "Model $model is already installed"
        else
            print_status "Installing model: $model (this may take several minutes)..."
            ollama pull "$model" >> "$LOG_DIR/ollama_pull.log" 2>&1
            if [ $? -eq 0 ]; then
                print_success "Model $model installed successfully"
            else
                print_error "Failed to install model $model"
                exit 1
            fi
        fi
    done
    
    print_success "All required models are available"
}

# Test framework
test_framework() {
    print_status "Testing framework components..."
    
    # Activate virtual environment
    source "$PYTHON_ENV/bin/activate"
    
    # Test core framework
    print_status "Testing core framework..."
    if python3 -c "from run import process_prompt; print('Framework import successful')" 2>/dev/null; then
        print_success "Core framework is working"
    else
        print_error "Core framework test failed"
        exit 1
    fi
    
    # Test API components
    print_status "Testing API components..."
    if python3 -c "from api import initialize_framework; print('API import successful')" 2>/dev/null; then
        print_success "API components are working"
    else
        print_error "API component test failed"
        exit 1
    fi
}

# Start API server
start_api() {
    print_status "Starting API server..."
    
    # Activate virtual environment
    source "$PYTHON_ENV/bin/activate"
    
    # Start API in background
    nohup python3 api.py > "$LOG_DIR/api.log" 2>&1 &
    API_PID=$!
    
    # Save PID for later cleanup
    echo $API_PID > "$LOG_DIR/api.pid"
    
    # Wait for API to start
    if wait_for_service "localhost" "$API_PORT" "API Server"; then
        print_success "API server started successfully (PID: $API_PID)"
        print_status "API available at: http://localhost:$API_PORT"
        
        # Test health endpoint
        sleep 2
        if curl -s "http://localhost:$API_PORT/api/health" | grep -q "healthy"; then
            print_success "API health check passed"
        else
            print_warning "API health check failed"
        fi
    else
        print_error "Failed to start API server"
        exit 1
    fi
}

# Start frontend server (optional)
start_frontend() {
    if [ -f "frontend_demo.html" ] && port_available "$FRONTEND_PORT"; then
        print_status "Starting frontend demo server..."
        
        nohup python3 -m http.server "$FRONTEND_PORT" > "$LOG_DIR/frontend.log" 2>&1 &
        FRONTEND_PID=$!
        
        # Save PID
        echo $FRONTEND_PID > "$LOG_DIR/frontend.pid"
        
        sleep 2
        if wait_for_service "localhost" "$FRONTEND_PORT" "Frontend Server"; then
            print_success "Frontend server started (PID: $FRONTEND_PID)"
            print_status "Frontend available at: http://localhost:$FRONTEND_PORT/frontend_demo.html"
        else
            print_warning "Frontend server failed to start"
        fi
    fi
}

# Display startup summary
show_summary() {
    echo ""
    echo "🎉 Framework startup completed successfully!"
    echo "================================================"
    echo ""
    echo "📡 API Endpoints:"
    echo "   Health Check: http://localhost:$API_PORT/api/health"
    echo "   Process Text: http://localhost:$API_PORT/api/process"
    echo "   Strategies:   http://localhost:$API_PORT/api/strategies"
    echo ""
    
    if [ -f "$LOG_DIR/frontend.pid" ]; then
        echo "🌐 Frontend Demo:"
        echo "   URL: http://localhost:$FRONTEND_PORT/frontend_demo.html"
        echo ""
    fi
    
    echo "📊 Test Commands:"
    echo "   curl http://localhost:$API_PORT/api/health"
    echo "   python3 test_api_complete.py"
    echo ""
    echo "📝 Logs Location:"
    echo "   API Logs:     $LOG_DIR/api.log"
    echo "   Ollama Logs:  $LOG_DIR/ollama.log"
    echo "   Install Logs: $LOG_DIR/pip_install.log"
    echo ""
    echo "🛑 To stop services:"
    echo "   ./stop_server.sh"
    echo "   OR: kill \$(cat $LOG_DIR/*.pid)"
    echo ""
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    
    if [ -f "$LOG_DIR/api.pid" ]; then
        API_PID=$(cat "$LOG_DIR/api.pid")
        if kill -0 "$API_PID" 2>/dev/null; then
            kill "$API_PID"
            rm -f "$LOG_DIR/api.pid"
        fi
    fi
    
    if [ -f "$LOG_DIR/frontend.pid" ]; then
        FRONTEND_PID=$(cat "$LOG_DIR/frontend.pid")
        if kill -0 "$FRONTEND_PID" 2>/dev/null; then
            kill "$FRONTEND_PID"
            rm -f "$LOG_DIR/frontend.pid"
        fi
    fi
    
    exit 1
}

# Set trap for cleanup on exit
trap cleanup INT TERM

# Main execution
main() {
    # Parse command line arguments
    SKIP_TESTS=false
    SKIP_FRONTEND=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-frontend)
                SKIP_FRONTEND=true
                shift
                ;;
            --api-port)
                API_PORT="$2"
                shift 2
                ;;
            --frontend-port)
                FRONTEND_PORT="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-tests         Skip framework testing"
                echo "  --skip-frontend      Skip frontend server startup"
                echo "  --api-port PORT      Set API port (default: 5000)"
                echo "  --frontend-port PORT Set frontend port (default: 8080)"
                echo "  -h, --help          Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute startup sequence
    check_requirements
    setup_python_env
    setup_ollama
    
    if [ "$SKIP_TESTS" = false ]; then
        test_framework
    fi
    
    start_api
    
    if [ "$SKIP_FRONTEND" = false ]; then
        start_frontend
    fi
    
    show_summary
    
    # Keep script running
    print_status "Framework is running. Press Ctrl+C to stop."
    while true; do
        sleep 10
        
        # Check if API is still running
        if [ -f "$LOG_DIR/api.pid" ]; then
            API_PID=$(cat "$LOG_DIR/api.pid")
            if ! kill -0 "$API_PID" 2>/dev/null; then
                print_error "API server has stopped unexpectedly"
                exit 1
            fi
        fi
    done
}

# Run main function
main "$@"