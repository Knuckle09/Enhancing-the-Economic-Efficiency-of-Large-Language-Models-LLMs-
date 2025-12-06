#!/bin/bash

# Stop RL-Based Text Optimization Framework Services

set -e

LOG_DIR="logs"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

echo "🛑 Stopping RL-Based Text Optimization Framework"
echo "==============================================="

# Stop API server
if [ -f "$LOG_DIR/api.pid" ]; then
    API_PID=$(cat "$LOG_DIR/api.pid")
    if kill -0 "$API_PID" 2>/dev/null; then
        print_status "Stopping API server (PID: $API_PID)..."
        kill "$API_PID"
        sleep 2
        if kill -0 "$API_PID" 2>/dev/null; then
            print_warning "Force killing API server..."
            kill -9 "$API_PID"
        fi
        rm -f "$LOG_DIR/api.pid"
        print_success "API server stopped"
    else
        print_warning "API server was not running"
        rm -f "$LOG_DIR/api.pid"
    fi
else
    print_warning "No API server PID file found"
fi

# Stop frontend server
if [ -f "$LOG_DIR/frontend.pid" ]; then
    FRONTEND_PID=$(cat "$LOG_DIR/frontend.pid")
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        print_status "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill "$FRONTEND_PID"
        sleep 1
        if kill -0 "$FRONTEND_PID" 2>/dev/null; then
            kill -9 "$FRONTEND_PID"
        fi
        rm -f "$LOG_DIR/frontend.pid"
        print_success "Frontend server stopped"
    else
        print_warning "Frontend server was not running"
        rm -f "$LOG_DIR/frontend.pid"
    fi
else
    print_warning "No frontend server PID file found"
fi

# Stop any remaining Python processes related to the framework
print_status "Checking for remaining framework processes..."
REMAINING_PROCS=$(pgrep -f "api.py\|run.py" 2>/dev/null || true)
if [ -n "$REMAINING_PROCS" ]; then
    print_status "Stopping remaining framework processes..."
    echo "$REMAINING_PROCS" | xargs kill 2>/dev/null || true
    sleep 2
    
    # Force kill if still running
    REMAINING_PROCS=$(pgrep -f "api.py\|run.py" 2>/dev/null || true)
    if [ -n "$REMAINING_PROCS" ]; then
        echo "$REMAINING_PROCS" | xargs kill -9 2>/dev/null || true
    fi
fi

# Optionally stop Ollama (commented out by default)
# print_status "Stopping Ollama service..."
# pkill -f "ollama serve" 2>/dev/null || true

print_success "All framework services stopped"
echo ""
echo "📝 Log files preserved in $LOG_DIR/"
echo "🚀 To restart: ./start_production.sh"