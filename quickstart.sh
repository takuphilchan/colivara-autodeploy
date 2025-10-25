#!/bin/bash

# Document Q&A Quick Start Script
# This script helps you get started quickly with zero-cost Ollama setup

echo "🚀 Document Q&A - Quick Start"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Ollama is installed
echo "📦 Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
else
    echo -e "${YELLOW}! Ollama not found${NC}"
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Ollama installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install Ollama${NC}"
        exit 1
    fi
fi

# Check if Ollama is running
echo ""
echo "🔍 Checking if Ollama is running..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${YELLOW}! Ollama is not running${NC}"
    echo "Starting Ollama in background..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama started successfully${NC}"
    else
        echo -e "${RED}✗ Failed to start Ollama${NC}"
        echo "Please run 'ollama serve' manually in another terminal"
        exit 1
    fi
fi

# Check for vision models
echo ""
echo "🔍 Checking for vision models..."
MODELS=$(ollama list | grep -E "qwen2.5vl|llama3.2-vision" || true)

if [ -z "$MODELS" ]; then
    echo -e "${YELLOW}! No vision models found${NC}"
    echo "Would you like to download qwen2.5vl:32b? (This may take a while)"
    echo "Options:"
    echo "  1) qwen2.5vl:32b (Recommended - 20GB)"
    echo "  2) llama3.2-vision:11b (Smaller - 7GB)"
    echo "  3) Skip for now"
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            echo "Downloading qwen2.5vl:32b..."
            ollama pull qwen2.5vl:32b
            ;;
        2)
            echo "Downloading llama3.2-vision:11b..."
            ollama pull llama3.2-vision:11b
            ;;
        *)
            echo "Skipping model download. You can download later with:"
            echo "  ollama pull qwen2.5vl:32b"
            ;;
    esac
else
    echo -e "${GREEN}✓ Vision models found:${NC}"
    echo "$MODELS"
fi

# Check Python environment
echo ""
echo "🐍 Checking Python environment..."
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Check database
echo ""
echo "🗄️  Checking database..."
if [ -f "app.db" ]; then
    echo -e "${GREEN}✓ Database exists${NC}"
else
    echo "Initializing database..."
    python -c "from models.database import DatabaseManager; db = DatabaseManager(); db.init_database()"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Database initialized${NC}"
    else
        echo -e "${RED}✗ Failed to initialize database${NC}"
        exit 1
    fi
fi

# Create .env if it doesn't exist
echo ""
echo "⚙️  Checking configuration..."
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Flask Configuration
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
DEBUG=True

# Database
DATABASE_FILE=app.db
SESSION_LIFETIME_HOURS=24

# Ollama (Local - Free)
OLLAMA_URL=http://localhost:11434
MODEL_NAME=qwen2.5vl:32b

# ColiVara (Optional - for document processing)
COLIVARA_BASE_URL=http://localhost:8001/v1
API_KEY=xhFgoUo3UEdhmlIjtq41ds7QJwDM1Yxo

# MinIO (Optional - for document storage)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=miniokey
MINIO_SECRET_KEY=miniosecret
MINIO_BUCKET=colivara
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# All checks passed
echo ""
echo -e "${GREEN}✓ All checks passed!${NC}"
echo ""
echo "=============================="
echo "🎉 Ready to Start!"
echo "=============================="
echo ""
echo "To start the application:"
echo "  1. Run: python app.py"
echo "  2. Open: http://localhost:5000"
echo ""
echo "Configuration:"
echo "  - Provider: Ollama (Local)"
echo "  - Model: qwen2.5vl:32b"
echo "  - Cost: \$0 (completely free)"
echo ""
echo "Quick Tips:"
echo "  - Access Settings from top navigation"
echo "  - Upload documents in Documents page"
echo "  - Start chatting in Query page"
echo "  - Chat history saved automatically"
echo ""
echo "For help, see:"
echo "  - DEPLOYMENT_GUIDE.md"
echo "  - UPDATE_SUMMARY.md"
echo ""

# Ask if user wants to start the app now
read -p "Start the application now? (y/n): " start_now

if [[ $start_now == "y" || $start_now == "Y" ]]; then
    echo ""
    echo "Starting Flask application..."
    echo "Press Ctrl+C to stop"
    echo ""
    python app.py
else
    echo ""
    echo "To start later, run:"
    echo "  source venv/bin/activate"
    echo "  python app.py"
    echo ""
fi
