#!/bin/bash

# Deep Deer Application Runner Script

echo "🦌 Starting Deep Deer Application..."

# Check if virtual environment is activated (uv)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected. Running with uv..."
    UV_PREFIX="uv run"
else
    echo "✅ Virtual environment detected"
    UV_PREFIX=""
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ .env file created from .env.example"
        echo "⚠️  Please edit .env file to add your API keys and configuration"
    else
        echo "❌ .env.example not found. Please create .env file manually"
        exit 1
    fi
fi

# Check if Ollama is running (only if using Ollama provider)
if grep -q "LLM_PROVIDER=ollama" .env 2>/dev/null || grep -q "EMBEDDING_PROVIDER=ollama" .env 2>/dev/null; then
    echo "🔍 Checking Ollama service..."
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is running"
    else
        echo "⚠️  Ollama is not running. Please start Ollama with: ollama serve"
        echo "   Or change LLM_PROVIDER and EMBEDDING_PROVIDER in .env to use HuggingFace instead"
    fi
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p vector_stores
mkdir -p logs

# Load environment variables from .env
export $(grep -v '^#' .env | xargs)

# Run the application
echo "🚀 Starting FastAPI server..."
echo "📍 Server will be available at: http://localhost:${PORT:-8000}"
echo "📖 API docs: http://localhost:${PORT:-8000}/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "="*60

$UV_PREFIX python main.py