#!/bin/bash

# Quick start script for Docker deployment

set -e

echo "🐳 Starting Adaptive Learning System with Docker..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "📝 Creating .env from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please edit it and add your OPENAI_API_KEY"
        echo "   Then run this script again."
        exit 1
    else
        echo "❌ .env.example not found. Please create .env manually with OPENAI_API_KEY"
        exit 1
    fi
fi

# Check if OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=.*[^=]$" .env 2>/dev/null; then
    echo "⚠️  OPENAI_API_KEY not set in .env file"
    echo "   Please edit .env and add your OpenAI API key"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

echo "✅ Pre-flight checks passed"
echo ""
echo "🔨 Building and starting containers..."
echo ""

# Build and start
docker-compose up --build -d

echo ""
echo "✅ Services started!"
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🌐 Access the application:"
echo "   Frontend: http://localhost:8501"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart: docker-compose restart"
echo ""

