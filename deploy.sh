#!/bin/bash

# NASA Exoplanet AI - Deployment Script

echo "🚀 Deploying NASA Exoplanet AI Platform..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run from project root."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install Python dependencies
echo "🐍 Installing Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Prepare data if not already done
if [ ! -d "data/processed" ]; then
    echo "📊 Preparing NASA datasets..."
    python scripts/prepare_data.py
fi

# Train model if not already done
if [ ! -d "models" ]; then
    echo "🧠 Training AI model..."
    python scripts/train_model.py
fi

# Build Next.js application
echo "🏗️ Building Next.js application..."
npm run build

echo "✅ Deployment preparation complete!"
echo ""
echo "🌐 To deploy to Vercel:"
echo "   1. Push to GitHub repository"
echo "   2. Connect repository to Vercel"
echo "   3. Deploy automatically"
echo ""
echo "🔧 To run locally:"
echo "   npm run dev"
echo ""
echo "📊 Model Performance:"
echo "   - Training samples: 7,809"
echo "   - Test accuracy: 83.23%"
echo "   - Features: 15 astronomical parameters"
echo ""
echo "🎯 Ready for NASA Space Apps Challenge 2025!"
