#!/bin/bash

echo "🔧 Updating pip and tools..."
pip install --upgrade pip setuptools wheel --progress-bar on

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt --progress-bar on

echo "✅ Installation complete!"