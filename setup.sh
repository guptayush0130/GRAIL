#!/bin/bash

# AGORA v2 Setup Script
# This script helps set up the AGORA v2 environment

echo "=========================================="
echo "AGORA v2 - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Check if config.py exists
if [ ! -f "config.py" ]; then
    echo "⚠️  WARNING: config.py not found!"
    echo ""
    echo "Please create config.py from config_template.py:"
    echo "  1. cp config_template.py config.py"
    echo "  2. Edit config.py and add your API keys"
    echo ""
else
    echo "✓ config.py found"
    echo ""
fi

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Make sure config.py has your API keys"
echo "  2. Run a test: python agora.py"
echo "  3. Or try examples: python example_usage.py"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""

