#!/bin/bash

# Start ChatDBT with LM Studio backend

echo "Starting ChatDBT with LM Studio backend..."

# Navigate to backend directory
cd "$(dirname "$0")/backend"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp env_example.txt .env
    echo "Please edit .env file with your configuration before running the server."
    exit 1
fi

# Start the backend server
echo "Starting backend server..."
python main.py
