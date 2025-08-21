#!/bin/bash

# Start ChatDBT with Local LLM frontend

echo "Starting ChatDBT with Local LLM frontend..."

# Navigate to frontend directory
cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the frontend server
echo "Starting frontend server..."
npm run dev
