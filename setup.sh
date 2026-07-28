#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Dependencies installed successfully!"
echo "Please copy .env.example to .env and configure your LLM API Key."
