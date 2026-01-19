#!/bin/bash
# Kronos entrypoint script - Install dependencies and run freqtrade

echo "Installing Kronos dependencies..."
pip install --quiet huggingface_hub safetensors einops

echo "Starting freqtrade..."
exec freqtrade "$@"
