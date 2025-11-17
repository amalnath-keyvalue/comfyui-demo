#!/bin/bash

set -e

MODEL_PATH="ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
MODEL_URL="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"

if [ ! -d "ComfyUI" ]; then
    echo "==> Cloning ComfyUI repository..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
    
    echo "==> Setting up ComfyUI virtual environment..."
    cd ComfyUI
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    deactivate
    cd ..
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "==> Downloading Stable Diffusion 1.5 model (4GB)..."
    mkdir -p "ComfyUI/models/checkpoints"
    wget -O "$MODEL_PATH" "$MODEL_URL"
fi

if [ ! -f ".env" ]; then
    echo "==> Creating .env file..."
    cp env.example .env
    echo ""
    echo "⚠️  Please edit .env and add your GROQ_API_KEY"
    echo "   Get one free at: https://console.groq.com"
    echo ""
    read -p "Press enter after you've added your API key..."
fi

echo "==> Ensuring agent dependencies are installed..."
poetry install --no-interaction

echo ""
echo "==> Starting ComfyUI Agent..."
echo ""
poetry run python -m src.agent.main
