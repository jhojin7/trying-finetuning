#!/bin/bash
# Quick setup script for detectron2 on Mac Studio
# Run this to install all dependencies using uv

set -e

echo "🚀 Setting up detectron2 environment for Mac Studio..."

# Check Python version
python3 --version

# Install PyTorch with MPS support (for Apple Silicon GPU acceleration)
echo "📦 Installing PyTorch with MPS support..."
uv pip install torch torchvision torchaudio

# Install detectron2 dependencies first
echo "📦 Installing detectron2 dependencies..."
uv pip install opencv-python matplotlib pycocotools

# Install detectron2 (building from source for Mac compatibility)
echo "📦 Installing detectron2..."
uv pip install 'git+https://github.com/facebookresearch/detectron2.git'

# If the above fails, try this alternative:
# uv pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html

# Install additional helpful libraries
echo "📦 Installing additional libraries..."
uv pip install jupyter notebook pillow

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: jupyter notebook"
echo "2. Open: notebooks/detectron2_finetuning.ipynb"
echo "3. Run the cells to start training"
