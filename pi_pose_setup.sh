#!/bin/bash

# Setup script for Preterm Infant Pose Estimation Project
# This script initializes the project structure and dependencies

set -e  # Exit on error

echo "=========================================="
echo "Preterm Infant Pose Estimation Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.7"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_success "Python $python_version detected"
else
    print_error "Python 3.7+ required. Found: $python_version"
    exit 1
fi

# Create project directory structure
print_info "Creating project directories..."
mkdir -p configs
mkdir -p models
mkdir -p data/annotations
mkdir -p data/images/train
mkdir -p data/images/val
mkdir -p data/images/test
mkdir -p utils
mkdir -p tools
mkdir -p outputs
mkdir -p logs
mkdir -p visualizations
mkdir -p checkpoints
mkdir -p results

print_success "Project directories created"

# Create __init__.py files
print_info "Creating Python package files..."
touch models/__init__.py
touch data/__init__.py
touch utils/__init__.py
touch tools/__init__.py

print_success "Package files created"

# Check if conda is available
if command -v conda &> /dev/null; then
    print_info "Conda detected. Creating virtual environment..."
    
    # Create conda environment
    conda create -n pi_pose python=3.8 -y
    print_success "Conda environment 'pi_pose' created"
    
    print_info "To activate: conda activate pi_pose"
else
    print_info "Conda not found. Using venv..."
    
    # Create virtual environment with venv
    python3 -m venv venv
    source venv/bin/activate
    print_success "Virtual environment created"
    
    print_info "To activate: source venv/bin/activate"
fi

# Install dependencies
print_info "Installing dependencies..."
print_info "This may take a few minutes..."

# Check if we're in conda or venv
if [[ "$CONDA_DEFAULT_ENV" == "pi_pose" ]] || [[ -n "$VIRTUAL_ENV" ]]; then
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    print_success "Dependencies installed"
else
    print_error "Please activate the virtual environment first"
    print_info "Run: conda activate pi_pose  OR  source venv/bin/activate"
    print_info "Then run: pip install -r requirements.txt"
fi

# Create sample configuration files
print_info "Creating configuration files..."

# This would be done by the config.py script
python3 -c "
from config import get_default_config, get_preemie_config, save_config
import os

os.makedirs('configs', exist_ok=True)
save_config(get_default_config(), 'configs/default.yaml')
save_config(get_preemie_config(), 'configs/preemie_optimized.yaml')
print('Configuration files created')
" 2>/dev/null || print_info "Config files will be created when you run config.py"

print_success "Setup complete!"

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Activate environment:"
echo "   conda activate pi_pose  OR  source venv/bin/activate"
echo ""
echo "2. Prepare your dataset in COCO format:"
echo "   python tools/convert_to_coco.py --input_dir <annotations> --image_dir <images> --output_file data/annotations/train.json"
echo ""
echo "3. Analyze your dataset:"
echo "   python tools/analyze_dataset.py --ann_file data/annotations/train.json"
echo ""
echo "4. Start training:"
echo "   python train.py --data_dir ./data --output_dir ./outputs"
echo ""
echo "5. Run inference:"
echo "   python inference.py --checkpoint outputs/model_best.pth --image test.jpg"
echo ""
echo "For more information, see README.md"
echo "=========================================="
