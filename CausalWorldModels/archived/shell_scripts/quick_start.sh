#!/bin/bash

# Causal World Models - Quick Start Script
# Activates conda environment and runs the pipeline

echo "🚀 Causal World Models - Quick Start"
echo "====================================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "📋 Activating conda environment..."
    
    # Activate base environment (or specify your preferred env)
    eval "$(conda shell.bash hook)"
    conda activate base
    
    echo "✅ Conda environment activated"
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
    
    # Install missing dependencies in conda environment
    echo "📦 Installing missing dependencies..."
    pip install gym opencv-python psutil
    
    # Run the setup test
    echo "🧪 Running setup test..."
    python test_setup.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Setup successful! Ready to run experiments:"
        echo "   python run_simple_pipeline.py --dry_run"
        echo "   python experiments/phase1_orchestrator.py --dry_run"
        echo ""
        echo "Or run a quick validation:"
        echo "   python run_simple_pipeline.py"
    else
        echo "⚠️  Some issues detected - check output above"
    fi
    
else
    echo "⚠️  Conda not found. Using system Python."
    echo "   You may need to install dependencies manually:"
    echo "   pip3 install gym opencv-python psutil numpy matplotlib torch"
    echo ""
    echo "Or try running directly:"
    echo "   python3 test_setup.py"
fi