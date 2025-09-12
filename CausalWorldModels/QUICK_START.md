# Quick Start Guide - Causal World Models

## Prerequisites

### Required Dependencies
```bash
# Install PyTorch (for training)
pip install torch torchvision numpy matplotlib

# Install additional dependencies  
pip install gym opencv-python scikit-learn

# For evolution strategies (Phase 2B)
pip install cma
```

### Hardware Requirements
- **Minimum**: 16GB RAM, 4 CPU cores
- **Recommended**: 64GB+ RAM, 8+ CPU cores  
- **Optimal**: M2 Ultra 128GB RAM (as designed)

---

## Phase 1: Architecture Validation (Ready to Run)

### 1. Generate Training Data
```bash
cd CausalWorldModels

# Generate 1000 episodes with structured causal coverage
python3 01_generate_causal_data.py --total_episodes 1000 --parallel_envs 16
```

### 2. Run Architecture Experiments
```bash
# Full experiment suite (8 architectures in staggered groups)
python3 experiments/phase1_orchestrator.py

# Or dry run to see execution plan
python3 experiments/phase1_orchestrator.py --dry_run
```

### 3. Monitor Progress
```bash
# Check experiment logs
tail -f data/models/phase1/logs/*.log

# Monitor system resources
htop
```

Expected timeline: **2-3 days** for all 8 experiments

---

## Phase 2A: Causal Validation (After Phase 1)

### 1. Select Best Architectures
Phase 1 will automatically rank architectures. Use top 2-3 for Phase 2A.

### 2. Implement Causal RNN (TODO)
```bash
# This script needs to be created based on Phase 1 results
python3 02_train_causal_rnn.py --architecture categorical_512D
```

### 3. Run Causal Experiments
```bash
# Test causal factor importance
python3 experiments/phase2a_orchestrator.py
```

---

## Quick Tests

### Test Environment
```bash
python3 test_pipeline.py
```

Expected output:
```
✅ Environment renders 64x64x3 observations with causal effects
✅ Causal states properly encoded to 45-dimensional vectors  
✅ Exploration strategies show different behavioral patterns
✅ Reward function includes causal modifiers
✅ Ready for Phase 1 architecture experiments!
```

### Test Data Generation
```bash
python3 01_generate_causal_data.py --total_episodes 10 --parallel_envs 2
```

Should generate 10 episodes in `data/causal_episodes/`

### Test Orchestration
```bash
python3 experiments/phase1_orchestrator.py --dry_run
```

Should show memory allocation plan for 8 experiments.

---

## Configuration

### Memory Limits
```bash
# Adjust memory limits based on your hardware
python3 experiments/phase1_orchestrator.py --max_memory_gb 64  # For 64GB systems
python3 experiments/phase1_orchestrator.py --max_memory_gb 32  # For 32GB systems
```

### Episode Count
```bash
# For quick testing
python3 01_generate_causal_data.py --total_episodes 100

# For full experiments  
python3 01_generate_causal_data.py --total_episodes 5000
```

### Training Parameters
Edit `experiments/phase1_orchestrator.py` to adjust:
- `training_epochs`: Reduce for faster experiments
- `batch_size`: Reduce if memory issues occur
- `learning_rate`: Architecture-specific tuning

---

## Troubleshooting

### Memory Issues
```bash
# Reduce batch sizes in phase1_orchestrator.py
# Or run fewer experiments in parallel
python3 experiments/phase1_orchestrator.py --max_memory_gb 32
```

### Training Instability  
```bash
# Check logs for specific architecture issues
cat data/models/phase1/results/*_result.json

# Reduce learning rates or increase regularization
```

### Slow Training
```bash
# Reduce episode count for testing
python3 01_generate_causal_data.py --total_episodes 500

# Reduce training epochs
# Edit phase1_orchestrator.py: training_epochs=25
```

---

## Expected Results

### Phase 1 Output
```
📊 PHASE 1 RESULTS SUMMARY
Total Duration: 24.3 hours
Experiments: 7/8 successful  
Success Rate: 87.5%
Memory Used: 103.0GB / 128GB

🏆 Top Performing Architectures:
  1. gaussian_256D (2.8h)
  2. hierarchical_512D (3.1h)  
  3. categorical_512D (3.4h)
```

### Data Structure
```
CausalWorldModels/
├── data/
│   ├── causal_episodes/           # Training data
│   │   ├── episode_000000.npz     # Individual episodes
│   │   └── generation_summary.npz # Data statistics
│   └── models/phase1/             # Experiment results
│       ├── results/               # Individual results
│       ├── checkpoints/           # Model weights  
│       └── phase1_summary.json    # Overall summary
```

---

## Next Steps After Phase 1

1. **Analyze Results**: Review `phase1_summary.json` 
2. **Select Architectures**: Choose top 2-3 for Phase 2A
3. **Implement Causal RNN**: Extend best VAE with causal conditioning
4. **Plan Phase 2A**: 6 causal experiments with selected architectures

---

## Support

### Debug Mode
```bash
# Run with detailed logging
python3 experiments/phase1_orchestrator.py --verbose

# Test individual components
python3 -c "from test_pipeline import test_pipeline; test_pipeline()"
```

### Configuration Files
- `experiments/phase1_orchestrator.py`: Experiment settings
- `01_generate_causal_data.py`: Data generation parameters
- `test_pipeline.py`: System verification

### Common Issues
- **ImportError**: Install missing dependencies with pip
- **Memory Error**: Reduce `max_memory_gb` parameter
- **CUDA Error**: Code defaults to CPU, GPU optional

---

*Ready to start? Run `python3 test_pipeline.py` to verify your setup!*