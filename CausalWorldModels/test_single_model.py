#!/usr/bin/env python3
"""
Test Single Model - Quick validation that one architecture works
Tests one model completely to ensure the pipeline is working
"""

import subprocess
import time
import os
from pathlib import Path

def test_single_model():
    """Test baseline_32D model to validate the training pipeline"""
    print("🧪 TESTING SINGLE MODEL - baseline_32D")
    print("=" * 50)
    
    # Clear any existing log
    log_file = Path("./data/logs/phase1/test_baseline.log")
    if log_file.exists():
        log_file.unlink()
    
    # Run baseline_32D for just 2 epochs to test the pipeline
    cmd = [
        'python', 'experiments/train_causal_vae_experiment.py',
        '--architecture', 'baseline_32D',
        '--force_causal',
        '--epochs', '2',  # Just 2 epochs for testing
        '--batch_size', '32',
        '--learning_rate', '0.0001',
        '--data_dir', './data/causal_episodes',
        '--output_dir', './data/models/causal/test_baseline',
        '--experiment_name', 'test_baseline'
    ]
    
    print("🚀 Starting test training...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run with real-time output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
    start_time = time.time()
    output_lines = []
    
    try:
        for line in process.stdout:
            print(line.rstrip())
            output_lines.append(line.rstrip())
            
            # Check for success indicators
            if "✅ CAUSAL-CONDITIONED training completed" in line:
                print("\n🎉 TEST SUCCESSFUL - Model trained successfully!")
                break
            elif "❌ Training failed:" in line:
                print(f"\n❌ TEST FAILED - {line.strip()}")
                break
        
        process.wait()
        elapsed = time.time() - start_time
        
        print(f"\n📊 Test completed in {elapsed:.1f} seconds")
        print(f"Exit code: {process.returncode}")
        
        # Analyze results
        if process.returncode == 0:
            print("✅ Pipeline is working correctly!")
            return True
        else:
            print("❌ Pipeline has issues")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        process.terminate()
        return False

if __name__ == "__main__":
    success = test_single_model()
    if success:
        print("\n💡 Ready to run all 8 experiments!")
    else:
        print("\n🔧 Fix pipeline issues before running full experiments")
