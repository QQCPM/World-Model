#!/usr/bin/env python3
"""
Simple Training Dashboard - Quick status check
Shows current training status in a compact format
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

def get_experiment_status():
    """Get quick status of all experiments"""
    log_dir = Path("./data/logs/phase1")
    experiments = [
        'baseline_32D', 'gaussian_256D', 'beta_vae_4.0', 'no_conv_normalization',
        'hierarchical_512D', 'categorical_512D', 'vq_vae_256D', 'deeper_encoder'
    ]
    
    status = {}
    
    for exp in experiments:
        log_file = log_dir / f"{exp}.log"
        
        if not log_file.exists():
            status[exp] = "❌ No log"
            continue
            
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            if "✅ CAUSAL-CONDITIONED training completed" in content:
                status[exp] = "✅ Complete"
            elif "❌ Training failed:" in content:
                status[exp] = "❌ Failed"
            elif "🔥 Starting CAUSAL training" in content:
                # Find latest epoch
                epoch_matches = re.findall(r'Epoch\s+(\d+)/(\d+)', content)
                if epoch_matches:
                    current, total = epoch_matches[-1]
                    progress = int(current) / int(total) * 100
                    status[exp] = f"🔄 {current}/{total} ({progress:.0f}%)"
                else:
                    status[exp] = "🔄 Training"
            else:
                status[exp] = "⏳ Starting"
                
        except Exception as e:
            status[exp] = f"❌ Error: {str(e)[:20]}"
    
    return status

def show_dashboard():
    """Display training dashboard"""
    print("🧠 CAUSAL VAE TRAINING DASHBOARD")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    status = get_experiment_status()
    
    # Group 1
    print("📊 GROUP 1 (Foundation):")
    group1 = ['baseline_32D', 'gaussian_256D', 'beta_vae_4.0', 'no_conv_normalization']
    for exp in group1:
        print(f"  {exp:<20} {status.get(exp, '❓ Unknown')}")
    
    print()
    
    # Group 2  
    print("🚀 GROUP 2 (Advanced):")
    group2 = ['hierarchical_512D', 'categorical_512D', 'vq_vae_256D', 'deeper_encoder']
    for exp in group2:
        print(f"  {exp:<20} {status.get(exp, '❓ Unknown')}")
    
    print()
    
    # Summary
    completed = sum(1 for s in status.values() if "Complete" in s)
    training = sum(1 for s in status.values() if "Training" in s or "%" in s)
    failed = sum(1 for s in status.values() if "Failed" in s)
    
    print(f"📈 SUMMARY: {completed}/8 complete, {training} training, {failed} failed")
    
    if completed == 8:
        print("🎉 ALL EXPERIMENTS COMPLETE! Ready for Phase 2A")

if __name__ == "__main__":
    show_dashboard()
