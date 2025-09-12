#!/usr/bin/env python3
"""
Training Curves Plotter
Generates loss curves and training visualizations from log files
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def parse_training_log(log_file):
    """Extract training metrics from log file"""
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract epoch data: "Epoch X/Y 🔗: Train Loss=A, Val Loss=B, Time=Cs"
    epoch_pattern = r'Epoch\s+(\d+)/\d+\s+🔗:\s+Train Loss=([0-9.]+),\s+Val Loss=([0-9.]+),\s+Time=([0-9.]+)s'
    matches = re.findall(epoch_pattern, content)
    
    if not matches:
        return None
    
    epochs = []
    train_losses = []
    val_losses = []
    times = []
    
    for match in matches:
        epochs.append(int(match[0]))
        train_losses.append(float(match[1]))
        val_losses.append(float(match[2]))
        times.append(float(match[3]))
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'times': times,
        'name': log_file.stem
    }

def plot_single_experiment(data, save_path=None):
    """Plot training curves for a single experiment"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(data['epochs'], data['train_loss'], 'b-', label='Training Loss', alpha=0.8)
    ax1.plot(data['epochs'], data['val_loss'], 'r-', label='Validation Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{data["name"]} - Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training time per epoch
    ax2.plot(data['epochs'], data['times'], 'g-', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title(f'{data["name"]} - Training Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    else:
        plt.show()

def plot_all_experiments(log_dir, save_dir=None):
    """Plot comparison of all experiments"""
    log_dir = Path(log_dir)
    experiments = [
        'baseline_32D', 'gaussian_256D', 'beta_vae_4.0', 'no_conv_normalization',
        'hierarchical_512D', 'categorical_512D', 'vq_vae_256D', 'deeper_encoder'
    ]
    
    # Parse all experiments
    all_data = {}
    for exp in experiments:
        log_file = log_dir / f"{exp}.log"
        data = parse_training_log(log_file)
        if data:
            all_data[exp] = data
    
    if not all_data:
        print("No training data found!")
        return
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    
    # Training loss comparison
    for i, (name, data) in enumerate(all_data.items()):
        ax1.plot(data['epochs'], data['train_loss'], color=colors[i], 
                label=name, alpha=0.8, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Validation loss comparison
    for i, (name, data) in enumerate(all_data.items()):
        ax2.plot(data['epochs'], data['val_loss'], color=colors[i], 
                label=name, alpha=0.8, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Final validation loss ranking
    final_losses = []
    names = []
    for name, data in all_data.items():
        if data['val_loss']:
            final_losses.append(data['val_loss'][-1])
            names.append(name)
    
    if final_losses:
        sorted_data = sorted(zip(names, final_losses), key=lambda x: x[1])
        names_sorted, losses_sorted = zip(*sorted_data)
        
        bars = ax3.bar(range(len(names_sorted)), losses_sorted, 
                      color=colors[:len(names_sorted)], alpha=0.7)
        ax3.set_xlabel('Architecture')
        ax3.set_ylabel('Final Validation Loss')
        ax3.set_title('Final Performance Ranking')
        ax3.set_xticks(range(len(names_sorted)))
        ax3.set_xticklabels(names_sorted, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, loss in zip(bars, losses_sorted):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{loss:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Training efficiency (loss reduction per minute)
    for i, (name, data) in enumerate(all_data.items()):
        if len(data['epochs']) > 1:
            total_time = sum(data['times']) / 60  # minutes
            loss_reduction = data['train_loss'][0] - data['train_loss'][-1]
            efficiency = loss_reduction / total_time if total_time > 0 else 0
            ax4.bar(i, efficiency, color=colors[i], alpha=0.7, label=name)
    
    ax4.set_xlabel('Architecture')
    ax4.set_ylabel('Loss Reduction per Minute')
    ax4.set_title('Training Efficiency')
    ax4.set_xticks(range(len(all_data)))
    ax4.set_xticklabels(all_data.keys(), rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "training_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot: {save_path}")
    else:
        plt.show()
    
    # Print summary
    print("\n📊 TRAINING SUMMARY:")
    print("=" * 50)
    if final_losses:
        for i, (name, loss) in enumerate(sorted_data):
            print(f"{i+1:2d}. {name:<20} Final Val Loss: {loss:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--log_dir', default='./data/logs/phase1',
                       help='Directory containing log files')
    parser.add_argument('--save_dir', default='./data/plots',
                       help='Directory to save plots')
    parser.add_argument('--experiment', 
                       help='Plot single experiment (e.g., gaussian_256D)')
    
    args = parser.parse_args()
    
    # Create save directory
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    if args.experiment:
        # Plot single experiment
        log_file = Path(args.log_dir) / f"{args.experiment}.log"
        data = parse_training_log(log_file)
        if data:
            save_path = Path(args.save_dir) / f"{args.experiment}_curves.png" if args.save_dir else None
            plot_single_experiment(data, save_path)
        else:
            print(f"No data found for {args.experiment}")
    else:
        # Plot all experiments
        plot_all_experiments(args.log_dir, args.save_dir)

if __name__ == "__main__":
    main()
