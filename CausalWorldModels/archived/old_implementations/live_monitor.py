#!/usr/bin/env python3
"""
Simple Live Training Monitor
Shows real-time progress of all VAE training processes in a clean dashboard.
"""
import os
import json
import time
import subprocess
from datetime import datetime

def get_running_processes():
    """Get all running training processes"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = []
        for line in result.stdout.split('\n'):
            if 'python3 05_train_vae.py' in line and '--architecture' in line:
                parts = line.split()
                pid = parts[1]
                # Extract architecture name
                arch_start = line.find('--architecture') + len('--architecture') + 1
                arch_end = line.find(' ', arch_start)
                if arch_end == -1:
                    arch_end = len(line)
                arch = line[arch_start:arch_end]
                processes.append({'pid': pid, 'architecture': arch})
        return processes
    except:
        return []

def get_latest_progress(architecture):
    """Get latest training progress for an architecture"""
    log_file = f"logs/{architecture}_training_log.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)

            epochs_completed = len(data.get('train_losses', []))
            if epochs_completed > 0:
                latest_train_loss = data['train_losses'][-1]
                latest_val_loss = data['val_losses'][-1]
                return {
                    'epoch': epochs_completed - 1,
                    'train_loss': latest_train_loss,
                    'val_loss': latest_val_loss,
                    'status': 'training'
                }
        except:
            pass

    # Check if model is completed
    model_file = f"models/{architecture}_best.pth"
    if os.path.exists(model_file):
        return {'status': 'completed', 'epoch': 50}

    return {'status': 'starting', 'epoch': 0}

def format_loss(loss):
    """Format loss value for display"""
    if loss > 1000:
        return f"{loss:.0f}"
    elif loss > 10:
        return f"{loss:.1f}"
    else:
        return f"{loss:.3f}"

def get_progress_bar(epoch, total_epochs=50, width=20):
    """Create a simple progress bar"""
    progress = epoch / total_epochs
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {epoch}/{total_epochs}"

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def main():
    """Main monitoring loop"""
    architectures = [
        'baseline_32D', 'vq_vae_256D', 'gaussian_256D',
        'categorical_512D', 'beta_vae_4.0', 'hierarchical_512D',
        'no_conv_normalization', 'deeper_encoder'
    ]

    print("🚀 Starting Live Training Monitor...")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            clear_screen()

            # Header
            print("=" * 80)
            print(f"🧠 VAE TRAINING MONITOR - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 80)

            # Get running processes
            running_processes = {p['architecture']: p['pid'] for p in get_running_processes()}

            # Status for each architecture
            completed = 0
            running = 0

            for arch in architectures:
                progress = get_latest_progress(arch)
                status = progress['status']

                # Status emoji
                if status == 'completed':
                    emoji = "✅"
                    completed += 1
                elif status == 'training':
                    emoji = "🏃"
                    running += 1
                elif arch in running_processes:
                    emoji = "🔄"
                    running += 1
                else:
                    emoji = "⏳"

                # Format line
                line = f"{emoji} {arch:<20}"

                if status == 'completed':
                    line += f" {get_progress_bar(50)} DONE"
                elif status == 'training':
                    epoch = progress['epoch']
                    train_loss = format_loss(progress['train_loss'])
                    val_loss = format_loss(progress['val_loss'])
                    line += f" {get_progress_bar(epoch)} Loss: {train_loss}/{val_loss}"
                elif arch in running_processes:
                    line += f" {get_progress_bar(0)} STARTING... (PID: {running_processes[arch]})"
                else:
                    line += f" {get_progress_bar(0)} PENDING"

                print(line)

            # Summary
            print("=" * 80)
            print(f"📊 SUMMARY: {completed}/8 Complete | {running} Running | {8-completed-running} Pending")

            if completed == 8:
                print("🎉 ALL TRAINING COMPLETE!")
                break

            print("📝 Updates every 30 seconds... (Ctrl+C to exit)")
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user")

if __name__ == "__main__":
    main()