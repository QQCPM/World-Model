#!/bin/bash
# Quick Training Status Check
# Usage: ./quick_check.sh

echo "🔍 QUICK TRAINING STATUS CHECK"
echo "=============================="
echo "Time: $(date '+%H:%M:%S')"
echo

# Check for running training processes
echo "🔄 RUNNING PROCESSES:"
pgrep -f "train_causal_vae_experiment" | while read pid; do
    if [ ! -z "$pid" ]; then
        cmd=$(ps -p $pid -o args= 2>/dev/null)
        if [[ $cmd == *"--architecture"* ]]; then
            arch=$(echo $cmd | grep -o -- '--architecture [^ ]*' | cut -d' ' -f2)
            runtime=$(ps -p $pid -o etime= | tr -d ' ')
            memory=$(ps -p $pid -o rss= | awk '{print int($1/1024)"MB"}')
            echo "  ✅ $arch (PID:$pid) Runtime:$runtime Memory:$memory"
        fi
    fi
done

# Check if no processes found
if [ -z "$(pgrep -f train_causal_vae_experiment)" ]; then
    echo "  ❌ No training processes found"
fi

echo
echo "📊 LOG FILE STATUS:"

# Check each experiment log
for exp in baseline_32D gaussian_256D beta_vae_4.0 no_conv_normalization hierarchical_512D categorical_512D vq_vae_256D deeper_encoder; do
    logfile="./data/logs/phase1/${exp}.log"
    if [ -f "$logfile" ]; then
        # Check last modification time
        last_mod=$(stat -f "%Sm" -t "%H:%M:%S" "$logfile" 2>/dev/null || stat -c "%y" "$logfile" 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1)
        
        # Check file size
        size=$(wc -l < "$logfile" 2>/dev/null || echo "0")
        
        # Check last line for status
        if tail -1 "$logfile" 2>/dev/null | grep -q "✅.*training completed"; then
            status="✅ COMPLETED"
        elif tail -1 "$logfile" 2>/dev/null | grep -q "❌.*Training failed"; then
            status="❌ FAILED"
        elif tail -5 "$logfile" 2>/dev/null | grep -q "Epoch.*Train Loss"; then
            epoch=$(tail -10 "$logfile" 2>/dev/null | grep "Epoch" | tail -1 | grep -o "Epoch [0-9]*/[0-9]*" | head -1)
            status="🔄 $epoch"
        elif [ "$size" -gt 10 ]; then
            status="🔄 Training"
        else
            status="⏳ Starting"
        fi
        
        printf "  %-20s %s (${size} lines, updated: %s)\n" "$exp" "$status" "$last_mod"
    else
        printf "  %-20s ❓ No log file\n" "$exp"
    fi
done

echo
echo "💾 SYSTEM RESOURCES:"
if command -v free >/dev/null 2>&1; then
    free -h | head -2
elif command -v vm_stat >/dev/null 2>&1; then
    # macOS
    vm_stat | head -5
fi

echo
echo "🎯 QUICK ACTIONS:"
echo "  python process_detective.py     # Full investigation"
echo "  python watch_training.py        # Live monitoring"
echo "  tail -f data/logs/phase1/[exp].log  # Watch specific log"
