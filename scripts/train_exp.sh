#!/bin/bash
# Script to monitor GPU 1 and run experiments when it becomes available

# Change to the project root directory
cd /home/fsoto/Documents/LCsSSL

# Log file path
LOG_DIR="logs/gpu_monitor"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/gpu_monitor_$(date +"%Y%m%d_%H%M%S").log"

# Function to check if GPU 1 is in use
function is_gpu_free() {
    # Check if GPU 1 is in use with nvidia-smi
    # Returns 0 (true in bash) if GPU is free, 1 (false in bash) if it's in use
    local gpu_id=$1
    
    # Get GPU utilization percentage
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
    
    # Get GPU memory usage percentage
    local mem_util=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits -i $gpu_id)
    
    echo "$(date): GPU $gpu_id - Utilization: $gpu_util%, Memory: $mem_util%" >> $LOG_FILE
    
    # Consider GPU free if both utilization and memory usage are below thresholds
    if [ "$gpu_util" -lt 10 ] && [ "$mem_util" -lt 10 ]; then
        return 0  # GPU is free
    else
        return 1  # GPU is in use
    fi
}

function run_experiments() {
    echo "$(date): Starting experiments on GPU 1" >> $LOG_FILE
    
    # Export environment variable to use GPU 1
    export CUDA_VISIBLE_DEVICES=1
    
    echo "$(date): Running atat_exp_lc experiment" >> $LOG_FILE
    python src/train.py experiment=atat_exp_lc logger=wandb 
    
    echo "$(date): First experiment completed, starting second experiment" >> $LOG_FILE
    python src/train.py experiment=atat_exp_tabular logger=wandb 
    
    echo "$(date): All experiments completed" >> $LOG_FILE
}

echo "Starting GPU monitor script at $(date)" > $LOG_FILE
echo "Will check GPU 1 availability every 5 minutes" >> $LOG_FILE

# Main loop
while true; do
    if is_gpu_free 1; then
        echo "$(date): GPU 1 is free, starting experiments" >> $LOG_FILE
        run_experiments
        echo "$(date): Experiments completed, exiting" >> $LOG_FILE
        break
    else
        echo "$(date): GPU 1 is currently in use, will check again in 5 minutes" >> $LOG_FILE
        sleep 300  # Wait for 5 minutes
    fi
done

echo "$(date): Script finished" >> $LOG_FILE