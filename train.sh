#!/bin/bash

TRAINING_SCRIPT="sft.sh"

LOG_FILE="logs/training_process_monitor.log"
TRAINING_LOG="logs/training_process_output.log"

# Log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

monitor_training() {
    log "========== Training process monitor started =========="
    log "Launching training script"
    bash $TRAINING_SCRIPT 2>&1 | tee -a "$TRAINING_LOG" &
    echo "Training script: $TRAINING_SCRIPT"
    echo "Current time: $(date +'%Y-%m-%d %H:%M:%S')"
    TRAINING_PID=$!
    log "Training process started, PID: $TRAINING_PID"

    while true; do
        if kill -0 $TRAINING_PID 2>/dev/null; then
            sleep 60
        else
            log "Training process (PID: $TRAINING_PID) terminated, restarting..."
            # Record GPU status
            log "Recording GPU status:"
            gpustat >> "$LOG_FILE"
            sleep 10
            bash $TRAINING_SCRIPT 2>&1 | tee -a "$TRAINING_LOG" &
            TRAINING_PID=$!
            log "Training process restarted, new PID: $TRAINING_PID"
        fi
    done
}

monitor_training 