#!/bin/bash

# Define base paths for convenience
MODEL_DIR="prm-clinic/output/Llama-8B_Instruct_special_token_loss"
CHECKPOINT="checkpoint-500"
CHECKPOINT_PATH="$MODEL_DIR/$CHECKPOINT"

RESULTS_DIR="prm-clinic/test/verification_tasks"
CONFIG_FILE="prm-clinic/config/verificaiton_tasks_evaluation_config.yaml"
LOG_DIR="prm-clinic/test/verification_tasks"

# Enable dry run mode if --dry-run is provided as an argument
DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
  DRY_RUN=true
fi

# Extract model name from MODEL_DIR (e.g., "Llama-8B_Instruct_zero_out_dialogue")
MODEL_NAME=$(basename "$MODEL_DIR")

# Define output path for results
MODEL_RESULTS_DIR="$RESULTS_DIR/$MODEL_NAME"

# Create log directory if it doesn't exist
if [ "$DRY_RUN" = false ]; then
  mkdir -p "$LOG_DIR"
else
  echo "mkdir -p \"$LOG_DIR/$MODEL_NAME\""
fi

# Run the Python script with the specified checkpoint path
CMD="python prm-clinic/src/eval/verfication_tasks/master_script.py \
    --models \"$CHECKPOINT_PATH\" \
    --output_dir \"$MODEL_RESULTS_DIR\" \
    --config_file \"$CONFIG_FILE\" > \"$LOG_DIR/$MODEL_NAME.log\" 2>&1"

if [ "$DRY_RUN" = false ]; then
  echo "$CMD"
  eval "$CMD"
else
  echo "$CMD"
fi
