#!/bin/bash


# Define base paths for convenience
MODEL_DIR="prm-clinic/output/Llama-8B_Instruct_special_token_loss"
CHECKPOINT="checkpoint-500"
CHECKPOINT_PATH="$MODEL_DIR/$CHECKPOINT"

RESULTS_DIR="prm-clinic/test/A_Prefer"
CONFIG_FILE="prm-clinic/config/A_Prefer_evaluation_config.yaml"
LOG_DIR="prm-clinic/test/A_Prefer"
INPUT_CSV_FILE="prm-clinic/data/test_set_A_prefer/A_Prefer.csv"
GOLD_LABELS_FILE="prm-clinic/data/test_set_A_prefer/preference_label.csv"

# Enable dry run mode if DRY_RUN is set
DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
  DRY_RUN=true
fi


# Extract model name (e.g., "model1")
MODEL_NAME=$(basename "$MODEL_DIR")

# Define output path for results
MODEL_RESULTS_DIR="$RESULTS_DIR/$MODEL_NAME"

# Create log directory if it doesn't exist
if [ "$DRY_RUN" = false ]; then
  mkdir -p "$LOG_DIR"
else
  echo "mkdir -p \"$LOG_DIR\""
fi


# Run the Python script with the specified checkpoint path
CMD="python prm-clinic/src/eval/A_Prefer/master_script.py\
      --models \"$CHECKPOINT_PATH\" \
      --output_dir "$MODEL_RESULTS_DIR" \
      --config_file "$CONFIG_FILE" \
      --input_csv_file "$INPUT_CSV_FILE" \
      --gold_labels_file "$GOLD_LABELS_FILE" > "$LOG_DIR/$MODEL_NAME.log" 2>&1"

if [ "$DRY_RUN" = false ]; then
  echo "$CMD"
  eval "$CMD"
else
  echo "$CMD"
fi


# # Run the master script with appropriate arguments
# if [ "$DRY_RUN" = false ]; then
#   echo "Running master_script.py for model: $MODEL_NAME"
#   python prm-clinic/src/eval/A_Prefer/master_script.py\
#       --models "${CHECKPOINT_PATHS[@]}" \
#       --output_dir "$MODEL_RESULTS_DIR" \
#       --config_file "$CONFIG_FILE" \
#       --input_csv_file "$INPUT_CSV_FILE" \
#       --gold_labels_file "$GOLD_LABELS_FILE" > "$LOG_DIR/$MODEL_NAME.log" 2>&1
# else
#   echo "python prm-clinic/src/eval/A_Prefer/master_script.py \
#       --models ${CHECKPOINT_PATHS[@]} \
#       --output_dir \"$MODEL_RESULTS_DIR\" \
#       --config_file \"$CONFIG_FILE\" \
#       --input_csv_file \"$INPUT_CSV_FILE\" \
#       --gold_labels_file \"$GOLD_LABELS_FILE\" > \"$LOG_DIR/$MODEL_NAME.log\" 2>&1"
# fi


