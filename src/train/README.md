## Losses Over Both Dialogue and Clinical Notes Tokens
The `train_model_dialogue_note_loss.py` script calculates the loss over both dialogue and clinical notes tokens. You can select the `loss_mode` from `vanilla`, `special_token`, or `score_token_only`, which correspond to the settings used for the ablation studies in Table 3 of the manuscript.

## Losses Over Clinical Notes Tokens Only
The `train_model_notes_only_loss.py` script calculates the loss over clinical notes tokens only.


# Example Usage of the Training Script
Below is an example command-line usage of the Python training script with various `loss_mode` settings. Make sure you have your `train_config.json` and dataset CSVs in the correct paths before running the commands.

To run the training using the vanilla loss mode:
```bash
deepspeed train_model_dialogue_note_loss.py \
    --file_path "data/train_data.csv" \
    --dataset "my_dataset" \
    --loss_mode "vanilla"
