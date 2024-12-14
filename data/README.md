# Dataset

## PRM-Clinic
This folder contains the PRM-Clinic dataset used for PRM training. After unzipping, you will find two JSON files containing metadata for the negative samples (which include errors) and the gold-reference samples. For the negative samples, the metadata detail the introduced errors, incompleteness, and paraphrases, as well as the original content.

The CSV file includes concatenated dialogues and clinical notes, formatted and ready for PRM training. In the clinical notes, each step has been concatenated with its respective special step token and step score label token.

## test_set_verification_tasks
This folder contains the test set for tasks A-Verification, A-Validation, and Dialogue-G, as described in Table 2. The notes from A-Verify are sourced from [`data/RLHF_data/r5_RLHF_note_reviewed_combined.csv`](https://github.com/hanyin88/llama-clinic.git) in the LLaMA-Clinic repository, after filtering out the ACIBENCH training set, which is already included in PRM-Clinic.  

After unzipping, you will find two JSON files containing metadata for the negative samples (which include errors) and the gold-reference samples, similar to the PRM-Clinic dataset. Unlike PRM-Clinic, this folder includes two CSV files:  
1. **`verification_tasks_true_label.csv`** - Contains concatenated cases with true score labels.  
2. **`verification_tasks_placehold.csv`** - Contains concatenated cases where true score labels have been replaced with placeholder labels.

## test_set_A_prefer
This folder contains dataset for task A-Prefer.  The notes from A-Prefer are sourced from [`data/RLHF_data/r4_RLHF_note_reviewed_combined.csv`](https://github.com/hanyin88/llama-clinic.git) in the LLaMA-Clinic repository, after filtering out the ACIBENCH training set, which is already included in PRM-Clinic. The `Training_note` column contains concatenated dialogues and clinical notes, formatted and prepared for a forward pass. Placeholder labels were used at all step token label positions.

## physician_reader_study
This folder contains the physician preference data for the final blinded physician reader study. For the Dual High vs. Dual Low and High A-Verify vs. Low A-Verify groups, three reviewers each reviewed all cases. For the High A-Verify vs. Low A-Verify group, multiple reviewers reviewed a subset of cases.