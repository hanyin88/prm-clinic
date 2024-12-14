This folder contains Python scripts and a shell script that run all verification evaluation tasks on a specified model checkpoint. Before running the script, ensure that the necessary directories and configuration files are properly set up.

## How to Run
```bash
./example_script.sh --dry-run
```

The PRM's accuracy can be found in the `500_PRM_note_level_scores.csv`. The `RM_label_level_and_ORM_scores.csv` contains accuracy for all labels, and ORM's accuracy can be identified where `preceding_token == End_of_note_token` and `true_label == +`.
