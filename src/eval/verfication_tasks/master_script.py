import os
import subprocess
import argparse
import pandas as pd
from typing import List, Dict


def main():
    parser = argparse.ArgumentParser(description="Automate evaluation of multiple models")
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of model paths to evaluate')
    parser.add_argument('--config_file', type=str, default='prm_project/config/evaluation/evaluation_config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    args = parser.parse_args()
    
    models = args.models
    config_file = args.config_file
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    #print the models with comments 
    print(f'Running evaluation for {len(models)} models')
    print(f'Models: {models}')

    for model_path in models:
        # Remove trailing slash if present
        model_path = model_path.rstrip('/')
        model_name = os.path.basename(model_path)

        # print the model name
        print(model_name)

        evaluation_output_file = os.path.join(output_dir, f'{model_name}_logits.json')

        # Skip if output file already exists
        if not os.path.exists(evaluation_output_file):
                
        # Run forward_pass.py
            evaluation_cmd = [
                'python', 'prm-clinic/src/eval/verfication_tasks/forward_pass.py',
                '--model_name_or_path', model_path,
                '--output_file', evaluation_output_file,
                '--config_file', config_file
            ]
            print(f'Running evaluation.py for model {model_name}')
            subprocess.run(evaluation_cmd, check=True)

        
        # Now run PRM_note_level_scores.py
        metric1_output_file = os.path.join(output_dir, f'{model_name}_PRM_note_level_scores.csv')
        if not os.path.exists(metric1_output_file):
            metric1_cmd = [
                'python', 'prm-clinic/src/eval/verfication_tasks/PRM_note_level_scores.py',
                '--input_file', evaluation_output_file,
                '--output_file', metric1_output_file
            ]
            print(f'Running PRM_note_level_scores.py for model {model_name}')
            subprocess.run(metric1_cmd, check=True)
        
            
        # Similarly for PRM_label_level_and_ORM_scores.py
        metric2_output_file = os.path.join(output_dir, f'{model_name}_PRM_label_level_and_ORM_scores.csv')
        if not os.path.exists(metric2_output_file):

            metric2_cmd = [
                'python', 'prm-clinic/src/eval/verfication_tasks/PRM_label_level_and_ORM_scores.py',
                '--input_file', evaluation_output_file,
                '--output_file', metric2_output_file,
                '--mapping_file', 'prm-clinic/config/special_token.json'
            ]
            print(f'Running PRM_label_level_and_ORM_scores.py for model {model_name}')
            subprocess.run(metric2_cmd, check=True)

if __name__ == '__main__':
    main()
