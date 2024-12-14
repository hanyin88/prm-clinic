import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Master script to run evaluation pipeline for multiple models.')
    parser.add_argument('--models', nargs='+', required=True, help='List of model checkpoint paths.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs.')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--input_csv_file', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--gold_labels_file', type=str, required=True, help='Path to the CSV file containing gold labels.')
    args = parser.parse_args()

    for model_path in args.models:
        model_name = os.path.basename(model_path.rstrip('/'))
        model_output_dir = os.path.join(args.output_dir)
        os.makedirs(model_output_dir, exist_ok=True)

        # Paths for outputs
        forward_pass_output = os.path.join(model_output_dir, f'{model_name}_logits.json')
        
        if not os.path.exists(forward_pass_output):

            # Run forward_pass.py
            forward_pass_cmd = [
                'python', 'prm-clinic/src/eval/A_Prefer/forward_pass.py',
                '--model_path', model_path,
                '--config_file', args.config_file,
                '--input_csv_file', args.input_csv_file,
                '--output_file', forward_pass_output
            ]
            print(f"Running forward pass for model: {model_name}")
            subprocess.run(forward_pass_cmd, check=True)

        # Run accuracy.py
        accuracy_cmd = [
            'python', 'prm-clinic/src/eval/A_Prefer/compute_accuracy.py',
            '--data_file', forward_pass_output,
            '--gold_labels_file', args.gold_labels_file,
            '--results_dir', model_output_dir,
            '--model_name', model_name
        ]
        print(f"Calculating accuracy for model: {model_name}")
        subprocess.run(accuracy_cmd, check=True)

if __name__ == '__main__':
    main()
