import json
import math
from collections import defaultdict
import yaml
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="PRM_note_level_scores.py Evaluation")
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSON file from evaluation.py')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output CSV file to save metric results')
    parser.add_argument('--config_file', type=str, default='prm-clinic/config/evaluation_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load data from JSON file
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Group items by ('Case_id', 'Sample_no')
    groups = defaultdict(list)
    for item in data:
        key = (item['Case_id'], item['Sample_no'])
        groups[key].append(item)
    
    # Prepare a dictionary to hold Case_id and their corresponding samples
    case_samples = defaultdict(list)
    
    # Compute sum of log probabilities for each group
    for key, items in groups.items():
        Case_id, Sample_no = key
        plus_probs = [float(item['plus_prob']) for item in items]
        # Ensure probabilities are greater than 0 to avoid math domain error
        plus_probs = [p if p > 0 else 1e-10 for p in plus_probs]
        log_plus_probs = [math.log(p) for p in plus_probs]
        sum_log_plus_prob = sum(log_plus_probs)
        case_samples[Case_id].append({
            'Sample_no': Sample_no,
            'sum_log_plus_prob': sum_log_plus_prob
        })
    
    # Determine the sample with the highest probability for each Case_id
    results = []
    for Case_id, samples in case_samples.items():
        # Sort samples by sum of log probabilities in descending order
        samples.sort(key=lambda x: x['sum_log_plus_prob'], reverse=True)
        top_sample = samples[0]
        Sample_no = top_sample['Sample_no']
        is_ref = (Sample_no == 'ref')
        if 'A_Validate' in Case_id:
            group = 'A_Validate'
        elif 'A_Verify' in Case_id:
            group = 'A_Verify'
        elif 'Dialogue_G' in Case_id:
            group = 'Dialogue_G'
        else:
            group = 'Other'
        results.append({
            'Case_id': Case_id,
            'Sample_no': Sample_no,
            'is_ref': is_ref,
            'group': group
        })
    
    # Calculate accuracy for each group
    group_results = defaultdict(list)
    for res in results:
        group = res['group']
        group_results[group].append(res['is_ref'])
    
    # Prepare a DataFrame to save results
    df_list = []
    for group, is_ref_list in group_results.items():
        accuracy = sum(is_ref_list) / len(is_ref_list) * 100
        df_list.append({
            'Group': group,
            'Accuracy': accuracy,
            'Correct': sum(is_ref_list),
            'Total': len(is_ref_list)
        })
    
    df = pd.DataFrame(df_list)
    # Save to CSV
    df.to_csv(args.output_file, index=False)
    print(f"PRM_note_level_scores results saved to {args.output_file}")

if __name__ == '__main__':
    main()
