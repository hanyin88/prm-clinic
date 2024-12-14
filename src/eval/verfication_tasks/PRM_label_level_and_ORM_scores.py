import json
import yaml
from collections import defaultdict
import argparse
import pandas as pd

def load_key_mapping(mapping_file_path):
    with open(mapping_file_path, 'r') as file:
        return json.load(file)
        
def load_and_process_json(data, replacement_dict):
    # Load the nested JSON file

    def process_dict(d):
        # Iterate through each key-value pair in the dictionary
        for key, value in d.items():
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                process_dict(value)
            elif isinstance(value, list):
                # Process each item in a list if it is a dictionary
                for i in range(len(value)):
                    if isinstance(value[i], dict):
                        process_dict(value[i])
            else:
                # Replace value if it matches any key in the replacement dictionary
                if value in replacement_dict:
                    d[key] = replacement_dict[value]

    # Start processing from the root level
    if isinstance(data, dict):
        process_dict(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                process_dict(item)

    return data

def calculate_accuracy(data, group_key=None, subgroup_key=None, label_key=None):
    correct_predictions = 0
    total_samples = 0
    group_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    subgroup_accuracy = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    label_accuracy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0})))

    for item in data:
        if item['top_5_tokens'][0]['token'] == item['true_label']:
            correct_predictions += 1
            if group_key:
                group_accuracy[item[group_key]]['correct'] += 1
                if subgroup_key:
                    subgroup_accuracy[item[group_key]][item[subgroup_key]]['correct'] += 1
                    if label_key:
                        label_accuracy[item[group_key]][item[subgroup_key]][item[label_key]]['correct'] += 1
        if group_key:
            group_accuracy[item[group_key]]['total'] += 1
            if subgroup_key:
                subgroup_accuracy[item[group_key]][item[subgroup_key]]['total'] += 1
                if label_key:
                    label_accuracy[item[group_key]][item[subgroup_key]][item[label_key]]['total'] += 1
        total_samples += 1

    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    return overall_accuracy, group_accuracy, subgroup_accuracy, label_accuracy

def report_accuracy_to_list(label_accuracy, group_key, subgroup_key, label_key):
    df_list = []
    
    # Iterating through the hierarchical structure
    for group, subgroups in label_accuracy.items():
        for subgroup, labels in subgroups.items():
            for label, stats in labels.items():
                accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                df_list.append({
                    group_key: group,
                    subgroup_key: subgroup,
                    label_key: label,
                    'Accuracy': accuracy,
                    'Correct': stats['correct'],
                    'Total': stats['total']
                })
    return df_list


def main():
    parser = argparse.ArgumentParser(description="Metric 3 Evaluation")
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSON file from evaluation.py')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output CSV file to save metric results')
    parser.add_argument('--mapping_file', type=str, required=True, help='Path to key mapping JSON file')
    args = parser.parse_args()

    # Load the JSON data
    with open(args.input_file, 'r') as file:
        data = json.load(file)


    # Load the key mappings
    key_mapping = load_key_mapping(args.mapping_file)

    # Create a reverse dictionary for the key mappings
    map_kep = {v: k for k, v in key_mapping.items()}
    
    data = load_and_process_json(data, map_kep)
    
    # Accuracy across different Case_id groups
    for item in data:
        if 'A_Validate' in item['Case_id']:
            item['Case_id_group'] = 'A_Validate'
        elif 'A_Verify' in item['Case_id']:
            item['Case_id_group'] = 'A_Verify'
        elif 'Dialogue_G' in item['Case_id']:
            item['Case_id_group'] = 'Dialogue_G'
        else:
            item['Case_id_group'] = 'OTHER'

    # Overall accuracy
    _, _, _, label_accuracy = calculate_accuracy(data, group_key='Case_id_group', subgroup_key='preceding_token', label_key='true_label')


    # Creating DataFrame from the list of dictionaries
    df = pd.DataFrame(report_accuracy_to_list(label_accuracy, group_key='Case_id_group', subgroup_key='preceding_token', label_key='true_label'))

    # df = pd.DataFrame(df_list)
    # Save to CSV
    df.to_csv(args.output_file, index=False)
    print(f"PRM_label_level_and_ORM_scores results saved to {args.output_file}")

if __name__ == '__main__':
    main()
