import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from torch.nn import functional as F
import json
import pandas as pd
from tqdm import tqdm
import yaml
import argparse
import random
import numpy as np
from transformers import set_seed


def load_config(config_file_path):
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    return config

def load_special_tokens(special_tokens_file):
    with open(special_tokens_file, 'r') as f:
        special_tokens_dic = json.load(f)
    return special_tokens_dic

def load_model_and_tokenizer(model_name_or_path, special_tokens_dic):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Add special tokens to tokenizer
    print(f"Tokenizer size before adding special tokens: {len(tokenizer)}")
    
    # Convert special tokens to required format
    special_tokens = {'additional_special_tokens': list(special_tokens_dic.values())}
    
    # Add special tokens to tokenizer
    tokenizer.add_special_tokens(special_tokens)
    print(f"Tokenizer size after adding special tokens: {len(tokenizer)}")
    
    # # Print all special tokens
    print(f"Special tokens: {tokenizer.special_tokens_map}")

    
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    
    return model, tokenizer

def process_cases(df_merged, tokenizer, model, special_tokens_dic, device):
    results = []
    model.eval()
    
    placeholder_token = special_tokens_dic['Placeholder_token']
    plus_token = special_tokens_dic['+']
    minus_token = special_tokens_dic['-']
    
    # Get token IDs
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    plus_token_id = tokenizer.convert_tokens_to_ids(plus_token)
    minus_token_id = tokenizer.convert_tokens_to_ids(minus_token)
    
    # Process each case
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Processing cases"):
        Case_id = row['Case_id']
        Sample_no = row['Sample_no']
        input_text = row['Training_note']
        
        # Tokenize without padding or truncation
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            add_special_tokens=True,
            padding=False,
            truncation=False
        )

    
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
    
        # Identify positions of <Placeholder_token>
        placeholder_positions = (input_ids == placeholder_token_id).nonzero(as_tuple=False)
    
        if placeholder_positions.numel() == 0:
            print(f"No '{placeholder_token}' found in input text at index {idx}. Skipping.")
            continue
    
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits  # Shape: [1, sequence_length, vocab_size]
    
        # For each placeholder position
        for step_no, pos in enumerate(placeholder_positions):
            batch_index, seq_position = pos.tolist()
            Step_no = step_no  # Step number starting from 0
    
            # Get preceding 10 tokens
            start_pos = max(seq_position - 10, 0)
            preceding_token_ids = input_ids[batch_index, start_pos:seq_position]
            preceding_tokens = tokenizer.convert_ids_to_tokens(preceding_token_ids.tolist())
    
            # Immediate preceding token
            if seq_position > 0:
                preceding_token_id = input_ids[batch_index, seq_position - 1]
                preceding_token = tokenizer.convert_ids_to_tokens([preceding_token_id.item()])[0]
            else:
                preceding_token = None
    
            # Extract logits at the position before the placeholder_token
            # Because logits[t] corresponds to the prediction for token at position t+1
            token_logits = logits[batch_index, seq_position - 1, :]  # Corrected indexing
            token_probs = F.softmax(token_logits, dim=-1)  # Shape: [vocab_size]
    
            # Get logits and probabilities for "+" and "-"
            plus_logit = token_logits[plus_token_id].item()
            minus_logit = token_logits[minus_token_id].item()
            plus_prob = token_probs[plus_token_id].item()
            minus_prob = token_probs[minus_token_id].item()
    
            # Get top 5 tokens by probability
            topk_probs, topk_indices = torch.topk(token_probs, k=5)
            topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices.tolist())
            top_5_tokens = [{'token': token, 'prob': prob.item()} for token, prob in zip(topk_tokens, topk_probs)]

            # Prepare result for this placeholder
            result = {
                'Case_id': Case_id,
                'Sample_no': Sample_no,
                'Step_no': Step_no,
                'preceding_10_tokens': preceding_tokens,
                'preceding_token': preceding_token,
                'top_5_tokens': top_5_tokens,
                'plus_logit': plus_logit,
                'minus_logit': minus_logit,
                'plus_prob': plus_prob,
                'minus_prob': minus_prob
            }
            results.append(result)
    
    return results

def main():

    # Set seed
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    set_seed(seed_value)

    # Load configuration
    # Argument parsing
    parser = argparse.ArgumentParser(description='Run forward pass for a model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint or directory.')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--input_csv_file', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output JSON file.')
    args = parser.parse_args()

    # Load configuration
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Override config parameters with command-line arguments
    config['model_name_or_path'] = args.model_path
    config['input_csv_file'] = args.input_csv_file
    config['output_file'] = args.output_file
    
    # Set environment variables if needed
    if 'HF_HOME' in config:
        os.environ['HF_HOME'] = config['HF_HOME']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load special tokens
    special_tokens_dic = load_special_tokens(config['special_tokens_file'])
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config['model_name_or_path'], special_tokens_dic)
    
    # Move model to device
    model.to(device)
    
    # Load dataframes
    df = pd.read_csv(config['input_csv_file'])

    # Process cases
    results = process_cases(df, tokenizer, model, special_tokens_dic, device)
    
    # Save results to a JSON file
    output_file = config['output_file']
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
