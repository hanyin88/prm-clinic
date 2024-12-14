import os

from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Any
from transformers import PreTrainedTokenizerBase
import argparse
import json
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import wandb
import deepspeed
from datasets import load_dataset
import random
import numpy as np
from transformers import set_seed

def main():
    print(f"Is cuda available: {torch.cuda.is_available()}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--file_path', type=str, help='file_path of dataset')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    args, unknown = parser.parse_known_args()

    # Load configuration from JSON file
    with open('prm-clinic/config/train_config.json', 'r') as config_file:
        config = json.load(config_file)


    # Update config with command-line arguments if provided
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.model_name:
        config['model_name'] = args.model_name
    if args.file_path:
        config['file_path'] = args.file_path
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = "vanilla"


    # Create wandb run name dynamically
    wandb_run_name = f"zero_out_dialogue_{config['model_name'].split('/')[-1]}_lr_{config['learning_rate']}_{dataset_name}"
    config['wandb_run_name'] = wandb_run_name

    # Generate a unique output_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    else:
        config['output_dir'] = f"outputs/{config['wandb_run_name']}"

    # Ensure the output directory exists
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Set seed
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    set_seed(seed_value)

    os.environ['WANDB_API_KEY'] = config['wandb_api_key']

    # Initialize Weights & Biases
    wandb.login()
    wandb.init(project="PRM_Exp", name=config['wandb_run_name'], config=config)
    config = wandb.config  # Update config with wandb.config for hyperparameter search

    # Log initial information
    print("Starting training... Loading data and model.")

    # Load training dataset
    dataset = load_dataset("csv", data_files=config['file_path'])

    # Load tokenizer
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load special tokens
    with open(config['special_tokens_file'], 'r') as special_tokens_file:
        special_tokens_dict = json.load(special_tokens_file)
    special_tokens = {'additional_special_tokens': list(special_tokens_dict.values())}
    tokenizer.add_special_tokens(special_tokens)

    # Separate special tokens into two groups
    plus_minus_tokens = [special_tokens_dict['+'], special_tokens_dict['-']]
    other_special_tokens = [v for k, v in special_tokens_dict.items() if k not in ('+', '-')]

    plus_minus_token_ids = tokenizer.convert_tokens_to_ids(plus_minus_tokens)
    other_special_token_ids = tokenizer.convert_tokens_to_ids(other_special_tokens)

    # Tokenize the dataset
    # def tokenize_function(examples):
    #     return tokenizer([text + tokenizer.eos_token for text in examples["Training_note"]],
    #                      truncation=False,
    #                      add_special_tokens=True)


    def tokenize_function(examples):
        '''
        Tokenizes and prepares input data for training, ensuring that the model computes the loss 
        only over the 'Combined_note' section of the text. This function segments the input text 
        into two parts: 'prompt_and_dialogue' (which includes the Dialogue and prompt) and 
        'combined_note' (the portion over which the loss will be computed), based on a predefined separator.

        Steps:
        1. Identify the Split Point: Locate the separator within each 'Training_note' to divide the text 
        into 'prompt_and_dialogue' and 'combined_note' sections. If the separator is not found, raise 
        a ValueError to flag the issue.
        2. Tokenize Separately: Tokenize each part independently to create distinct input IDs and attention 
        masks for 'prompt_and_dialogue' and 'combined_note'.
        3. Create Labels: Set labels for tokens in 'prompt_and_dialogue' to -100 to mask them out of 
        loss computation. Keep the labels of 'combined_note' tokens as they are.
        4. Concatenate and Return: Concatenate the tokenized inputs, attention masks, and labels for 
        both sections, adding the end-of-sequence token, and return them as the processed input data.
        
        Parameters:
        examples (dict): Dictionary containing the 'Training_note' entries in the dataset.

        Returns:
        dict: A dictionary with lists for 'input_ids', 'attention_mask', and 'labels' ready for model input.
        '''

        tokenized_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
        separator = '###CLINICAL NOTE-ASSESSMENT AND PLAN: \n'
        for training_note in examples["Training_note"]:
            idx = training_note.find(separator)
            if idx == -1:
                raise ValueError(f"Separator '{separator}' not found in training_note: {training_note}")

            prompt_and_dialogue = training_note[:idx + len(separator)]
            combined_note = training_note[idx + len(separator):]

            # Tokenize prompt_and_dialogue with special tokens
            tokenized_prompt_and_dialogue = tokenizer(prompt_and_dialogue, add_special_tokens=True)

            # Tokenize combined_note with special tokens
            tokenized_combined_note = tokenizer(combined_note, add_special_tokens=True)

            # Remove the BOS token from the beginning of combined_note tokens if present
            bos_token_id = tokenizer.bos_token_id
            if tokenized_combined_note['input_ids'][0] == bos_token_id:
                # Remove the first token (BOS token)
                tokenized_combined_note['input_ids'] = tokenized_combined_note['input_ids'][1:]
                tokenized_combined_note['attention_mask'] = tokenized_combined_note['attention_mask'][1:]


            # Concatenate input_ids and attention_mask
            input_ids = tokenized_prompt_and_dialogue['input_ids'] + tokenized_combined_note['input_ids'] + [tokenizer.eos_token_id]
            attention_mask = tokenized_prompt_and_dialogue['attention_mask'] + tokenized_combined_note['attention_mask'] + [1]

            # Create labels: mask out the prompt and dialogue tokens
            labels = [-100] * len(tokenized_prompt_and_dialogue['input_ids']) + tokenized_combined_note['input_ids'] + [tokenizer.eos_token_id]

            # Append to the lists
            tokenized_inputs['input_ids'].append(input_ids)
            tokenized_inputs['attention_mask'].append(attention_mask)
            tokenized_inputs['labels'].append(labels)

        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_function,
                                    batched=True,
                                    remove_columns=["Training_note"])

    # Load evaluation dataset
    eval_dataset = load_dataset("csv", data_files=config['eval_file_path'])

    # Print length of eval dataset before flitering
    print(f"Length of eval dataset before filtering: {len(eval_dataset['train'])}")
    
    #!!!!! Optional: Filter out those with ACIBENCH-rlhf in the Case_id for evaluation
    eval_dataset = eval_dataset.filter(lambda x: 'ACIBENCH_rlhf' in x['Case_id'])

    # Print length of eval dataset after flitering
    print(f"Length of eval dataset after filtering to include only ACIBENCH-rlhf: {len(eval_dataset['train'])}")
    
    eval_tokenized_dataset = eval_dataset.map(tokenize_function,
                                              batched=True,
                                              remove_columns=["Training_note"])

    # # Set up data collator
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    # )

    @dataclass
    class DataCollatorForCausalLMWithPadding:
        """
        Data collator that will dynamically pad the inputs and labels for causal language modeling.
        """
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str] = 'longest'  # Can be True, 'longest', 'max_length'
        max_length: Optional[int] = None

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            # Separate labels from features
            labels = [feature.pop('labels') for feature in features]

            # Pad the inputs using the tokenizer
            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Pad the labels manually to match the input_ids length
            max_seq_length = batch['input_ids'].shape[1]
            padded_labels = []
            for label in labels:
                padding_length = max_seq_length - len(label)
                padded_label = label + [-100] * padding_length
                padded_labels.append(padded_label)
            
            batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)
            return batch
        
    data_collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        overwrite_output_dir=config.get('overwrite_output_dir', True),
        num_train_epochs=config['num_train_epochs'],
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        save_strategy=config.get('save_strategy', 'steps'),
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        logging_steps=config['logging_steps'],
        report_to="wandb",
        evaluation_strategy=config.get('evaluation_strategy', 'steps'),
        eval_steps=config['eval_steps'],
        load_best_model_at_end=config.get('load_best_model_at_end', False),
        bf16=config.get('bf16', True),
        dataloader_num_workers=config.get('dataloader_num_workers', 4),
        warmup_steps=config.get('warmup_steps', 100),
        gradient_checkpointing=config['gradient_checkpointing'],
        deepspeed=config['deep_speed_config_multi'],
    )

    # Load model
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.config.pad_token_id = tokenizer.pad_token_id = 128004
    model.config.eos_token_id = tokenizer.eos_token_id = 128009

    # Move model to GPU
    model.to('cuda')

    # Custom Trainer class
    class CustomTrainer(Trainer):
        def __init__(self, plus_minus_token_ids, other_special_token_ids, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.plus_minus_token_ids = plus_minus_token_ids
            self.other_special_token_ids = other_special_token_ids

        def compute_loss(self, model, inputs, return_outputs=False):
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss

            # Get logits and labels
            logits = outputs.logits
            labels = inputs.get('labels')

            # Compute loss over special tokens
            plus_minus_token_mask = torch.zeros_like(labels).bool()
            for token_id in self.plus_minus_token_ids:
                plus_minus_token_mask |= (labels == token_id)

            other_special_token_mask = torch.zeros_like(labels).bool()
            for token_id in self.other_special_token_ids:
                other_special_token_mask |= (labels == token_id)

            # Shift labels and masks
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            plus_minus_token_mask = plus_minus_token_mask[..., 1:]
            other_special_token_mask = other_special_token_mask[..., 1:]

            # Compute loss per token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_per_token = loss_per_token.view(shift_labels.size())

            # Calculate special token losses
            plus_minus_token_loss = (loss_per_token * plus_minus_token_mask.float()).sum() / plus_minus_token_mask.float().sum().clamp(min=1)
            other_special_token_loss = (loss_per_token * other_special_token_mask.float()).sum() / other_special_token_mask.float().sum().clamp(min=1)

            # Log special token losses to wandb
            wandb.log({
                'all_token_loss': loss.item(),
                'plus_minus_token_loss': plus_minus_token_loss.item(),
                'other_special_token_loss': other_special_token_loss.item()
            })

            if return_outputs:
                return (loss, outputs)
            else:
                return loss

        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
            # Original evaluation loop
            output = super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix
            )

            # Compute special token losses over evaluation dataset
            total_plus_minus_token_loss = 0.0
            total_plus_minus_token_count = 0
            total_other_special_token_loss = 0.0
            total_other_special_token_count = 0

            model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            for inputs in dataloader:
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    labels = inputs.get('labels')

                    # Compute loss over special tokens
                    plus_minus_token_mask = torch.zeros_like(labels).bool()
                    for token_id in self.plus_minus_token_ids:
                        plus_minus_token_mask |= (labels == token_id)

                    other_special_token_mask = torch.zeros_like(labels).bool()
                    for token_id in self.other_special_token_ids:
                        other_special_token_mask |= (labels == token_id)

                    # Shift labels and masks
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    plus_minus_token_mask = plus_minus_token_mask[..., 1:]
                    other_special_token_mask = other_special_token_mask[..., 1:]

                    # Compute loss per token
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss_per_token = loss_per_token.view(shift_labels.size())

                    plus_minus_token_loss = (loss_per_token * plus_minus_token_mask.float()).sum()
                    plus_minus_token_count = plus_minus_token_mask.float().sum().item()

                    other_special_token_loss = (loss_per_token * other_special_token_mask.float()).sum()
                    other_special_token_count = other_special_token_mask.float().sum().item()

                    total_plus_minus_token_loss += plus_minus_token_loss.item()
                    total_plus_minus_token_count += plus_minus_token_count

                    total_other_special_token_loss += other_special_token_loss.item()
                    total_other_special_token_count += other_special_token_count

            average_plus_minus_token_loss = total_plus_minus_token_loss / max(total_plus_minus_token_count, 1)
            average_other_special_token_loss = total_other_special_token_loss / max(total_other_special_token_count, 1)

            # Log special token losses
            wandb.log({
                'eval_all_token_loss': output.metrics['eval_loss'],
                'eval_plus_minus_token_loss': average_plus_minus_token_loss,
                'eval_other_special_token_loss': average_other_special_token_loss
            })

            return output

    # Initialize the Custom Trainer
    trainer = CustomTrainer(
        plus_minus_token_ids=plus_minus_token_ids,
        other_special_token_ids=other_special_token_ids,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=eval_tokenized_dataset['train'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Print DeepSpeed config before training
    with open(config['deep_speed_config_multi'], 'r') as ds_config_file:
        ds_config = json.load(ds_config_file)
    print("DeepSpeed config:")
    print(json.dumps(ds_config, indent=2))

    # Start training
    trainer.train()

    # Save the final model
    final_model_path = os.path.join(config['output_dir'], 'final_model')
    
    # Ensure the directory exists before saving the model or the config file
    os.makedirs(final_model_path, exist_ok=True)  
    
    trainer.save_model(final_model_path)
    
    # with open(os.path.join(final_model_path, 'train_config.json'), 'w') as f:
    #     json.dump(config, f, indent=4)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
