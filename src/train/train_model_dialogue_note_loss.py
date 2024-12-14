import os
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
    # Set environment variables
    print(f"Is cuda available: {torch.cuda.is_available()}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--file_path', type=str, help='File path of dataset')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--loss_mode', type=str, help='Loss mode: vanilla, special_token, or score_token_only', default='vanilla')
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
    if args.loss_mode:
        config['loss_mode'] = args.loss_mode
    else:
        config['loss_mode'] = 'vanilla'

    # Create wandb run name dynamically
    wandb_run_name = f"{config['model_name'].split('/')[-1]}_lr_{config['learning_rate']}_{dataset_name}_{config['loss_mode']}"
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
    def tokenize_function(examples):
        return tokenizer([text + tokenizer.eos_token for text in examples["Training_note"]],
                         truncation=False,
                         add_special_tokens=True)

    tokenized_dataset = dataset.map(tokenize_function,
                                    batched=True,
                                    remove_columns=["Training_note"])

    # Load evaluation dataset
    eval_dataset = load_dataset("csv", data_files=config['eval_file_path'])
    
    # Optional: Filter out evaluation samples if needed
    eval_dataset = eval_dataset.filter(lambda x: 'ACIBENCH_rlhf' in x['Case_id'])
    
    eval_tokenized_dataset = eval_dataset.map(tokenize_function,
                                              batched=True,
                                              remove_columns=["Training_note"])

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

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

    # Custom Trainer class with multiple loss modes
    class CustomTrainer(Trainer):
        def __init__(self, plus_minus_token_ids, other_special_token_ids, loss_mode, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.plus_minus_token_ids = plus_minus_token_ids
            self.other_special_token_ids = other_special_token_ids
            self.loss_mode = loss_mode

        def compute_loss(self, model, inputs, return_outputs=False):
            # Forward pass
            outputs = model(**inputs)
            base_loss = outputs.loss

            if self.loss_mode == 'vanilla':
                # Use the model's default loss
                loss = base_loss
            else:
                # Compute special-token-related losses
                logits = outputs.logits
                labels = inputs.get('labels')

                plus_minus_token_mask = torch.zeros_like(labels).bool()
                for token_id in self.plus_minus_token_ids:
                    plus_minus_token_mask |= (labels == token_id)

                other_special_token_mask = torch.zeros_like(labels).bool()
                for token_id in self.other_special_token_ids:
                    other_special_token_mask |= (labels == token_id)

                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                plus_minus_token_mask = plus_minus_token_mask[..., 1:]
                other_special_token_mask = other_special_token_mask[..., 1:]

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss_per_token = loss_per_token.view(shift_labels.size())

                plus_minus_token_loss = (loss_per_token * plus_minus_token_mask.float()).sum() / plus_minus_token_mask.float().sum().clamp(min=1)
                other_special_token_loss = (loss_per_token * other_special_token_mask.float()).sum() / other_special_token_mask.float().sum().clamp(min=1)

                # Log these intermediate losses
                wandb.log({
                    'all_token_loss': base_loss.item(),
                    'plus_minus_token_loss': plus_minus_token_loss.item(),
                    'other_special_token_loss': other_special_token_loss.item()
                })

                # Determine final loss based on mode
                if self.loss_mode == 'special_token':
                    # Sum of both special token losses
                    loss = plus_minus_token_loss + other_special_token_loss
                elif self.loss_mode == 'score_token_only':
                    # Only plus/minus token loss
                    loss = plus_minus_token_loss
                else:
                    # Default to vanilla if something unexpected is provided
                    loss = base_loss

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

                    plus_minus_token_mask = torch.zeros_like(labels).bool()
                    for token_id in self.plus_minus_token_ids:
                        plus_minus_token_mask |= (labels == token_id)

                    other_special_token_mask = torch.zeros_like(labels).bool()
                    for token_id in self.other_special_token_ids:
                        other_special_token_mask |= (labels == token_id)

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    plus_minus_token_mask = plus_minus_token_mask[..., 1:]
                    other_special_token_mask = other_special_token_mask[..., 1:]

                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss_per_token = loss_per_token.view(shift_labels.size())

                    pm_loss_sum = (loss_per_token * plus_minus_token_mask.float()).sum().item()
                    pm_count = plus_minus_token_mask.float().sum().item()

                    other_loss_sum = (loss_per_token * other_special_token_mask.float()).sum().item()
                    other_count = other_special_token_mask.float().sum().item()

                    total_plus_minus_token_loss += pm_loss_sum
                    total_plus_minus_token_count += pm_count

                    total_other_special_token_loss += other_loss_sum
                    total_other_special_token_count += other_count

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
        loss_mode=config['loss_mode'],
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
    os.makedirs(final_model_path, exist_ok=True)  
    trainer.save_model(final_model_path)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
