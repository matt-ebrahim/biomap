#!/usr/bin/env python3
"""
Train BioMegatron fine-tuned as classifier for entity linking.

This script:
1. Loads mention-MONDO pairs from CSV files
2. Creates positive and negative training pairs
3. Fine-tunes BioMegatron for ranking MONDO candidates
4. Saves the trained model for inference
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
import pathlib
import os

# Create models directory if it doesn't exist
models_dir = pathlib.Path('models')
models_dir.mkdir(exist_ok=True)

print("Loading BioMegatron tokenizer and model...")
# Use community-uploaded BioMegatron model
model_name = "EMBO/BioMegatron345mUncased"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=1  # regression score for ranking
)

def pair_df(df):
    """
    Generate mention x MONDO string pairs for training.
    Creates both positive pairs and negative pairs for contrastive learning.
    """
    print(f"Processing {len(df)} mention-MONDO pairs...")
    
    # Positive pairs: actual mention-MONDO matches
    df['input'] = df['mention'] + ' [SEP] ' + df['mondo_id']
    df['label'] = 1.0  # positive pairs get score 1
    
    # Negative pairs: random incorrect MONDO for each mention
    neg = df.copy()
    neg['mondo_id'] = np.random.permutation(df['mondo_id'].values)
    neg['input'] = neg['mention'] + ' [SEP] ' + neg['mondo_id']
    neg['label'] = 0.0  # negative pairs get score 0
    
    # Combine positive and negative pairs
    paired_df = pd.concat([df, neg], ignore_index=True)
    print(f"Created {len(paired_df)} total pairs ({len(df)} positive, {len(neg)} negative)")
    
    return paired_df

# Load and prepare training data
print("Loading training and development datasets...")
train = pair_df(pd.read_csv('data/mondo_train.csv'))
dev = pair_df(pd.read_csv('data/mondo_dev.csv'))

def tokenize(batch):
    """Tokenize input pairs for the model."""
    return tok(
        batch["input"], 
        truncation=True, 
        padding="max_length",
        max_length=64  # Keep sequences short for efficiency
    )

print("Tokenizing datasets...")
ds_train = Dataset.from_pandas(train).map(tokenize, batched=True)
ds_dev = Dataset.from_pandas(dev).map(tokenize, batched=True)

# Training arguments
args = TrainingArguments(
    output_dir=str(models_dir / "biomegatron_mondo_cls"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,  # Keep only best 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=None,  # Disable wandb/tensorboard logging
)

print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train.remove_columns(['input']),
    eval_dataset=ds_dev.remove_columns(['input']),
    tokenizer=tok
)

print("Starting training...")
print(f"Training examples: {len(ds_train)}")
print(f"Evaluation examples: {len(ds_dev)}")
print(f"Training for {args.num_train_epochs} epochs")

# Train the model
trainer.train()

# Save the final model
final_model_path = models_dir / "biomegatron_mondo_cls_final"
print(f"Saving final model to {final_model_path}")
trainer.save_model(str(final_model_path))

print("Training complete!")
print(f"Model saved in: {final_model_path}")
print("\nFor inference:")
print("- Load the model and tokenizer")
print("- Feed mention + [SEP] + MONDO_candidate pairs")
print("- Rank candidates by regression score")
print("- Higher scores indicate better matches") 