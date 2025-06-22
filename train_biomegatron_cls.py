#!/usr/bin/env python3
"""
Train BioMegatron fine-tuned as classifier for entity linking.

This script:
1. Loads mention-MONDO pairs from CSV files
2. Creates positive and negative training pairs with proper negative sampling
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
from collections import defaultdict

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

def create_mention_to_representative_text(df):
    """Create mapping from MONDO ID to representative mention text."""
    mondo_to_mentions = df.groupby('mondo_id')['mention'].apply(list).to_dict()
    
    mondo_to_text = {}
    for mondo_id, mentions in mondo_to_mentions.items():
        # Use most frequent mention as representative
        mention_counts = pd.Series(mentions).value_counts()
        representative_mention = mention_counts.index[0]
        mondo_to_text[mondo_id] = representative_mention
    
    return mondo_to_text

def pair_df(df):
    """
    Generate mention x MONDO text pairs for training with proper negative sampling.
    """
    print(f"Processing {len(df)} mention-MONDO pairs...")
    
    # Create MONDO ID to representative text mapping
    mondo_to_text = create_mention_to_representative_text(df)
    
    # Create mention -> valid MONDO IDs mapping for proper negative sampling
    mention_to_mondos = defaultdict(set)
    for _, row in df.iterrows():
        mention_to_mondos[row['mention']].add(row['mondo_id'])
    
    all_mondo_ids = set(df['mondo_id'].unique())
    
    # Positive pairs: actual mention-MONDO matches using representative text
    positive_pairs = []
    for _, row in df.iterrows():
        mention = row['mention']
        mondo_id = row['mondo_id']
        representative_text = mondo_to_text[mondo_id]
        
        positive_pairs.append({
            'mention': mention,
            'mondo_id': mondo_id,
            'input': f"{mention} [SEP] {representative_text}",
            'label': 1.0
        })
    
    # Negative pairs: ensure they are actually incorrect
    negative_pairs = []
    for _, row in df.iterrows():
        mention = row['mention']
        valid_mondos = mention_to_mondos[mention]
        invalid_mondos = all_mondo_ids - valid_mondos
        
        if len(invalid_mondos) > 0:
            # Sample a random invalid MONDO ID
            negative_mondo = np.random.choice(list(invalid_mondos))
            representative_text = mondo_to_text[negative_mondo]
            
            negative_pairs.append({
                'mention': mention,
                'mondo_id': negative_mondo,
                'input': f"{mention} [SEP] {representative_text}",
                'label': 0.0
            })
    
    # Combine positive and negative pairs
    all_pairs = positive_pairs + negative_pairs
    paired_df = pd.DataFrame(all_pairs)
    
    print(f"Created {len(paired_df)} total pairs ({len(positive_pairs)} positive, {len(negative_pairs)} negative)")
    print("✅ FIXED: Using proper negative sampling with representative mention text")
    
    # Show examples
    print("\nSample positive pairs:")
    for i in range(min(3, len(positive_pairs))):
        print(f"  '{positive_pairs[i]['input']}' → {positive_pairs[i]['label']}")
    
    print("\nSample negative pairs:")
    for i in range(min(3, len(negative_pairs))):
        print(f"  '{negative_pairs[i]['input']}' → {negative_pairs[i]['label']}")
    
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

# Training arguments with improved settings
args = TrainingArguments(
    output_dir=str(models_dir / "biomegatron_mondo_cls"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Increased epochs for better learning
    learning_rate=1e-5,  # Reduced learning rate for stability
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=None,
    warmup_steps=100,  # Add warmup for stability
    weight_decay=0.01,  # Add regularization
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
print("- Feed mention + [SEP] + representative_mention pairs")
print("- Rank candidates by regression score")
print("- Higher scores indicate better matches")
print("🔧 FIXED: Proper negative sampling and representative text usage") 