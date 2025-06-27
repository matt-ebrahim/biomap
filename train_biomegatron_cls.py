#!/usr/bin/env python3
"""
Train biomedical language model fine-tuned as classifier for entity linking.

This script:
1. Allows users to choose between different biomedical language models
2. Creates positive and negative training pairs with proper negative sampling
3. Fine-tunes selected model for ranking MONDO candidates
4. Saves the BEST model based on validation loss (not final model)
5. Creates comprehensive training metrics and visualizations
6. Supports checkpoint resumption for continued training
7. Organizes models by parameter size and type
"""

import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import TrainerCallback
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
import pathlib
import os
import json
import glob
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# Set high-quality matplotlib settings for production plots
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
sns.set_style("whitegrid")

# Model configurations for different biomedical language models
BIOMEDICAL_MODELS = {
    "biomegatron-345m": {
        "model_name": "EMBO/BioMegatron345mUncased",
        "display_name": "BioMegatron 345M",
        "parameters": "345m",
        "description": "BioMegatron 345M parameters, pretrained on PubMed abstracts and full-text articles",
        "verified": True
    },
    "clinical-bert": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "display_name": "Clinical BERT",
        "parameters": "110m",
        "description": "Clinical BERT pretrained on clinical notes and biomedical text",
        "verified": True
    },
    "pubmed-bert": {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "display_name": "PubMed BERT",
        "parameters": "110m", 
        "description": "BERT pretrained on PubMed abstracts",
        "verified": True
    },
    "biobert": {
        "model_name": "dmis-lab/biobert-base-cased-v1.1",
        "display_name": "BioBERT",
        "parameters": "110m",
        "description": "BioBERT pretrained on PubMed and PMC articles",
        "verified": True
    },
    "biomistral-7b": {
        "model_name": "BioMistral/BioMistral-7B",
        "display_name": "BioMistral 7B",
        "parameters": "7b",
        "description": "BioMistral 7B parameters, medical domain LLM based on Mistral",
        "verified": True
    }
}

def get_model_parameters_count(model):
    """Get the actual number of parameters from a loaded model."""
    try:
        return sum(p.numel() for p in model.parameters())
    except:
        return None

def format_parameter_count(param_count):
    """Format parameter count to human readable string."""
    if param_count is None:
        return "unknown"
    
    if param_count >= 1e9:
        return f"{param_count/1e9:.1f}B"
    elif param_count >= 1e6:
        return f"{param_count/1e6:.0f}M"
    elif param_count >= 1e3:
        return f"{param_count/1e3:.0f}K"
    else:
        return str(param_count)

def verify_model_availability(model_key):
    """Verify if a model is available on HuggingFace."""
    if model_key not in BIOMEDICAL_MODELS:
        return False
    
    model_info = BIOMEDICAL_MODELS[model_key]
    if not model_info.get("verified", False):
        print(f"⚠️  Warning: {model_info['display_name']} availability not verified")
        return False
    
    try:
        from transformers import AutoTokenizer
        # Quick check if model exists
        AutoTokenizer.from_pretrained(model_info["model_name"])
        return True
    except Exception as e:
        print(f"❌ Error: Could not load {model_info['display_name']}: {e}")
        return False

def list_available_models():
    """List all available biomedical models with their details."""
    print("\n🔬 Available Biomedical Language Models:")
    print("=" * 80)
    
    for key, info in BIOMEDICAL_MODELS.items():
        status = "✅ Verified" if info.get("verified", False) else "⚠️  Unverified"
        print(f"\n📋 Model ID: {key}")
        print(f"   Name: {info['display_name']}")
        print(f"   Parameters: {info['parameters']}")
        print(f"   Status: {status}")
        print(f"   Description: {info['description']}")
        print(f"   HuggingFace: {info['model_name']}")
    
    print("\n" + "=" * 80)
    print("💡 Note: BioMegatron 800M and 1.2B variants exist but may not be publicly available on HuggingFace")
    print("💡 For larger models, consider BioMistral-7B which is readily available and well-performing")
    print("=" * 80)

def create_model_directory_name(model_info, actual_params=None):
    """Create directory name based on model info."""
    if actual_params:
        param_str = format_parameter_count(actual_params).lower()
    else:
        param_str = model_info['parameters']
    
    # Extract base model name (remove organization prefix)
    base_name = model_info['model_name'].split('/')[-1].lower()
    base_name = re.sub(r'[^a-z0-9]', '_', base_name)
    
    return f"models_{param_str}_{base_name}"

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the given directory."""
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None, 0
    
    # Extract step numbers and find the latest
    checkpoint_steps = []
    for checkpoint in checkpoints:
        match = re.search(r'checkpoint-(\d+)', checkpoint)
        if match:
            checkpoint_steps.append((int(match.group(1)), checkpoint))
    
    if not checkpoint_steps:
        return None, 0
    
    # Sort by step number and get the latest
    checkpoint_steps.sort(key=lambda x: x[0])
    latest_step, latest_checkpoint = checkpoint_steps[-1]
    
    # Estimate epoch from trainer state if available
    trainer_state_file = os.path.join(latest_checkpoint, "trainer_state.json")
    epoch = 0
    if os.path.exists(trainer_state_file):
        with open(trainer_state_file, 'r') as f:
            state = json.load(f)
            epoch = int(state.get('epoch', 0))
    
    return latest_checkpoint, epoch

def ask_resume_training(checkpoint_path, checkpoint_epoch, target_epochs):
    """Ask user whether to resume training or start from scratch."""
    print(f"\nFound existing checkpoint at: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Target epochs: {target_epochs}")
    
    if checkpoint_epoch >= target_epochs:
        print(f"Checkpoint epoch ({checkpoint_epoch}) >= target epochs ({target_epochs})")
        print("Training is already complete or exceeds target epochs.")
        return False
    
    print(f"Resuming would continue training from epoch {checkpoint_epoch + 1} to {target_epochs}")
    
    while True:
        choice = input("Resume training from checkpoint? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")

class MetricsCallback(TrainerCallback):
    """Custom callback to track training metrics for visualization."""
    
    def __init__(self):
        self.training_loss = []
        self.validation_loss = []
        self.epochs = []
        self.learning_rates = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.training_loss.append(logs['loss'])
                self.learning_rates.append(logs.get('learning_rate', 0))
            if 'eval_loss' in logs:
                self.validation_loss.append(logs['eval_loss'])
                self.epochs.append(state.epoch)

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
    print("Using proper negative sampling with representative mention text")
    
    # Show examples
    print("\nSample positive pairs:")
    for i in range(min(3, len(positive_pairs))):
        print(f"  '{positive_pairs[i]['input']}' → {positive_pairs[i]['label']}")
    
    print("\nSample negative pairs:")
    for i in range(min(3, len(negative_pairs))):
        print(f"  '{negative_pairs[i]['input']}' → {negative_pairs[i]['label']}")
    
    return paired_df

def tokenize(batch):
    """Tokenize input pairs for the model."""
    return tok(
        batch["input"], 
        truncation=True, 
        padding="max_length",
        max_length=64  # Keep sequences short for efficiency
    )

def create_training_plots(metrics_callback, metrics_dir):
    """Create high-quality training and validation loss plots."""
    
    # Training & Validation Loss Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training Loss over Steps
    steps = range(len(metrics_callback.training_loss))
    ax1.plot(steps, metrics_callback.training_loss, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Validation Loss over Epochs
    if metrics_callback.epochs and metrics_callback.validation_loss:
        ax2.plot(metrics_callback.epochs, metrics_callback.validation_loss, 'r-o', 
                linewidth=2, markersize=8, label='Validation Loss', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Over Epochs', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Mark best epoch
        best_epoch_idx = np.argmin(metrics_callback.validation_loss)
        best_epoch = metrics_callback.epochs[best_epoch_idx]
        best_loss = metrics_callback.validation_loss[best_epoch_idx]
        ax2.annotate(f'Best: Epoch {best_epoch:.1f}\nLoss: {best_loss:.4f}', 
                    xy=(best_epoch, best_loss), xytext=(10, 10),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(metrics_dir / 'training_validation_loss.png', dpi=300, bbox_inches='tight')
    plt.savefig(metrics_dir / 'training_validation_loss.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {metrics_dir}/training_validation_loss.png")
    print(f"Saved: {metrics_dir}/training_validation_loss.pdf")

def create_roc_analysis(dev, final_model_path, metrics_dir):
    """Create ROC curve analysis on validation data."""
    
    print("Generating ROC curve analysis...")
    
    # Load the best model for evaluation
    best_model = AutoModelForSequenceClassification.from_pretrained(str(final_model_path))
    best_tokenizer = AutoTokenizer.from_pretrained(str(final_model_path))
    
    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    best_model.to(device)
    best_model.eval()
    
    # Get predictions on validation set
    val_inputs = []
    val_labels = []
    
    for example in dev.itertuples():
        val_inputs.append(example.input)
        val_labels.append(example.label)
    
    # Tokenize validation inputs
    val_encodings = best_tokenizer(
        val_inputs,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )
    
    # Move to device
    val_encodings = {k: v.to(device) for k, v in val_encodings.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = best_model(**val_encodings)
        predictions = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    
    # Convert labels to numpy
    val_labels = np.array(val_labels)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(val_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Create ROC plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})', alpha=0.8)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curve - Biomedical Entity Linking Classifier', fontweight='bold', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add AUC text box
    plt.text(0.6, 0.2, f'AUC Score: {roc_auc:.4f}', fontsize=14,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(metrics_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(metrics_dir / 'roc_curve.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {metrics_dir}/roc_curve.png")
    print(f"Saved: {metrics_dir}/roc_curve.pdf")
    
    # Classification report
    binary_predictions = (predictions > 0.5).astype(int)
    class_report = classification_report(val_labels, binary_predictions, 
                                       target_names=['Negative', 'Positive'], 
                                       output_dict=True)
    
    return roc_auc, class_report

def train_biomedical_classifier(args, model_info):
    """Main training function for biomedical classifier."""
    
    # Load and prepare training data
    print("\nLoading training and development datasets...")
    train = pair_df(pd.read_csv('data/mondo_train.csv'))
    dev = pair_df(pd.read_csv('data/mondo_dev.csv'))
    
    print("\nLoading model and tokenizer...")
    try:
        global tok
        tok = AutoTokenizer.from_pretrained(model_info['model_name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            model_info['model_name'], 
            num_labels=1  # regression score for ranking
        )
        
        # Get actual parameter count
        actual_param_count = get_model_parameters_count(model)
        actual_param_str = format_parameter_count(actual_param_count)
        
        print(f"Model loaded successfully!")
        print(f"Actual parameters: {actual_param_str} ({actual_param_count:,} parameters)")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check if the model is available and accessible.")
        return
    
    # Create directories with model-specific naming
    models_dir = pathlib.Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Create model-specific directory
    model_dir_name = create_model_directory_name(model_info, actual_param_count)
    model_base_dir = models_dir / model_dir_name
    model_base_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = model_base_dir / "checkpoints"
    final_model_dir = model_base_dir / "final_model"
    metrics_dir = model_base_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    print(f"Model directory: {model_base_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Final model: {final_model_dir}")
    print(f"Metrics: {metrics_dir}")
    
    # Check for existing checkpoints
    latest_checkpoint, checkpoint_epoch = find_latest_checkpoint(str(checkpoint_dir))
    resume_from_checkpoint = None
    
    if latest_checkpoint and checkpoint_epoch < args.epochs:
        if ask_resume_training(latest_checkpoint, checkpoint_epoch, args.epochs):
            resume_from_checkpoint = latest_checkpoint
            print(f"Will resume training from {latest_checkpoint}")
        else:
            print("Starting training from scratch")
    elif latest_checkpoint and checkpoint_epoch >= args.epochs:
        print(f"Training already completed ({checkpoint_epoch} epochs >= {args.epochs} target epochs)")
        print("If you want to train more epochs, increase the --epochs parameter")
        return
    
    print("\nTokenizing datasets...")
    ds_train = Dataset.from_pandas(train).map(tokenize, batched=True)
    ds_dev = Dataset.from_pandas(dev).map(tokenize, batched=True)
    
    # Initialize metrics callback
    metrics_callback = MetricsCallback()
    
    # Training arguments with BEST MODEL SAVING based on eval_loss
    args_training = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=3,   # Keep only 3 best checkpoints to save space
        load_best_model_at_end=True,        # Load best model at end
        metric_for_best_model="eval_loss",   # Use eval_loss as metric
        greater_is_better=False,             # Lower eval_loss is better
        report_to=None,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        dataloader_pin_memory=False,  # Disable pin_memory for MPS compatibility
    )
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=args_training,
        train_dataset=ds_train.remove_columns(['input']),
        eval_dataset=ds_dev.remove_columns(['input']),
        tokenizer=tok,
        callbacks=[metrics_callback]  # Add metrics tracking
    )
    
    if resume_from_checkpoint:
        print(f"Starting training from checkpoint: {resume_from_checkpoint}")
        print(f"Resuming from epoch {checkpoint_epoch + 1} to {args.epochs}")
    else:
        print("Starting training from scratch...")
        print(f"Training examples: {len(ds_train)}")
        print(f"Evaluation examples: {len(ds_dev)}")
        print(f"Training for {args_training.num_train_epochs} epochs")
    
    print("Best model will be saved based on lowest validation loss")
    
    # Train the model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the final model (which is actually the BEST model due to load_best_model_at_end=True)
    print(f"Saving BEST model (lowest eval_loss) to {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    
    # Save model metadata
    model_metadata = {
        "model_key": args.model,
        "model_name": model_info['model_name'],
        "display_name": model_info['display_name'],
        "description": model_info['description'],
        "actual_parameters": actual_param_count,
        "formatted_parameters": actual_param_str,
        "training_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_length": args.max_length,
        "negative_samples": args.negative_samples,
        "resumed_from_checkpoint": resume_from_checkpoint is not None,
        "checkpoint_path": resume_from_checkpoint if resume_from_checkpoint else None
    }
    
    with open(model_base_dir / 'model_info.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Generate comprehensive metrics and visualizations
    print("\n" + "="*60)
    print("GENERATING PRODUCTION-QUALITY METRICS & VISUALIZATIONS")
    print("="*60)
    
    create_training_plots(metrics_callback, metrics_dir)
    roc_auc, class_report = create_roc_analysis(dev, final_model_dir, metrics_dir)
    
    # Save metrics summary
    metrics_summary = {
        'model_info': model_metadata,
        'roc_auc': float(roc_auc),
        'best_epoch': float(metrics_callback.epochs[np.argmin(metrics_callback.validation_loss)]) if metrics_callback.validation_loss else None,
        'best_validation_loss': float(min(metrics_callback.validation_loss)) if metrics_callback.validation_loss else None,
        'final_training_loss': float(metrics_callback.training_loss[-1]) if metrics_callback.training_loss else None,
        'classification_report': class_report,
        'total_training_samples': len(ds_train),
        'total_validation_samples': len(ds_dev),
    }
    
    with open(metrics_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"Saved: {metrics_dir}/metrics_summary.json")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model saved to: {final_model_dir}")
    print(f"Model info: {model_base_dir}/model_info.json")
    print(f"Metrics: {metrics_dir}/")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    if metrics_callback.validation_loss:
        best_val_loss = min(metrics_callback.validation_loss)
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Train biomedical language model for entity linking')
    parser.add_argument('--model', type=str, required=True, 
                        help='Model to use (e.g., biomegatron-345m, clinical-bert, pubmed-bert, biobert, biomistral-7b)')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Training batch size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--max_length', type=int, default=128, 
                        help='Maximum sequence length (default: 128)')
    parser.add_argument('--negative_samples', type=int, default=3,
                        help='Number of negative samples per positive (default: 3)')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--list_models', action='store_true',
                        help='List available models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    # Verify model availability
    if not verify_model_availability(args.model):
        print(f"\n❌ Model '{args.model}' is not available or verified.")
        print("Use --list_models to see available models.")
        return
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    
    # Get model configuration
    model_info = BIOMEDICAL_MODELS[args.model]
    
    print(f"\n🚀 Starting training with {model_info['display_name']}")
    print(f"📊 Parameters: {model_info['parameters']}")
    print(f"🤗 HuggingFace: {model_info['model_name']}")
    print(f"📝 Description: {model_info['description']}")
    
    # Train the model
    train_biomedical_classifier(args, model_info)

if __name__ == "__main__":
    main() 