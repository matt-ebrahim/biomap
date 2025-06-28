#!/usr/bin/env python3
"""
Train biomedical language model fine-tuned as classifier for entity linking.

This script supports multiple biomedical language models, creates positive and negative 
training pairs, fine-tunes the selected model for ranking MONDO candidates, and saves 
the best model based on validation loss with comprehensive metrics.
"""

import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import TrainerCallback, set_seed, BitsAndBytesConfig
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

# Production matplotlib settings
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

# Model configurations
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
        print(f"Warning: {model_info['display_name']} availability not verified")
        return False
    
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(model_info["model_name"])
        return True
    except Exception as e:
        print(f"Error: Could not load {model_info['display_name']}: {e}")
        return False

def list_available_models():
    """List all available biomedical models with their details."""
    print("\nAvailable Biomedical Language Models:")
    print("=" * 80)
    
    for key, info in BIOMEDICAL_MODELS.items():
        status = "Verified" if info.get("verified", False) else "Unverified"
        print(f"\nModel ID: {key}")
        print(f"   Name: {info['display_name']}")
        print(f"   Parameters: {info['parameters']}")
        print(f"   Status: {status}")
        print(f"   Description: {info['description']}")
        print(f"   HuggingFace: {info['model_name']}")
    
    print("\n" + "=" * 80)

def create_model_directory_name(model_info, actual_params=None):
    """Create directory name based on model info."""
    if actual_params:
        param_str = format_parameter_count(actual_params).lower()
    else:
        param_str = model_info['parameters']
    
    base_name = model_info['model_name'].split('/')[-1].lower()
    base_name = re.sub(r'[^a-z0-9]', '_', base_name)
    
    return f"models_{param_str}_{base_name}"

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the given directory."""
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None, 0
    
    checkpoint_steps = []
    for checkpoint in checkpoints:
        match = re.search(r'checkpoint-(\d+)', checkpoint)
        if match:
            checkpoint_steps.append((int(match.group(1)), checkpoint))
    
    if not checkpoint_steps:
        return None, 0
    
    checkpoint_steps.sort(key=lambda x: x[0])
    latest_step, latest_checkpoint = checkpoint_steps[-1]
    
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
    print(f"Checkpoint epoch: {checkpoint_epoch}, Target epochs: {target_epochs}")
    
    if checkpoint_epoch >= target_epochs:
        print("Training is already complete or exceeds target epochs.")
        return False
    
    print(f"Resume training from epoch {checkpoint_epoch + 1} to {target_epochs}? (y/n): ", end="")
    response = input().strip().lower()
    return response in ['y', 'yes']

class MetricsCallback(TrainerCallback):
    """Custom callback to track training metrics with persistence across training sessions."""
    
    def __init__(self, metrics_dir=None):
        self.training_loss = []
        self.validation_loss = []
        self.epochs = []
        self._last_train_loss = None
        self.metrics_dir = metrics_dir
        self.metrics_file = None
        
        if self.metrics_dir:
            self.metrics_file = self.metrics_dir / 'loss_history.json'
            self.load_previous_metrics()
    
    def load_previous_metrics(self):
        """Load metrics from previous training sessions."""
        if self.metrics_file and self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.training_loss = data.get('training_loss', [])
                    self.validation_loss = data.get('validation_loss', [])
                    self.epochs = data.get('epochs', [])
                    print(f"Loaded previous metrics: {len(self.epochs)} epochs from {self.metrics_file}")
            except Exception as e:
                print(f"Warning: Could not load previous metrics: {e}")
                self.training_loss = []
                self.validation_loss = []
                self.epochs = []
    
    def save_metrics(self):
        """Save current metrics to file."""
        if self.metrics_file:
            try:
                data = {
                    'training_loss': self.training_loss,
                    'validation_loss': self.validation_loss,
                    'epochs': self.epochs
                }
                with open(self.metrics_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save metrics: {e}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Track training loss when available
            if 'loss' in logs:
                self._last_train_loss = logs['loss']
            
            # Capture metrics at evaluation time (once per epoch)
            if 'eval_loss' in logs:
                current_epoch = state.epoch
                
                # Check if this epoch already exists (resuming training case)
                if current_epoch in self.epochs:
                    # Update existing epoch data
                    epoch_index = self.epochs.index(current_epoch)
                    self.validation_loss[epoch_index] = logs['eval_loss']
                    self.training_loss[epoch_index] = self._last_train_loss
                else:
                    # Add new epoch data
                    self.validation_loss.append(logs['eval_loss'])
                    self.epochs.append(current_epoch)
                    self.training_loss.append(self._last_train_loss)
                
                # Save metrics after each epoch
                self.save_metrics()

def create_mention_to_representative_text(df):
    """Create mapping from MONDO ID to representative mention text."""
    mention_mapping = {}
    for _, row in df.iterrows():
        mondo_id = row['mondo_id']
        mention = row['mention']
        
        if mondo_id not in mention_mapping:
            mention_mapping[mondo_id] = mention
        elif len(mention) < len(mention_mapping[mondo_id]):
            mention_mapping[mondo_id] = mention
    
    return mention_mapping

def pair_df(df):
    """Create positive and negative training pairs from the dataset."""
    mention_mapping = create_mention_to_representative_text(df)
    
    positive_pairs = []
    for _, row in df.iterrows():
        mention = row['mention']
        mondo_id = row['mondo_id']
        representative_text = mention_mapping.get(mondo_id, mondo_id)
        
        positive_pairs.append({
            'input': f"{mention} [SEP] {representative_text}",
            'label': 1.0
        })
    
    # Create negative pairs
    negative_pairs = []
    all_mondo_ids = list(mention_mapping.keys())
    
    for _, row in df.iterrows():
        mention = row['mention']
        correct_mondo_id = row['mondo_id']
        
        # Sample random incorrect MONDO IDs
        incorrect_mondo_ids = [mid for mid in all_mondo_ids if mid != correct_mondo_id]
        sampled_incorrect = np.random.choice(incorrect_mondo_ids, size=min(3, len(incorrect_mondo_ids)), replace=False)
        
        for incorrect_mondo_id in sampled_incorrect:
            representative_text = mention_mapping[incorrect_mondo_id]
            negative_pairs.append({
                'input': f"{mention} [SEP] {representative_text}",
                'label': 0.0
            })
    
    paired_df = pd.DataFrame(positive_pairs + negative_pairs)
    return paired_df

def tokenize(batch):
    """Tokenize input text for training."""
    return tok(
        batch["input"],
        truncation=True,
        padding=True,
        max_length=64
    )

def create_training_plots(metrics_callback, metrics_dir):
    """Create comprehensive training visualization plots with full epoch history."""
    if not metrics_callback.validation_loss:
        print("No validation loss data available for plotting")
        return
    
    # Filter out None values and ensure we have matching data
    valid_indices = []
    valid_epochs = []
    valid_train_loss = []
    valid_val_loss = []
    
    for i, (epoch, train_loss, val_loss) in enumerate(zip(
        metrics_callback.epochs, 
        metrics_callback.training_loss, 
        metrics_callback.validation_loss
    )):
        if train_loss is not None and val_loss is not None:
            valid_indices.append(i)
            valid_epochs.append(epoch)
            valid_train_loss.append(train_loss)
            valid_val_loss.append(val_loss)
    
    if not valid_epochs:
        print("No valid training/validation loss pairs found for plotting")
        return
    
    # Sort by epoch to ensure proper chronological order
    sorted_data = sorted(zip(valid_epochs, valid_train_loss, valid_val_loss))
    valid_epochs, valid_train_loss, valid_val_loss = zip(*sorted_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training and validation loss over all epochs
    ax1.plot(valid_epochs, valid_train_loss, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(valid_epochs, valid_val_loss, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Progress (Epochs 1-{max(valid_epochs):.0f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add epoch range information
    total_epochs = len(valid_epochs)
    min_epoch = min(valid_epochs)
    max_epoch = max(valid_epochs)
    ax1.text(0.02, 0.98, f'Total Epochs: {total_epochs}\nEpoch Range: {min_epoch:.0f}-{max_epoch:.0f}', 
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Loss difference (overfitting monitor)
    if len(valid_train_loss) > 0:
        loss_diff = np.array(valid_val_loss) - np.array(valid_train_loss)
        ax2.plot(valid_epochs, loss_diff, 'g-', linewidth=2, marker='d', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss - Training Loss')
        ax2.set_title('Overfitting Monitor')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Perfect Fit')
        ax2.legend()
        
        # Add statistics
        mean_diff = np.mean(loss_diff)
        std_diff = np.std(loss_diff)
        ax2.text(0.02, 0.98, f'Mean Diff: {mean_diff:.4f}\nStd Diff: {std_diff:.4f}', 
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(metrics_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to {metrics_dir / 'training_metrics.png'}")
    print(f"Plotted {len(valid_epochs)} epochs of training history (epochs {min(valid_epochs):.0f}-{max(valid_epochs):.0f})")

def create_roc_analysis(dev, final_model_path, metrics_dir):
    """Create ROC curve analysis of the final model."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(final_model_path)
        tokenizer = AutoTokenizer.from_pretrained(final_model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for _, row in dev.iterrows():
                inputs = tokenizer(row['input'], return_tensors='pt', truncation=True, padding=True, max_length=64)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                pred = torch.sigmoid(outputs.logits).cpu().numpy()[0][0]
                
                predictions.append(pred)
                true_labels.append(row['label'])
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Model Performance')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(metrics_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Classification report
        binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
        class_report = classification_report(true_labels, binary_predictions, output_dict=True)
        
        return roc_auc, class_report
        
    except Exception as e:
        print(f"Error in ROC analysis: {e}")
        return 0.0, {}

def save_globally_best_model(trainer, metrics_callback, checkpoint_dir, final_model_dir):
    """
    Find the globally best model across all training epochs (including previous sessions)
    and save it as the final model.
    """
    if not metrics_callback.validation_loss:
        print("No validation loss data available, saving current model")
        trainer.save_model(str(final_model_dir))
        return
    
    # Find the epoch with the globally best (lowest) validation loss
    best_val_loss = min(metrics_callback.validation_loss)
    best_epoch_idx = np.argmin(metrics_callback.validation_loss)
    best_epoch = metrics_callback.epochs[best_epoch_idx]
    
    print(f"\nFinding globally best model across all {len(metrics_callback.epochs)} epochs...")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch:.0f}")
    
    # Check if the best epoch corresponds to a saved checkpoint
    checkpoint_pattern = os.path.join(checkpoint_dir, f"checkpoint-*")
    all_checkpoints = glob.glob(checkpoint_pattern)
    
    # Map checkpoint steps to epochs using trainer state files
    checkpoint_epochs = {}
    for checkpoint_path in all_checkpoints:
        trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_file):
            try:
                with open(trainer_state_file, 'r') as f:
                    state = json.load(f)
                    epoch = int(state.get('epoch', 0))
                    checkpoint_epochs[epoch] = checkpoint_path
            except:
                continue
    
    # Try to find the checkpoint for the best epoch
    best_checkpoint_path = None
    
    # First, try exact match
    if best_epoch in checkpoint_epochs:
        best_checkpoint_path = checkpoint_epochs[best_epoch]
        print(f"Found exact checkpoint for best epoch {best_epoch:.0f}: {best_checkpoint_path}")
    else:
        # If exact match not found, find the closest saved checkpoint
        available_epochs = sorted(checkpoint_epochs.keys())
        if available_epochs:
            closest_epoch = min(available_epochs, key=lambda x: abs(x - best_epoch))
            best_checkpoint_path = checkpoint_epochs[closest_epoch]
            print(f"Best epoch {best_epoch:.0f} checkpoint not found, using closest: epoch {closest_epoch} ({best_checkpoint_path})")
    
    # Load and save the best model
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        try:
            print(f"Loading best model from: {best_checkpoint_path}")
            # Load the model from the best checkpoint
            best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_path)
            best_tokenizer = AutoTokenizer.from_pretrained(best_checkpoint_path)
            
            # Save to final model directory
            best_model.save_pretrained(str(final_model_dir))
            best_tokenizer.save_pretrained(str(final_model_dir))
            
            print(f"Globally best model saved to: {final_model_dir}")
            print(f"Model corresponds to epoch {best_epoch:.0f} with validation loss {best_val_loss:.4f}")
            
        except Exception as e:
            print(f"Error loading best checkpoint: {e}")
            print("Falling back to saving current model")
            trainer.save_model(str(final_model_dir))
    else:
        print("Best checkpoint not found, saving current model")
        trainer.save_model(str(final_model_dir))

def train_biomedical_classifier(args, model_info):
    """Main training function for biomedical classifier."""
    
    train = pair_df(pd.read_csv('data/mondo_train.csv'))
    dev = pair_df(pd.read_csv('data/mondo_dev.csv'))
    
    try:
        global tok
        tok = AutoTokenizer.from_pretrained(model_info['model_name'])
        
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        
        # Try 8-bit quantization first, fallback to regular loading if not supported
        try:
            # Configure 8-bit quantization to reduce memory usage
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_info['model_name'],
                num_labels=1,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("Model loaded with 8-bit quantization")
        except Exception as quant_error:
            print(f"8-bit quantization failed: {quant_error}")
            print("Falling back to regular model loading...")
            # Fallback to regular loading without quantization (Mac-compatible)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_info['model_name'],
                num_labels=1,
                low_cpu_mem_usage=True
            )
        
        actual_param_count = get_model_parameters_count(model)
        actual_param_str = format_parameter_count(actual_param_count)
        
        print(f"Model loaded: {model_info['display_name']} ({actual_param_str})")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create directories
    models_dir = pathlib.Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_dir_name = create_model_directory_name(model_info, actual_param_count)
    model_base_dir = models_dir / model_dir_name
    model_base_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = model_base_dir / "checkpoints"
    final_model_dir = model_base_dir / "final_model"
    metrics_dir = model_base_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    # Check for existing checkpoints
    latest_checkpoint, checkpoint_epoch = find_latest_checkpoint(str(checkpoint_dir))
    resume_from_checkpoint = None
    
    if latest_checkpoint and checkpoint_epoch < args.epochs:
        if ask_resume_training(latest_checkpoint, checkpoint_epoch, args.epochs):
            resume_from_checkpoint = latest_checkpoint
        else:
            print("Starting training from scratch")
    elif latest_checkpoint and checkpoint_epoch >= args.epochs:
        print(f"Training already completed ({checkpoint_epoch} epochs >= {args.epochs} target epochs)")
        return
    
    # Tokenize datasets
    ds_train = Dataset.from_pandas(train).map(tokenize, batched=True)
    ds_dev = Dataset.from_pandas(dev).map(tokenize, batched=True)
    
    metrics_callback = MetricsCallback(metrics_dir)
    
    # Training arguments
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
        save_total_limit=10,  # Increased to preserve more checkpoints for global best model selection
        load_best_model_at_end=False,  # Disabled: we implement global best model selection
        report_to=None,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=args_training,
        train_dataset=ds_train.remove_columns(['input']),
        eval_dataset=ds_dev.remove_columns(['input']),
        processing_class=tok,
        callbacks=[metrics_callback]
    )
    
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        print(f"Training {len(ds_train)} examples for {args_training.num_train_epochs} epochs")
    
    # Train the model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Find and save the globally best model across all epochs (including previous sessions)
    save_globally_best_model(trainer, metrics_callback, checkpoint_dir, final_model_dir)
    
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
    
    # Generate metrics and visualizations
    create_training_plots(metrics_callback, metrics_dir)
    roc_auc, class_report = create_roc_analysis(dev, final_model_dir, metrics_dir)
    
    # Save metrics summary
    metrics_summary = {
        'model_info': model_metadata,
        'roc_auc': float(roc_auc),
        'best_epoch': float(metrics_callback.epochs[np.argmin(metrics_callback.validation_loss)]) if metrics_callback.validation_loss else None,
        'best_validation_loss': float(min(metrics_callback.validation_loss)) if metrics_callback.validation_loss else None,
        'final_training_loss': float(metrics_callback.training_loss[-1]) if metrics_callback.training_loss and metrics_callback.training_loss[-1] is not None else None,
        'classification_report': class_report,
        'total_training_samples': len(ds_train),
        'total_validation_samples': len(ds_dev),
        'total_epochs_trained': len(metrics_callback.epochs),
        'epoch_range': f"{min(metrics_callback.epochs):.0f}-{max(metrics_callback.epochs):.0f}" if metrics_callback.epochs else None,
        'training_sessions': 'resumed' if resume_from_checkpoint else 'fresh_start',
        'best_model_selection': 'global_across_all_epochs',
        'globally_best_epoch': float(metrics_callback.epochs[np.argmin(metrics_callback.validation_loss)]) if metrics_callback.validation_loss else None
    }
    
    with open(metrics_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print("\nTraining completed successfully!")
    print(f"Final model saved to: {final_model_dir}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    if metrics_callback.validation_loss:
        best_val_loss = min(metrics_callback.validation_loss)
        best_epoch = metrics_callback.epochs[np.argmin(metrics_callback.validation_loss)]
        total_epochs = len(metrics_callback.epochs)
        epoch_range = f"{min(metrics_callback.epochs):.0f}-{max(metrics_callback.epochs):.0f}"
        
        print(f"\nGlobal Training Summary:")
        print(f"  Best validation loss: {best_val_loss:.4f} (epoch {best_epoch:.0f})")
        print(f"  Total epochs trained: {total_epochs} (range: {epoch_range})")
        print(f"  Final model uses globally best checkpoint across all epochs")
        if resume_from_checkpoint:
            print(f"  Training resumed from checkpoint and combined with previous history")

def main():
    parser = argparse.ArgumentParser(description='Train biomedical language model for entity linking')
    parser.add_argument('--model', type=str, required=False, 
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
    
    if not args.model:
        print("Error: --model argument is required when not using --list_models")
        parser.print_help()
        return
    
    # Verify model availability
    if not verify_model_availability(args.model):
        print(f"Model '{args.model}' is not available or verified.")
        return
    
    model_info = BIOMEDICAL_MODELS[args.model]
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print(f"Starting training with {model_info['display_name']}")
    
    train_biomedical_classifier(args, model_info)

if __name__ == "__main__":
    main() 