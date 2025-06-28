#!/usr/bin/env python3
"""
Train BioMistral-7B language model fine-tuned as classifier for entity linking.

This script is optimized for large language models (7B parameters) with:
- Model parallelism across multiple GPUs using Accelerate/DeepSpeed
- 8-bit quantization with bitsandbytes
- FP16/BF16 mixed precision training
- Memory-efficient training strategies
- Gradient checkpointing for reduced memory usage
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
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DeepSpeedPlugin
import warnings

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

# BioMistral model configuration
BIOMISTRAL_MODEL = {
    "model_name": "BioMistral/BioMistral-7B",
    "display_name": "BioMistral 7B",
    "parameters": "7b",
    "description": "BioMistral 7B parameters, medical domain LLM based on Mistral",
    "verified": True
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

def setup_accelerator(args):
    """Setup Accelerator with appropriate configuration for large model training."""
    
    # Configure DeepSpeed if requested
    deepspeed_plugin = None
    if args.use_deepspeed:
        deepspeed_config = {
            "zero_optimization": {
                "stage": args.deepspeed_stage,
                "offload_optimizer": {
                    "device": "cpu" if args.deepspeed_stage >= 2 else "none"
                },
                "offload_param": {
                    "device": "cpu" if args.deepspeed_stage >= 3 else "none"
                }
            },
            "fp16": {
                "enabled": args.fp16
            },
            "bf16": {
                "enabled": args.bf16
            },
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": args.batch_size
        }
        
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=1.0,
            zero_stage=args.deepspeed_stage,
            offload_optimizer_device="cpu" if args.deepspeed_stage >= 2 else "none",
            offload_param_device="cpu" if args.deepspeed_stage >= 3 else "none"
        )
        print(f"Using DeepSpeed ZeRO Stage {args.deepspeed_stage}")
    
    # Configure DDP settings
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        broadcast_buffers=False
    )
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if args.fp16 else "bf16" if args.bf16 else "no",
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[ddp_kwargs] if not args.use_deepspeed else None
    )
    
    print(f"Accelerator initialized:")
    print(f"  Device: {accelerator.device}")
    print(f"  Num processes: {accelerator.num_processes}")
    print(f"  Mixed precision: {accelerator.mixed_precision}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    return accelerator

def create_quantization_config(args):
    """Create quantization configuration based on arguments."""
    if not args.use_8bit and not args.use_4bit:
        return None
    
    if args.use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.use_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    
    return None

def load_biomistral_model(args):
    """Load BioMistral model with memory optimizations."""
    print(f"Loading BioMistral-7B model...")
    print(f"Memory optimizations:")
    print(f"  8-bit quantization: {args.use_8bit}")
    print(f"  4-bit quantization: {args.use_4bit}")
    print(f"  FP16: {args.fp16}")
    print(f"  BF16: {args.bf16}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BIOMISTRAL_MODEL['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create quantization config
    quantization_config = create_quantization_config(args)
    
    # Model loading arguments
    model_kwargs = {
        "num_labels": 1,
        "torch_dtype": torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            BIOMISTRAL_MODEL['model_name'],
            **model_kwargs
        )
        
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        actual_param_count = get_model_parameters_count(model)
        actual_param_str = format_parameter_count(actual_param_count)
        
        print(f"Model loaded successfully: {BIOMISTRAL_MODEL['display_name']} ({actual_param_str})")
        
        if quantization_config:
            print(f"Quantization applied: {'4-bit' if args.use_4bit else '8-bit'}")
        
        return model, tokenizer, actual_param_count
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

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

def pair_df(df, negative_samples=3):
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
        sampled_incorrect = np.random.choice(incorrect_mondo_ids, size=min(negative_samples, len(incorrect_mondo_ids)), replace=False)
        
        for incorrect_mondo_id in sampled_incorrect:
            representative_text = mention_mapping[incorrect_mondo_id]
            negative_pairs.append({
                'input': f"{mention} [SEP] {representative_text}",
                'label': 0.0
            })
    
    paired_df = pd.DataFrame(positive_pairs + negative_pairs)
    return paired_df

def tokenize(batch, tokenizer, max_length=128):
    """Tokenize input text for training."""
    return tokenizer(
        batch["input"],
        truncation=True,
        padding=True,
        max_length=max_length
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
    ax1.set_title(f'BioMistral-7B Training Progress (Epochs 1-{max(valid_epochs):.0f})')
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
    plt.savefig(metrics_dir / 'biomistral_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to {metrics_dir / 'biomistral_training_metrics.png'}")
    print(f"Plotted {len(valid_epochs)} epochs of training history (epochs {min(valid_epochs):.0f}-{max(valid_epochs):.0f})")

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
    
    print(f"\nFinding globally best BioMistral model across all {len(metrics_callback.epochs)} epochs...")
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
            print(f"Loading best BioMistral model from: {best_checkpoint_path}")
            # Load the model from the best checkpoint
            best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_path)
            best_tokenizer = AutoTokenizer.from_pretrained(best_checkpoint_path)
            
            # Save to final model directory
            best_model.save_pretrained(str(final_model_dir))
            best_tokenizer.save_pretrained(str(final_model_dir))
            
            print(f"Globally best BioMistral model saved to: {final_model_dir}")
            print(f"Model corresponds to epoch {best_epoch:.0f} with validation loss {best_val_loss:.4f}")
            
        except Exception as e:
            print(f"Error loading best checkpoint: {e}")
            print("Falling back to saving current model")
            trainer.save_model(str(final_model_dir))
    else:
        print("Best checkpoint not found, saving current model")
        trainer.save_model(str(final_model_dir))

def train_biomistral_classifier(args):
    """Main training function for BioMistral-7B classifier."""
    
    # Setup accelerator for distributed/parallel training
    accelerator = setup_accelerator(args)
    
    # Load data
    train = pair_df(pd.read_csv('data/mondo_train.csv'), args.negative_samples)
    dev = pair_df(pd.read_csv('data/mondo_dev.csv'), args.negative_samples)
    
    # Load model and tokenizer
    model, tokenizer, actual_param_count = load_biomistral_model(args)
    
    # Create directories
    models_dir = pathlib.Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_dir_name = f"models_{format_parameter_count(actual_param_count).lower()}_biomistral7b"
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
    ds_train = Dataset.from_pandas(train).map(
        lambda batch: tokenize(batch, tokenizer, args.max_length), 
        batched=True
    )
    ds_dev = Dataset.from_pandas(dev).map(
        lambda batch: tokenize(batch, tokenizer, args.max_length), 
        batched=True
    )
    
    metrics_callback = MetricsCallback(metrics_dir)
    
    # Training arguments optimized for large models
    args_training = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=10,
        load_best_model_at_end=False,  # We implement global best model selection
        report_to=None,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        dataloader_pin_memory=False,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
    )
    
    trainer = Trainer(
        model=model,
        args=args_training,
        train_dataset=ds_train.remove_columns(['input']),
        eval_dataset=ds_dev.remove_columns(['input']),
        processing_class=tokenizer,
        callbacks=[metrics_callback]
    )
    
    # Prepare with accelerator
    trainer.model, trainer.optimizer, trainer.train_dataloader, trainer.eval_dataloader = accelerator.prepare(
        trainer.model, trainer.optimizer, trainer.train_dataloader, trainer.eval_dataloader
    )
    
    if resume_from_checkpoint:
        print(f"Resuming BioMistral training from checkpoint: {resume_from_checkpoint}")
    else:
        print(f"Training BioMistral-7B: {len(ds_train)} examples for {args_training.num_train_epochs} epochs")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    
    # Train the model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Find and save the globally best model across all epochs
    save_globally_best_model(trainer, metrics_callback, checkpoint_dir, final_model_dir)
    
    # Save model metadata
    model_metadata = {
        "model_name": BIOMISTRAL_MODEL['model_name'],
        "display_name": BIOMISTRAL_MODEL['display_name'],
        "description": BIOMISTRAL_MODEL['description'],
        "actual_parameters": actual_param_count,
        "formatted_parameters": format_parameter_count(actual_param_count),
        "training_epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_length": args.max_length,
        "negative_samples": args.negative_samples,
        "resumed_from_checkpoint": resume_from_checkpoint is not None,
        "checkpoint_path": resume_from_checkpoint if resume_from_checkpoint else None,
        "memory_optimizations": {
            "use_8bit": args.use_8bit,
            "use_4bit": args.use_4bit,
            "fp16": args.fp16,
            "bf16": args.bf16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "use_deepspeed": args.use_deepspeed,
            "deepspeed_stage": args.deepspeed_stage if args.use_deepspeed else None
        }
    }
    
    with open(model_base_dir / 'model_info.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Generate metrics and visualizations
    create_training_plots(metrics_callback, metrics_dir)
    
    # Save metrics summary
    metrics_summary = {
        'model_info': model_metadata,
        'best_epoch': float(metrics_callback.epochs[np.argmin(metrics_callback.validation_loss)]) if metrics_callback.validation_loss else None,
        'best_validation_loss': float(min(metrics_callback.validation_loss)) if metrics_callback.validation_loss else None,
        'final_training_loss': float(metrics_callback.training_loss[-1]) if metrics_callback.training_loss and metrics_callback.training_loss[-1] is not None else None,
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
    
    print("\nBioMistral-7B training completed successfully!")
    print(f"Final model saved to: {final_model_dir}")
    
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
    parser = argparse.ArgumentParser(description='Train BioMistral-7B for entity linking with memory optimizations')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Per-device training batch size (default: 1)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32,
                        help='Gradient accumulation steps (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--max_length', type=int, default=128, 
                        help='Maximum sequence length (default: 128)')
    parser.add_argument('--negative_samples', type=int, default=3,
                        help='Number of negative samples per positive (default: 3)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Memory optimization parameters
    parser.add_argument('--use_8bit', action='store_true',
                        help='Use 8-bit quantization')
    parser.add_argument('--use_4bit', action='store_true',
                        help='Use 4-bit quantization (QLoRA)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 mixed precision')
    parser.add_argument('--bf16', action='store_true',
                        help='Use BF16 mixed precision')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    
    # Parallelism parameters
    parser.add_argument('--use_deepspeed', action='store_true',
                        help='Use DeepSpeed for model parallelism')
    parser.add_argument('--deepspeed_stage', type=int, default=2, choices=[1, 2, 3],
                        help='DeepSpeed ZeRO stage (1, 2, or 3)')
    parser.add_argument('--dataloader_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    
    args = parser.parse_args()
    
    # Validation
    if args.use_4bit and args.use_8bit:
        print("Error: Cannot use both 4-bit and 8-bit quantization simultaneously")
        return
    
    if args.fp16 and args.bf16:
        print("Error: Cannot use both FP16 and BF16 simultaneously")
        return
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print(f"Starting BioMistral-7B training with memory optimizations")
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  8-bit quantization: {args.use_8bit}")
    print(f"  4-bit quantization: {args.use_4bit}")
    print(f"  FP16: {args.fp16}")
    print(f"  BF16: {args.bf16}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  DeepSpeed: {args.use_deepspeed}")
    if args.use_deepspeed:
        print(f"  DeepSpeed stage: {args.deepspeed_stage}")
    
    train_biomistral_classifier(args)

if __name__ == "__main__":
    main() 