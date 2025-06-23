#!/usr/bin/env python3
"""
Train BioMegatron fine-tuned as classifier for entity linking.

This script:
1. Loads mention-MONDO pairs from CSV files
2. Creates positive and negative training pairs with proper negative sampling
3. Fine-tunes BioMegatron for ranking MONDO candidates
4. Saves the BEST model based on validation loss (not final model)
5. Creates comprehensive training metrics and visualizations
6. Supports checkpoint resumption for continued training
"""

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

# Create directories
models_dir = pathlib.Path('models')
models_dir.mkdir(exist_ok=True)
metrics_dir = models_dir / "model-metrics"
metrics_dir.mkdir(exist_ok=True)

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

# Training configuration
TARGET_EPOCHS = 10
CHECKPOINT_DIR = str(models_dir / "biomegatron_mondo_cls")

# Check for existing checkpoints
latest_checkpoint, checkpoint_epoch = find_latest_checkpoint(CHECKPOINT_DIR)
resume_from_checkpoint = None

if latest_checkpoint and checkpoint_epoch < TARGET_EPOCHS:
    if ask_resume_training(latest_checkpoint, checkpoint_epoch, TARGET_EPOCHS):
        resume_from_checkpoint = latest_checkpoint
        print(f"Will resume training from {latest_checkpoint}")
    else:
        print("Starting training from scratch")
elif latest_checkpoint and checkpoint_epoch >= TARGET_EPOCHS:
    print(f"Training already completed ({checkpoint_epoch} epochs >= {TARGET_EPOCHS} target epochs)")
    print("If you want to train more epochs, increase TARGET_EPOCHS in the script")
    exit(0)

print("Loading BioMegatron tokenizer and model...")
# Use community-uploaded BioMegatron model
model_name = "EMBO/BioMegatron345mUncased"
tok = AutoTokenizer.from_pretrained(model_name)

# Load model from checkpoint if resuming, otherwise from pretrained
if resume_from_checkpoint:
    print(f"Loading model from checkpoint: {resume_from_checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(resume_from_checkpoint)
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1  # regression score for ranking
    )

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

# Initialize metrics callback
metrics_callback = MetricsCallback()

# Training arguments with BEST MODEL SAVING based on eval_loss
args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=TARGET_EPOCHS,
    learning_rate=1e-5,   # Conservative learning rate
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=3,   # Keep only 3 best checkpoints to save space
    load_best_model_at_end=True,        # Load best model at end
    metric_for_best_model="eval_loss",   # Use eval_loss as metric
    greater_is_better=False,             # Lower eval_loss is better
    report_to=None,
    warmup_steps=100,
    weight_decay=0.01,
    dataloader_pin_memory=False,  # Disable pin_memory for MPS compatibility
)

print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train.remove_columns(['input']),
    eval_dataset=ds_dev.remove_columns(['input']),
    tokenizer=tok,
    callbacks=[metrics_callback]  # Add metrics tracking
)

if resume_from_checkpoint:
    print(f"Starting training from checkpoint: {resume_from_checkpoint}")
    print(f"Resuming from epoch {checkpoint_epoch + 1} to {TARGET_EPOCHS}")
else:
    print("Starting training from scratch...")
    print(f"Training examples: {len(ds_train)}")
    print(f"Evaluation examples: {len(ds_dev)}")
    print(f"Training for {args.num_train_epochs} epochs")

print("Best model will be saved based on lowest validation loss")

# Train the model
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# Save the final model (which is actually the BEST model due to load_best_model_at_end=True)
final_model_path = models_dir / "biomegatron_mondo_cls_final"
print(f"Saving BEST model (lowest eval_loss) to {final_model_path}")
trainer.save_model(str(final_model_path))

# Generate comprehensive metrics and visualizations
print("\n" + "="*60)
print("GENERATING PRODUCTION-QUALITY METRICS & VISUALIZATIONS")
print("="*60)

def create_training_plots():
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

def create_roc_analysis():
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
    plt.title('ROC Curve - BioMegatron Entity Linking Classifier', fontweight='bold', fontsize=16)
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
    
    # Save metrics to JSON
    metrics_summary = {
        'roc_auc': float(roc_auc),
        'best_epoch': float(metrics_callback.epochs[np.argmin(metrics_callback.validation_loss)]) if metrics_callback.validation_loss else None,
        'best_validation_loss': float(min(metrics_callback.validation_loss)) if metrics_callback.validation_loss else None,
        'final_training_loss': float(metrics_callback.training_loss[-1]) if metrics_callback.training_loss else None,
        'classification_report': class_report,
        'training_epochs': args.num_train_epochs,
        'total_training_samples': len(ds_train),
        'total_validation_samples': len(ds_dev),
        'resumed_from_checkpoint': resume_from_checkpoint is not None,
        'checkpoint_path': resume_from_checkpoint if resume_from_checkpoint else None
    }
    
    with open(metrics_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"Saved: {metrics_dir}/metrics_summary.json")
    
    return roc_auc, class_report

# Generate all visualizations
create_training_plots()
roc_auc, class_report = create_roc_analysis()

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Best model saved: {final_model_path}")
print(f"All metrics saved: {metrics_dir}")

# Get final metrics from classification report
final_metrics = {
    'roc_auc': roc_auc,
    'precision': class_report['weighted avg']['precision'],
    'recall': class_report['weighted avg']['recall'],
    'f1_score': class_report['weighted avg']['f1-score']
}

print(f"\nMODEL PERFORMANCE SUMMARY:")
for metric, value in final_metrics.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")

print(f"  Best Epoch: {trainer.state.best_model_checkpoint}")
print(f"  Final Validation Loss: {trainer.state.best_metric:.4f}")

if resume_from_checkpoint:
    print(f"  Resumed from: {resume_from_checkpoint}")

print("\nPRODUCTION-READY FEATURES:")
print("- Comprehensive metrics tracking and visualization")
print("- Best model selection based on validation loss")
print("- ROC curve analysis and performance metrics")
print("- Detailed training logs and checkpoints")
print("- High-resolution visualizations (300 DPI)")
print("- JSON metrics export for further analysis")
print("- Proper negative sampling and representative text usage")
print("- Checkpoint resumption for continued training") 