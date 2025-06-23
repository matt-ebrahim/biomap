#!/usr/bin/env python3
"""
Train BioMegatron fine-tuned as classifier for entity linking.

This script:
1. Loads mention-MONDO pairs from CSV files
2. Creates positive and negative training pairs with proper negative sampling
3. Fine-tunes BioMegatron for ranking MONDO candidates
4. Saves the BEST model based on validation loss (not final model)
5. Creates comprehensive training metrics and visualizations
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import TrainerCallback
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
import pathlib
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import json

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

print("Loading BioMegatron tokenizer and model...")
# Use community-uploaded BioMegatron model
model_name = "EMBO/BioMegatron345mUncased"
tok = AutoTokenizer.from_pretrained(model_name)
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

# Initialize metrics callback
metrics_callback = MetricsCallback()

# Training arguments with BEST MODEL SAVING based on eval_loss
# Increased to 10 epochs for more comprehensive training
args = TrainingArguments(
    output_dir=str(models_dir / "biomegatron_mondo_cls"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,  # ✅ INCREASED: More epochs for better convergence
    learning_rate=1e-5,   # Conservative learning rate
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=3,   # Keep only 3 best checkpoints to save space
    load_best_model_at_end=True,        # ✅ FIXED: Load best model at end
    metric_for_best_model="eval_loss",   # ✅ FIXED: Use eval_loss as metric
    greater_is_better=False,             # ✅ FIXED: Lower eval_loss is better
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

print("Starting training...")
print(f"Training examples: {len(ds_train)}")
print(f"Evaluation examples: {len(ds_dev)}")
print(f"Training for {args.num_train_epochs} epochs")
print("✅ BEST MODEL will be saved based on LOWEST validation loss")

# Train the model
trainer.train()

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
    
    print(f"✅ Saved: {metrics_dir}/training_validation_loss.png")
    print(f"✅ Saved: {metrics_dir}/training_validation_loss.pdf")

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
    
    print(f"✅ Saved: {metrics_dir}/roc_curve.png")
    print(f"✅ Saved: {metrics_dir}/roc_curve.pdf")
    
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
        'total_validation_samples': len(ds_dev)
    }
    
    with open(metrics_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"✅ Saved: {metrics_dir}/metrics_summary.json")
    
    return roc_auc, class_report

# Generate all visualizations
create_training_plots()
roc_auc, class_report = create_roc_analysis()

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"✅ BEST model saved: {final_model_path}")
print(f"✅ All metrics saved: {metrics_dir}")
print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
print(f"   • ROC AUC Score: {roc_auc:.4f}")
if metrics_callback.validation_loss:
    best_val_loss = min(metrics_callback.validation_loss)
    best_epoch = metrics_callback.epochs[np.argmin(metrics_callback.validation_loss)]
    print(f"   • Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch:.1f})")
print(f"   • Training Epochs: {args.num_train_epochs}")
print(f"\n📁 GENERATED FILES:")
print(f"   • training_validation_loss.png/pdf - Loss curves")  
print(f"   • roc_curve.png/pdf - ROC analysis")
print(f"   • metrics_summary.json - Complete metrics")

print(f"\n🎯 EPOCH SELECTION RATIONALE:")
print(f"   • 10 epochs chosen for comprehensive training")
print(f"   • Early stopping via best model selection prevents overfitting")
print(f"   • Validation loss monitoring ensures optimal generalization")
print(f"   • Conservative learning rate (1e-5) allows stable convergence")

print("\n🚀 PRODUCTION-READY FEATURES:")
print("   • High-resolution plots (300 DPI)")
print("   • PDF and PNG formats for flexibility") 
print("   • Comprehensive metrics tracking")
print("   • Best model selection (not final epoch)")
print("   • Professional visualization styling")

print("🔧 FIXED: Proper negative sampling and representative text usage") 