#!/usr/bin/env python3
"""
SapBERT-based ontology mapping between AzPheWAS phenotypes and MONDO terms.
Adapted from the original FinnGen mapping script.
"""

import pandas as pd
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import logging
import psutil
import gc
import re

from rapidfuzz import fuzz, process
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('azphewas_sapbert_mapping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_azphewas_phenotype(phenotype_text: str) -> str:
    """
    Parse AzPheWAS phenotype text to extract meaningful clinical terms.
    
    Examples:
    - "120000#Ever had osteoarthritis affecting one or more joints e g hip knee shoulder" 
      -> "Ever had osteoarthritis affecting one or more joints e g hip knee shoulder"
    - "130003#Source of report of A01 typhoid and paratyphoid fevers"
      -> "typhoid and paratyphoid fevers"
    - "41202#E871#Hypo-osmolality and hyponatraemia"
      -> "Hypo-osmolality and hyponatraemia"
    - "30600#Albumin"
      -> "Albumin"
    """
    
    # Split by # to analyze structure
    parts = phenotype_text.split('#')
    
    if len(parts) == 1:
        # No # separator, return as is (NMR entries)
        return phenotype_text.strip()
    
    elif len(parts) == 2:
        # Format: CODE#DESCRIPTION
        code, description = parts
        
        # Check if it's a "Source of report" entry
        if description.startswith("Source of report of"):
            # Extract condition name after ICD code
            # Pattern: "Source of report of [ICD_CODE] [CONDITION_NAME]"
            match = re.match(r'Source of report of ([A-Z]\d+(?:\.\d+)?)\s+(.+)', description)
            if match:
                return match.group(2).strip()
            else:
                # Fallback: remove "Source of report of" and any leading ICD-like codes
                clean_desc = description.replace("Source of report of", "").strip()
                # Remove leading ICD codes (like A01, B15, etc.)
                clean_desc = re.sub(r'^[A-Z]\d+(?:\.\d+)?\s+', '', clean_desc)
                return clean_desc.strip()
        else:
            # Regular CODE#DESCRIPTION format
            return description.strip()
    
    elif len(parts) == 3:
        # Format: CODE#ICD_CODE#DESCRIPTION (41202 series)
        return parts[2].strip()
    
    else:
        # More than 3 parts, take the last part
        return parts[-1].strip()

def create_ground_truth_mapping(azphewas_df: pd.DataFrame, 
                               mondo_df: pd.DataFrame,
                               similarity_threshold: float = 95.0,
                               max_matches_per_phenotype: int = 3) -> Dict[str, List[Tuple[str,float]]]:
    """
    Create ground truth mapping using fuzzy string matching.
    
    Args:
        azphewas_df: AzPheWAS phenotypes dataframe
        mondo_df: MONDO terms dataframe  
        similarity_threshold: Minimum similarity score for confident matches (0-100)
        max_matches_per_phenotype: Maximum number of matches to consider per phenotype
    
    Returns:
        Dictionary mapping phenocode to list of (full_id, similarity_score) tuples
    """
    logger.info("="*60)
    logger.info("CREATING GROUND TRUTH USING FUZZY STRING MATCHING")
    logger.info("="*60)
    logger.info(f"Similarity threshold: {similarity_threshold}%")
    logger.info(f"Max matches per phenotype: {max_matches_per_phenotype}")
    
    ground_truth = {}
    mondo_labels = mondo_df['label'].tolist()
    mondo_ids = mondo_df['full_id'].tolist()
    
    # Statistics tracking
    stats = {
        'total_phenotypes': len(azphewas_df),
        'phenotypes_with_matches': 0,
        'total_confident_matches': 0,
        'similarity_scores': []
    }
    
    logger.info(f"Processing {len(azphewas_df):,} AzPheWAS phenotypes against {len(mondo_df):,} MONDO terms...")
    
    for idx, (_, row) in enumerate(tqdm(azphewas_df.iterrows(), 
                                       desc="Fuzzy matching", 
                                       total=len(azphewas_df))):
        
        phenocode = row['phenocode']
        phenotype = row['phenotype']
        
        # Use rapidfuzz to find best matches
        matches = process.extract(
            phenotype, 
            mondo_labels,
            scorer=fuzz.token_sort_ratio,  # Good for handling word order differences
            score_cutoff=similarity_threshold,
            limit=max_matches_per_phenotype
        )
        
        confident_matches = []
        for match_label, score, match_idx in matches:
            if score >= similarity_threshold:
                mondo_id = mondo_ids[match_idx]
                confident_matches.append((mondo_id, score))
                stats['similarity_scores'].append(score)
        
        if confident_matches:
            ground_truth[phenocode] = confident_matches
            stats['phenotypes_with_matches'] += 1
            stats['total_confident_matches'] += len(confident_matches)
        
        # Log progress every 1000 phenotypes
        if (idx + 1) % 1000 == 0:
            progress = (idx + 1) / len(azphewas_df) * 100
            current_matches = stats['phenotypes_with_matches']
            logger.info(f"Progress: {progress:.1f}% - Found matches for {current_matches:,} phenotypes so far")
    
    # Log final statistics
    logger.info("\nGround Truth Creation Results:")
    logger.info(f"  Total AzPheWAS phenotypes: {stats['total_phenotypes']:,}")
    logger.info(f"  Phenotypes with confident matches: {stats['phenotypes_with_matches']:,} "
               f"({100*stats['phenotypes_with_matches']/stats['total_phenotypes']:.1f}%)")
    logger.info(f"  Total confident matches: {stats['total_confident_matches']:,}")
    logger.info(f"  Average matches per matched phenotype: "
               f"{stats['total_confident_matches']/max(1, stats['phenotypes_with_matches']):.1f}")
    
    if stats['similarity_scores']:
        scores = np.array(stats['similarity_scores'])
        logger.info(f"  Similarity score statistics:")
        logger.info(f"    Mean: {scores.mean():.1f}%")
        logger.info(f"    Median: {np.median(scores):.1f}%")
        logger.info(f"    Min: {scores.min():.1f}%")
        logger.info(f"    Max: {scores.max():.1f}%")
        logger.info(f"    Perfect matches (100%): {(scores == 100).sum():,}")
    
    # Show sample matches
    logger.info("\nSample ground truth matches:")
    sample_count = 0
    for phenocode, matches in ground_truth.items():
        if sample_count >= 5:
            break
        
        # Find the phenotype text
        phenotype = azphewas_df[azphewas_df['phenocode'] == phenocode]['phenotype'].iloc[0]
        logger.info(f"\n  AzPheWAS: {phenotype}")
        
        for mondo_id, score in matches[:2]:  # Show top 2 matches
            # Find the MONDO label
            mondo_label = mondo_df[mondo_df['full_id'] == mondo_id]['label'].iloc[0]
            logger.info(f"    → {mondo_id} ({score:.1f}%): {mondo_label}")
        
        sample_count += 1
    
    return ground_truth

def evaluate_sapbert_performance(azphewas_df: pd.DataFrame,
                               sapbert_matches: List[List[Tuple[str, str, float]]],
                               ground_truth: Dict[str, List[Tuple[str, float]]]) -> Dict:
    """
    Evaluate SapBERT performance against ground truth.
    
    Args:
        azphewas_df: AzPheWAS phenotypes dataframe
        sapbert_matches: SapBERT predictions (list of matches per phenotype)
        ground_truth: Ground truth mapping from fuzzy matching
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("="*60)
    logger.info("EVALUATING SAPBERT PERFORMANCE")
    logger.info("="*60)
    
    # Create evaluation data
    y_true = []  # Binary labels: 1 if SapBERT prediction matches ground truth
    y_scores = []  # SapBERT similarity scores
    detailed_results = []
    
    phenocodes = azphewas_df['phenocode'].tolist()
    
    for i, phenocode in enumerate(phenocodes):
        if phenocode not in ground_truth:
            continue  # Skip phenotypes without ground truth
        
        gt_mondo_ids = {mondo_id for mondo_id, _ in ground_truth[phenocode]}
        sapbert_predictions = sapbert_matches[i]
        
        # Check both 1st and 2nd SapBERT predictions
        for rank, (pred_mondo_id, pred_label, pred_score) in enumerate(sapbert_predictions):
            is_correct = pred_mondo_id in gt_mondo_ids
            
            y_true.append(1 if is_correct else 0)
            y_scores.append(pred_score)
            
            detailed_results.append({
                'phenocode': phenocode,
                'rank': rank + 1,
                'predicted_mondo_id': pred_mondo_id,
                'predicted_score': pred_score,
                'is_correct': is_correct,
                'ground_truth_ids': list(gt_mondo_ids)
            })
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    logger.info(f"Evaluation dataset:")
    logger.info(f"  Phenotypes with ground truth: {len(ground_truth):,}")
    logger.info(f"  Total predictions evaluated: {len(y_true):,}")
    logger.info(f"  Correct predictions: {np.sum(y_true):,} ({100*np.mean(y_true):.1f}%)")
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Youden's J statistic (optimal threshold)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    optimal_youden_j = youden_j[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
    
    # Calculate additional metrics
    tp = np.sum((y_true == 1) & (y_pred_optimal == 1))
    tn = np.sum((y_true == 0) & (y_pred_optimal == 0))
    fp = np.sum((y_true == 0) & (y_pred_optimal == 1))
    fn = np.sum((y_true == 1) & (y_pred_optimal == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true)
    
    # Rank-based evaluation
    rank_1_correct = 0
    rank_2_correct = 0
    total_evaluated_phenotypes = len(ground_truth)
    
    for phenocode in ground_truth.keys():
        i = phenocodes.index(phenocode)
        gt_mondo_ids = {mondo_id for mondo_id, _ in ground_truth[phenocode]}
        sapbert_predictions = sapbert_matches[i]
        
        if len(sapbert_predictions) > 0 and sapbert_predictions[0][0] in gt_mondo_ids:
            rank_1_correct += 1
        
        if any(pred[0] in gt_mondo_ids for pred in sapbert_predictions[:2]):
            rank_2_correct += 1
    
    metrics = {
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'youden_j': optimal_youden_j,
        'sensitivity': optimal_sensitivity,
        'specificity': optimal_specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'rank_1_accuracy': rank_1_correct / total_evaluated_phenotypes,
        'rank_2_accuracy': rank_2_correct / total_evaluated_phenotypes,
        'total_predictions': len(y_true),
        'correct_predictions': np.sum(y_true),
        'evaluated_phenotypes': total_evaluated_phenotypes,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'detailed_results': detailed_results
    }
    
    # Log results
    logger.info("\nPerformance Metrics:")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  Youden's J Statistic: {optimal_youden_j:.4f}")
    logger.info(f"  Optimal Threshold: {optimal_threshold:.4f}")
    logger.info(f"  Sensitivity (Recall): {optimal_sensitivity:.4f}")
    logger.info(f"  Specificity: {optimal_specificity:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  F1 Score: {f1_score:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"\nRank-based Metrics:")
    logger.info(f"  Rank-1 Accuracy: {rank_1_correct}/{total_evaluated_phenotypes} ({100*metrics['rank_1_accuracy']:.1f}%)")
    logger.info(f"  Rank-2 Accuracy: {rank_2_correct}/{total_evaluated_phenotypes} ({100*metrics['rank_2_accuracy']:.1f}%)")
    
    return metrics

def plot_roc_curve(metrics: Dict, output_dir: str = "."):
    """Plot ROC curve with Youden's J optimal point."""
    logger.info("Creating ROC curve plot...")
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2, 
             label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
    
    # Plot optimal point (Youden's J)
    optimal_idx = np.argmax(metrics['tpr'] - metrics['fpr'])
    plt.plot(metrics['fpr'][optimal_idx], metrics['tpr'][optimal_idx], 
             'ro', markersize=10, 
             label=f'Optimal Point (J = {metrics["youden_j"]:.4f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('SapBERT Performance: ROC Curve Analysis\n' + 
              f'Optimal Threshold: {metrics["optimal_threshold"]:.4f}, ' +
              f'Sensitivity: {metrics["sensitivity"]:.3f}, ' +
              f'Specificity: {metrics["specificity"]:.3f}', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text box with key metrics
    textstr = f'''Key Metrics:
AUC: {metrics["roc_auc"]:.4f}
Youden's J: {metrics["youden_j"]:.4f}
F1 Score: {metrics["f1_score"]:.4f}
Rank-1 Acc: {100*metrics["rank_1_accuracy"]:.1f}%
Rank-2 Acc: {100*metrics["rank_2_accuracy"]:.1f}%'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.65, 0.25, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'azphewas_mondo_sapbert_roc_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"ROC curve saved to: {plot_path}")
    
    return plot_path

def save_evaluation_results(metrics: Dict, ground_truth: Dict, output_dir: str = "."):
    """Save detailed evaluation results to CSV files."""
    logger.info("Saving evaluation results...")
    
    # Save summary metrics
    summary_df = pd.DataFrame([{
        'metric': 'ROC_AUC',
        'value': metrics['roc_auc'],
        'description': 'Area Under ROC Curve'
    }, {
        'metric': 'Youden_J',
        'value': metrics['youden_j'],
        'description': 'Youden\'s J Statistic (Sensitivity + Specificity - 1)'
    }, {
        'metric': 'Optimal_Threshold',
        'value': metrics['optimal_threshold'],
        'description': 'Optimal similarity threshold for classification'
    }, {
        'metric': 'Sensitivity',
        'value': metrics['sensitivity'],
        'description': 'True Positive Rate at optimal threshold'
    }, {
        'metric': 'Specificity',
        'value': metrics['specificity'],
        'description': 'True Negative Rate at optimal threshold'
    }, {
        'metric': 'Precision',
        'value': metrics['precision'],
        'description': 'Precision at optimal threshold'
    }, {
        'metric': 'F1_Score',
        'value': metrics['f1_score'],
        'description': 'F1 Score at optimal threshold'
    }, {
        'metric': 'Rank_1_Accuracy',
        'value': metrics['rank_1_accuracy'],
        'description': 'Percentage of ground truth matches found in rank 1'
    }, {
        'metric': 'Rank_2_Accuracy',
        'value': metrics['rank_2_accuracy'],
        'description': 'Percentage of ground truth matches found in top 2 ranks'
    }])
    
    summary_path = Path(output_dir) / 'azphewas_mondo_evaluation_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary metrics saved to: {summary_path}")
    
    # Save detailed results
    detailed_df = pd.DataFrame(metrics['detailed_results'])
    detailed_path = Path(output_dir) / 'azphewas_mondo_evaluation_detailed.csv'
    detailed_df.to_csv(detailed_path, index=False)
    logger.info(f"Detailed results saved to: {detailed_path}")
    
    # Save ground truth
    gt_rows = []
    for phenocode, matches in ground_truth.items():
        for mondo_id, similarity_score in matches:
            gt_rows.append({
                'phenocode': phenocode,
                'ground_truth_mondo_id': mondo_id,
                'fuzzy_similarity_score': similarity_score
            })
    
    gt_df = pd.DataFrame(gt_rows)
    gt_path = Path(output_dir) / 'azphewas_mondo_ground_truth_fuzzy_matches.csv'
    gt_df.to_csv(gt_path, index=False)
    logger.info(f"Ground truth matches saved to: {gt_path}")
    
    return summary_path, detailed_path, gt_path

def log_memory_usage(stage: str):
    """Log current memory usage statistics optimized for CUDA."""
    memory = psutil.virtual_memory()
    process = psutil.Process()
    
    logger.info(f"[{stage}] Memory Usage:")
    logger.info(f"  System Memory: {memory.percent:.1f}% used ({memory.used/1e9:.2f}GB / {memory.total/1e9:.2f}GB)")
    logger.info(f"  Process Memory: {process.memory_info().rss/1e9:.2f}GB RSS, {process.memory_percent():.2f}% of system")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            gpu_allocated = torch.cuda.memory_allocated(i)
            gpu_cached = torch.cuda.memory_reserved(i)
            gpu_free = gpu_memory - gpu_cached
            logger.info(f"  GPU {i} ({torch.cuda.get_device_name(i)}): "
                       f"{gpu_allocated/1e9:.2f}GB allocated, {gpu_cached/1e9:.2f}GB cached, "
                       f"{gpu_free/1e9:.2f}GB free, {gpu_memory/1e9:.2f}GB total")

def load_azphewas_data(file_path: str) -> pd.DataFrame:
    """Load AzPheWAS phenotype data with proper parsing."""
    logger.info(f"Loading AzPheWAS data from: {file_path}")
    
    if not Path(file_path).exists():
        logger.error(f"AzPheWAS file not found: {file_path}")
        raise FileNotFoundError(f"AzPheWAS file not found: {file_path}")
    
    file_size = Path(file_path).stat().st_size / 1e6
    logger.info(f"File size: {file_size:.1f}MB")
    
    load_start = time.time()
    df = pd.read_csv(file_path)
    load_time = time.time() - load_start
    
    logger.info(f"Loaded {len(df):,} AzPheWAS phenotypes in {load_time:.2f}s")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Parse phenotypes
    logger.info("Parsing phenotype descriptions...")
    parse_start = time.time()
    
    df['phenocode'] = df['Phenotype'].apply(lambda x: x.split('#')[0] if '#' in x else x[:10])
    df['parsed_phenotype'] = df['Phenotype'].apply(parse_azphewas_phenotype)
    
    # Filter out empty or very short phenotypes
    df = df[df['parsed_phenotype'].str.len() > 3].copy()
    
    parse_time = time.time() - parse_start
    logger.info(f"Phenotype parsing completed in {parse_time:.2f}s")
    
    # Show sample parsed phenotypes
    logger.info("Sample parsed phenotypes:")
    for i in range(min(10, len(df))):
        original = df.iloc[i]['Phenotype']
        parsed = df.iloc[i]['parsed_phenotype']
        logger.info(f"  Original: {original}")
        logger.info(f"  Parsed:   {parsed}")
        logger.info("")
    
    result_df = df[['phenocode', 'parsed_phenotype']].copy()
    result_df.columns = ['phenocode', 'phenotype']
    
    logger.info(f"Final AzPheWAS dataset: {len(result_df):,} phenotypes")
    return result_df

def load_mondo_data(file_path: str) -> pd.DataFrame:
    """Load MONDO terms data."""
    logger.info(f"Loading MONDO data from: {file_path}")
    
    if not Path(file_path).exists():
        logger.error(f"MONDO file not found: {file_path}")
        raise FileNotFoundError(f"MONDO file not found: {file_path}")
    
    file_size = Path(file_path).stat().st_size / 1e6
    logger.info(f"File size: {file_size:.1f}MB")
    
    load_start = time.time()
    df = pd.read_csv(file_path)
    load_time = time.time() - load_start
    
    logger.info(f"Loaded {len(df):,} MONDO terms in {load_time:.2f}s")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Remove obsolete terms and clean data
    if 'is_obsolete' in df.columns:
        df = df[df['is_obsolete'] == False].copy()
        logger.info("Filtered out obsolete terms")
    
    required_cols = ['mondo_id', 'full_id', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Required columns not found: {missing_cols}")
    
    result_df = df[required_cols].copy().dropna()
    result_df['label'] = result_df['label'].astype(str).str.replace('\n', ' ').str.strip()
    
    # Show sample data
    logger.info("Sample MONDO terms:")
    for i in range(min(5, len(result_df))):
        row = result_df.iloc[i]
        label_preview = row['label'][:80] + "..." if len(row['label']) > 80 else row['label']
        logger.info(f"  {row['full_id']}: {label_preview}")
    
    logger.info(f"Final MONDO dataset: {len(result_df):,} terms")
    return result_df

def setup_device():
    """Setup computing device optimized for CUDA."""
    logger.info("Setting up computing device...")
    
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            gpu_compute = torch.cuda.get_device_properties(i).major
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB, Compute {gpu_compute}.x)")
        
        # Set the primary GPU
        torch.cuda.set_device(0)
        logger.info(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA optimizations enabled")
        
    else:
        device = 'cpu'
        cpu_count = psutil.cpu_count()
        logger.info(f"CUDA not available - using CPU with {cpu_count} cores")
        logger.info(f"PyTorch version: {torch.__version__}")
    
    log_memory_usage("Device Setup")
    return device

def load_model(device='cuda'):
    """Load SapBERT model and tokenizer optimized for CUDA."""
    logger.info("Loading SapBERT model...")
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    logger.info(f"Model: {model_name}")
    
    start_time = time.time()
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_load_time = time.time() - start_time
    logger.info(f"Tokenizer loaded in {tokenizer_load_time:.2f}s")
    
    logger.info("Loading model weights...")
    model_start = time.time()
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        low_cpu_mem_usage=True,
        device_map='auto' if device == 'cuda' else None
    ).eval()
    
    if device == 'cuda':
        model = model.to(device)
        # Enable optimizations for inference
        model = torch.compile(model, mode='reduce-overhead') if hasattr(torch, 'compile') else model
        logger.info("Model compiled for optimized inference")
    
    model_load_time = time.time() - model_start
    total_load_time = time.time() - start_time
    
    logger.info(f"Model loaded in {model_load_time:.2f}s")
    logger.info(f"Total model setup time: {total_load_time:.2f}s")
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    log_memory_usage("Model Loading")
    return tokenizer, model

def encode_texts(texts, tokenizer, model, device, batch_size=64, max_length=64):
    """Encode texts to embeddings using SapBERT optimized for CUDA GPU."""
    logger.info(f"Starting text encoding (CUDA optimized)...")
    logger.info(f"Texts to encode: {len(texts):,}")
    logger.info(f"Batch size: {batch_size} (optimized for T4 GPU)")
    logger.info(f"Max length: {max_length}")
    logger.info(f"Device: {device}")
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    start_time = time.time()
    encoding_times = []
    
    # Sample a few texts to show what we're encoding
    logger.info("Sample texts to encode:")
    for i in range(min(3, len(texts))):
        sample_text = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
        logger.info(f"  [{i+1}] {sample_text}")
    
    logger.info(f"Processing {total_batches} batches...")
    
    # Use mixed precision for better performance on T4
    if device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using mixed precision for faster inference")
    
    for batch_idx in tqdm(range(0, len(texts), batch_size), 
                         desc="Encoding", 
                         total=total_batches,
                         unit="batch"):
        
        batch_start = time.time()
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        
        # Log detailed progress every 100 batches (less frequent for speed)
        if batch_idx // batch_size > 0 and (batch_idx // batch_size) % 100 == 0:
            progress = (batch_idx / len(texts)) * 100
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / (batch_idx // batch_size)
            remaining_batches = total_batches - (batch_idx // batch_size)
            eta = remaining_batches * avg_time_per_batch
            
            logger.info(f"Batch {batch_idx//batch_size}/{total_batches} ({progress:.1f}%) - "
                       f"Avg: {avg_time_per_batch:.3f}s/batch - ETA: {eta:.1f}s")
            log_memory_usage(f"Encoding Batch {batch_idx//batch_size}")
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt",
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(device)
        
        # Get embeddings with mixed precision if CUDA
        with torch.no_grad():
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    embeddings = model(**inputs).pooler_output
            else:
                embeddings = model(**inputs).pooler_output
        
        # Move to CPU and store
        embeddings_cpu = embeddings.cpu().numpy().astype(np.float32)
        all_embeddings.append(embeddings_cpu)
        
        batch_total_time = time.time() - batch_start
        encoding_times.append(batch_total_time)
        
        # Log detailed timing for first few batches
        if batch_idx // batch_size < 3:
            logger.info(f"Batch {batch_idx//batch_size + 1} timing: {batch_total_time:.3f}s")
            logger.info(f"  Input shape: {inputs['input_ids'].shape}")
            logger.info(f"  Output shape: {embeddings.shape}")
        
        # Clear GPU cache every 50 batches for optimal memory usage
        if device == 'cuda' and (batch_idx // batch_size) % 50 == 0:
            torch.cuda.empty_cache()
        
        # Less frequent garbage collection for better performance
        if (batch_idx // batch_size) % 200 == 0:
            gc.collect()
    
    # Combine all embeddings efficiently
    logger.info("Combining embeddings...")
    combine_start = time.time()
    embeddings = np.vstack(all_embeddings)
    combine_time = time.time() - combine_start
    
    # Fast L2 normalization for cosine similarity
    logger.info("Normalizing embeddings for cosine similarity...")
    normalize_start = time.time()
    
    # Ensure optimal format for normalization
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    # Fast NumPy L2 normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    normalize_time = time.time() - normalize_start
    total_time = time.time() - start_time
    
    # Log comprehensive timing statistics
    logger.info("Encoding completed!")
    logger.info(f"Total encoding time: {total_time:.2f}s")
    logger.info(f"Average time per batch: {np.mean(encoding_times):.3f}s")
    logger.info(f"Texts per second: {len(texts)/total_time:.1f}")
    logger.info(f"Combine time: {combine_time:.3f}s")
    logger.info(f"Normalize time: {normalize_time:.3f}s")
    logger.info(f"Final embeddings shape: {embeddings.shape}")
    logger.info(f"Memory usage: {embeddings.nbytes / 1e6:.1f}MB")
    
    log_memory_usage("Encoding Complete")
    return embeddings

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    """Build FAISS index optimized for CUDA GPU."""
    logger.info("Building FAISS index (CUDA optimized)...")
    logger.info(f"Input embeddings shape: {embeddings.shape}")
    logger.info(f"Input embeddings dtype: {embeddings.dtype}")
    
    start_time = time.time()
    embedding_dim = embeddings.shape[1]
    num_vectors = embeddings.shape[0]
    
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Number of vectors: {num_vectors:,}")
    
    # Create CPU index first
    logger.info("Creating IndexFlatIP for cosine similarity...")
    cpu_index = faiss.IndexFlatIP(embedding_dim)
    
    # Try to use GPU if available and requested
    if use_gpu and torch.cuda.is_available():
        try:
            logger.info("Attempting to create GPU index for faster search...")
            
            # Create GPU resources
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            
            logger.info("GPU index created successfully")
            index = gpu_index
            index_type = "GPU"
            
        except Exception as e:
            logger.warning(f"GPU index creation failed ({e}), falling back to CPU")
            index = cpu_index
            index_type = "CPU"
    else:
        index = cpu_index
        index_type = "CPU"
    
    # Add embeddings to index
    logger.info(f"Adding {num_vectors:,} embeddings to {index_type} index...")
    
    # Ensure embeddings are contiguous and float32
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    add_start = time.time()
    index.add(embeddings)
    add_time = time.time() - add_start
    
    total_time = time.time() - start_time
    
    logger.info(f"FAISS {index_type} index built successfully!")
    logger.info(f"  Index entries: {index.ntotal:,}")
    logger.info(f"  Add time: {add_time:.3f}s")
    logger.info(f"  Total build time: {total_time:.3f}s")
    logger.info(f"  Vectors per second: {num_vectors/add_time:.0f}")
    
    # Test the index
    logger.info("Testing index with sample query...")
    test_start = time.time()
    scores, indices = index.search(embeddings[:1], 5)
    test_time = time.time() - test_start
    
    logger.info(f"Test search completed in {test_time:.3f}s")
    logger.info(f"Sample search results (similarity scores): {scores[0]}")
    
    log_memory_usage("FAISS Index Built")
    return index

def find_top_matches(query_embeddings: np.ndarray, 
                    index: faiss.Index, 
                    mondo_df: pd.DataFrame,
                    top_k: int = 2) -> List[List[Tuple[str, str, float]]]:
    """Find top-k matches using CUDA-optimized FAISS index."""
    logger.info(f"Finding top-{top_k} matches for {len(query_embeddings):,} queries...")
    logger.info(f"Query embeddings shape: {query_embeddings.shape}")
    logger.info(f"Index contains {index.ntotal:,} vectors")
    
    search_start = time.time()
    
    # Ensure query embeddings are optimal format
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    
    logger.info(f"Performing optimized similarity search...")
    
    # Single batch search (no chunking needed for CUDA)
    scores, indices = index.search(query_embeddings, top_k)
    search_time = time.time() - search_start
    
    logger.info(f"Search completed in {search_time:.3f}s")
    logger.info(f"Search rate: {len(query_embeddings)/search_time:.0f} queries/second")
    
    # Analyze results
    valid_matches = (indices != -1).sum()
    total_possible = len(query_embeddings) * top_k
    logger.info(f"Valid matches: {valid_matches:,} / {total_possible:,} ({100*valid_matches/total_possible:.1f}%)")
    
    # Format results using full_id instead of mondo_id
    logger.info("Formatting results (using full MONDO IDs)...")
    format_start = time.time()
    results = []
    
    for i in range(len(query_embeddings)):
        query_results = []
        for j in range(top_k):
            if indices[i][j] != -1:
                mondo_idx = indices[i][j]
                mondo_id = mondo_df.iloc[mondo_idx]['full_id']  # Using full_id instead of mondo_id
                mondo_label = mondo_df.iloc[mondo_idx]['label']
                similarity_score = float(scores[i][j])
                query_results.append((str(mondo_id), mondo_label, similarity_score))
        results.append(query_results)
    
    format_time = time.time() - format_start
    logger.info(f"Results formatted in {format_time:.3f}s")
    
    # Show sample results to verify full_id format
    logger.info("Sample matching results (showing full MONDO IDs):")
    for i in range(min(3, len(results))):
        logger.info(f"  Query {i+1} results:")
        for j, (mondo_id, mondo_label, score) in enumerate(results[i]):
            label_preview = mondo_label[:60] + "..." if len(mondo_label) > 60 else mondo_label
            logger.info(f"    {j+1}. {mondo_id} (score: {score:.4f}): {label_preview}")
    
    return results

def create_output_dataframe(azphewas_df: pd.DataFrame, 
                          matches: List[List[Tuple[str, str, float]]]) -> pd.DataFrame:
    """Create output dataframe with AzPheWAS phenotypes and their MONDO matches."""
    logger.info("Creating output dataframe...")
    
    output_data = []
    for i, (_, row) in enumerate(azphewas_df.iterrows()):
        phenocode = row['phenocode']
        phenotype = row['phenotype']
        phenotype_matches = matches[i]
        
        output_row = {
            'phenocode': phenocode,
            'phenotype': phenotype,
            'mondo_id_1st': "",
            'mondo_label_1st': "",
            'mondo_score_1st': 0.0,
            'mondo_id_2nd': "",
            'mondo_label_2nd': "",
            'mondo_score_2nd': 0.0
        }
        
        if len(phenotype_matches) > 0:
            output_row['mondo_id_1st'] = phenotype_matches[0][0]
            output_row['mondo_label_1st'] = phenotype_matches[0][1]
            output_row['mondo_score_1st'] = phenotype_matches[0][2]
        
        if len(phenotype_matches) > 1:
            output_row['mondo_id_2nd'] = phenotype_matches[1][0]
            output_row['mondo_label_2nd'] = phenotype_matches[1][1]
            output_row['mondo_score_2nd'] = phenotype_matches[1][2]
        
        output_data.append(output_row)
    
    return pd.DataFrame(output_data)

def main():
    parser = argparse.ArgumentParser(description="Map AzPheWAS phenotypes to MONDO terms using SapBERT (CUDA optimized)")
    parser.add_argument("--azphewas", default="AzPheWAS_UniquePhenotypes.csv",
                       help="AzPheWAS phenotypes CSV file")
    parser.add_argument("--mondo", default="mondo_terms_human.csv", 
                       help="MONDO terms CSV file")
    parser.add_argument("--output", default="azphewas_mondo_mapping.csv",
                       help="Output CSV file")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for encoding (optimized for T4 GPU)")
    parser.add_argument("--max-length", type=int, default=64,
                       help="Maximum sequence length")
    parser.add_argument("--no-gpu-index", action='store_true',
                       help="Force CPU-only FAISS index")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("SapBERT AzPheWAS-MONDO Mapping (CUDA Optimized)")
    logger.info("="*80)
    logger.info(f"Arguments: {vars(args)}")
    
    overall_start = time.time()
    log_memory_usage("Script Start")
    
    try:
        # Setup device and model
        device = setup_device()
        tokenizer, model = load_model(device)
        
        # Load data
        azphewas_df = load_azphewas_data(args.azphewas)
        mondo_df = load_mondo_data(args.mondo)
        
        # Prepare texts
        azphewas_texts = azphewas_df['phenotype'].tolist()
        mondo_texts = mondo_df['label'].tolist()
        
        logger.info(f"Dataset summary: {len(azphewas_texts):,} AzPheWAS + {len(mondo_texts):,} MONDO")
        
        # Encode MONDO terms and build index
        logger.info("\n" + "="*50)
        logger.info("ENCODING MONDO TERMS")
        logger.info("="*50)
        
        mondo_embeddings = encode_texts(
            mondo_texts, tokenizer, model, device, 
            batch_size=args.batch_size, max_length=args.max_length
        )
        
        logger.info("\n" + "="*50)
        logger.info("BUILDING SEARCH INDEX")
        logger.info("="*50)
        
        mondo_index = build_faiss_index(mondo_embeddings, use_gpu=not args.no_gpu_index)
        
        # Encode AzPheWAS phenotypes
        logger.info("\n" + "="*50)
        logger.info("ENCODING AZPHEWAS PHENOTYPES")
        logger.info("="*50)
        
        azphewas_embeddings = encode_texts(
            azphewas_texts, tokenizer, model, device,
            batch_size=args.batch_size, max_length=args.max_length
        )
        
        # Find matches
        logger.info("\n" + "="*50)
        logger.info("FINDING SEMANTIC MATCHES")
        logger.info("="*50)
        
        matches = find_top_matches(azphewas_embeddings, mondo_index, mondo_df, top_k=2)
        
        # Create and save output
        logger.info("\n" + "="*50)
        logger.info("CREATING OUTPUT")
        logger.info("="*50)
        
        output_df = create_output_dataframe(azphewas_df, matches)
        
        logger.info(f"Saving results to: {args.output}")
        output_df.to_csv(args.output, index=False)
        
        # ADD EVALUATION SECTION HERE
        logger.info("\n" + "="*50)
        logger.info("QUANTITATIVE EVALUATION")
        logger.info("="*50)
        
        # Create ground truth using fuzzy matching
        ground_truth = create_ground_truth_mapping(
            azphewas_df, 
            mondo_df,
            similarity_threshold=95.0,  # Adjust as needed
            max_matches_per_phenotype=3
        )
        
        # Evaluate SapBERT performance
        if ground_truth:
            metrics = evaluate_sapbert_performance(azphewas_df, matches, ground_truth)
            
            # Create ROC plot
            plot_path = plot_roc_curve(metrics, output_dir=".")
            
            # Save results
            summary_path, detailed_path, gt_path = save_evaluation_results(
                metrics, ground_truth, output_dir="."
            )
            
            logger.info(f"\nEvaluation files created:")
            logger.info(f"  ROC plot: {plot_path}")
            logger.info(f"  Summary metrics: {summary_path}")
            logger.info(f"  Detailed results: {detailed_path}")
            logger.info(f"  Ground truth: {gt_path}")
        else:
            logger.warning("No ground truth matches found - skipping evaluation")
        
        file_size = Path(args.output).stat().st_size / 1e6
        total_time = time.time() - overall_start
        
        logger.info("\n" + "="*80)
        logger.info("MAPPING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Processed: {len(azphewas_df):,} AzPheWAS phenotypes")
        logger.info(f"Output: {args.output} ({file_size:.1f}MB)")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"Rate: {len(azphewas_df)/(total_time/60):.1f} phenotypes/minute")
        
        log_memory_usage("Final")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.error("Full traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    main() 