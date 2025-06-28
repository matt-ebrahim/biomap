#!/usr/bin/env python3
"""
Simplified evaluation script for SapBERT and BioMegatron entity linking baselines.

This script evaluates and compares:
1. SapBERT + FAISS similarity search
2. BioMegatron fine-tuned classifier  

Provides Hits@K and MRR metrics for comparison.
"""

import pandas as pd
import numpy as np
import torch
import json
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import argparse

# Import our custom modules
try:
    from inference_sapbert import SapBERTEntityLinker
    from inference_biomegatron import BiomedicalEntityLinker, find_available_models
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure inference_sapbert.py and inference_biomegatron.py are in the same directory")
    exit(1)

def evaluate_hits_and_mrr(ground_truth: List[str], 
                         ranked_lists: List[List[str]], 
                         ks: Tuple[int, ...] = (1, 3, 5, 10)) -> Dict[str, float]:
    """
    Calculate Hits@K and MRR metrics.
    
    Args:
        ground_truth: List of correct MONDO IDs
        ranked_lists: List of ranked candidate lists for each query
        ks: K values for Hits@K calculation
        
    Returns:
        Dictionary with Hits@K and MRR metrics
    """
    hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    total = len(ground_truth)
    
    for gt, ranked in zip(ground_truth, ranked_lists):
        # Calculate Hits@K
        for k in ks:
            if gt in ranked[:k]:
                hits[k] += 1
        
        # Calculate MRR
        if gt in ranked:
            rank_position = ranked.index(gt) + 1  # 1-indexed
            mrr_sum += 1.0 / rank_position
    
    # Convert to percentages and averages
    metrics = {}
    for k in ks:
        metrics[f"Hits@{k}"] = hits[k] / total
    metrics["MRR"] = mrr_sum / total
    
    return metrics

def evaluate_sapbert(test_df: pd.DataFrame, model_dir: str = "models") -> Dict[str, float]:
    """Evaluate SapBERT + FAISS approach."""
    print("\n" + "="*50)
    print("Evaluating SapBERT + FAISS")
    print("="*50)
    
    try:
        linker = SapBERTEntityLinker(model_dir)
        
        ranked_lists = []
        for mention in tqdm(test_df['mention'], desc="SapBERT inference"):
            results = linker.search(mention, top_k=10)
            ranked_list = [mondo_id for mondo_id, score in results]
            ranked_lists.append(ranked_list)
        
        metrics = evaluate_hits_and_mrr(test_df['mondo_id'].tolist(), ranked_lists)
        
        print("SapBERT Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"SapBERT evaluation failed: {e}")
        return {}

def evaluate_biomedical_model(test_df: pd.DataFrame, model_dir: str = "models", 
                             selected_model: str = None) -> Dict[str, float]:
    """Evaluate biomedical classifier approach with optimized batch processing."""
    print("\n" + "="*50)
    print("Evaluating Biomedical Classifier")
    print("="*50)
    
    try:
        # Find available models
        available_models = find_available_models()
        
        if not available_models:
            print("No trained biomedical models found!")
            return {}
        
        # Select model
        if selected_model and selected_model in available_models:
            model_key = selected_model
        else:
            # Use the first available model
            model_key = list(available_models.keys())[0]
            print(f"Auto-selecting model: {model_key}")
        
        model_data = available_models[model_key]
        model_path = model_data["path"]
        
        print(f"Using model: {model_data['display_name']} ({model_data['parameters']})")
        
        # Use larger batch size for better GPU utilization
        linker = BiomedicalEntityLinker(model_path, batch_size=128)
        
        ranked_lists = []
        mentions = test_df['mention'].tolist()
        
        print(f"Processing {len(mentions)} mentions...")
        for mention in tqdm(mentions, desc="Biomedical model inference"):
            results = linker.predict(mention, top_k=10)
            ranked_list = [mondo_id for mondo_id, score in results]
            ranked_lists.append(ranked_list)
        
        metrics = evaluate_hits_and_mrr(test_df['mondo_id'].tolist(), ranked_lists)
        
        print(f"Biomedical Model Results ({model_data['display_name']}):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics, model_data
        
    except Exception as e:
        print(f"Biomedical model evaluation failed: {e}")
        return {}, {}

def save_results(results: Dict[str, Dict[str, float]], output_file: str):
    """Save evaluation results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted comparison table of all results."""
    print("\n" + "="*80)
    print("SAPBERT vs BIOMEGATRON EVALUATION RESULTS")
    print("="*80)
    
    if not results:
        print("No results to display")
        return
    
    # Get all metrics
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    all_metrics = sorted(all_metrics)
    
    # Print header
    print(f"{'Model':<30}", end="")
    for metric in all_metrics:
        print(f"{metric:>12}", end="")
    print()
    print("-" * 80)
    
    # Print results for each model
    for model_name, metrics in results.items():
        print(f"{model_name:<30}", end="")
        for metric in all_metrics:
            value = metrics.get(metric, 0.0)
            print(f"{value:>12.4f}", end="")
        print()
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="SapBERT vs BioMegatron Entity Linking Evaluation")
    parser.add_argument("--test", default="data/mondo_test.csv", help="Test file")
    parser.add_argument("--models", default="models", help="Models directory")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--max_samples", type=int, help="Limit test samples")
    parser.add_argument("--skip_sapbert", action="store_true", help="Skip SapBERT evaluation")
    parser.add_argument("--skip_biomedical", action="store_true", help="Skip biomedical model evaluation")
    parser.add_argument("--biomedical_model", type=str, help="Specific biomedical model to use (e.g., models_345m_biomegatron345muncased)")
    parser.add_argument("--list_models", action="store_true", help="List available biomedical models and exit")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        available_models = find_available_models()
        if available_models:
            print("Available biomedical models:")
            for key, data in available_models.items():
                print(f"  {key}: {data['display_name']} ({data['parameters']})")
        else:
            print("No trained biomedical models found!")
        return
    
    # Load test data
    if not Path(args.test).exists():
        print(f"Error: Test file {args.test} not found")
        return
    
    print(f"Loading test data from {args.test}")
    test_df = pd.read_csv(args.test)
    
    if args.max_samples:
        test_df = test_df.head(args.max_samples)
        print(f"Limited to {args.max_samples} test samples")
    
    print(f"Evaluating on {len(test_df)} test examples")
    
    # Run evaluations
    results = {}
    
    # 1. SapBERT + FAISS
    if not args.skip_sapbert:
        sapbert_results = evaluate_sapbert(test_df, args.models)
        if sapbert_results:
            results["SapBERT + FAISS"] = sapbert_results
    
    # 2. Biomedical Classifier
    if not args.skip_biomedical:
        biomedical_results, model_data = evaluate_biomedical_model(test_df, args.models, args.biomedical_model)
        if biomedical_results:
            # Use model-specific name for results
            if model_data:
                model_name = f"{model_data['display_name']} ({model_data['parameters']})"
            else:
                model_name = "Biomedical Model"
            results[model_name] = biomedical_results
    
    # Display and save results
    if results:
        print_comparison_table(results)
        
        if args.output:
            save_results(results, args.output)
        else:
            timestamp = int(time.time())
            save_results(results, f"sapbert_biomegatron_results_{timestamp}.json")
    else:
        print("No evaluations were performed. Use --help to see available options.")

if __name__ == "__main__":
    main() 