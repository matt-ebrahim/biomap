#!/usr/bin/env python3
"""
Unified evaluation script for all entity linking baselines.

This script evaluates and compares:
1. SapBERT + FAISS similarity search
2. BioMegatron fine-tuned classifier  
3. GPT-4o zero-shot evaluation
4. Gemini search-grounded evaluation

Provides Hits@K and MRR metrics for comprehensive comparison.
"""

import pandas as pd
import numpy as np
import torch
import faiss
import json
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import our custom modules
try:
    from inference_sapbert import SapBERTEntityLinker
    from inference_biomegatron import BioMegatronEntityLinker
    from llm_zero_shot import llm_link, initialize_clients
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all baseline scripts are in the same directory")
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
    print("🔍 Evaluating SapBERT + FAISS")
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

def evaluate_biomegatron(test_df: pd.DataFrame, model_dir: str = "models") -> Dict[str, float]:
    """Evaluate BioMegatron classifier approach."""
    print("\n" + "="*50)
    print("🧠 Evaluating BioMegatron Classifier")
    print("="*50)
    
    try:
        linker = BioMegatronEntityLinker()
        
        ranked_lists = []
        for mention in tqdm(test_df['mention'], desc="BioMegatron inference"):
            results = linker.predict(mention, top_k=10)
            ranked_list = [mondo_id for mondo_id, score in results]
            ranked_lists.append(ranked_list)
        
        metrics = evaluate_hits_and_mrr(test_df['mondo_id'].tolist(), ranked_lists)
        
        print("BioMegatron Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"BioMegatron evaluation failed: {e}")
        return {}

def evaluate_llm_ranking_parallel(test_df: pd.DataFrame, 
                                 model: str = "gpt", 
                                 top_k: int = 10,
                                 max_workers: int = 8) -> Dict[str, float]:
    """
    Evaluate LLM-based ranking approach with parallel processing.
    
    Args:
        test_df: Test dataframe
        model: "gpt" or "gemini"
        top_k: Number of top predictions to return
        max_workers: Number of parallel threads
    """
    print(f"\n" + "="*50)
    print(f"🤖 Evaluating {model.upper()} Zero-shot Ranking (Parallel)")
    print("="*50)
    
    try:
        # Load MONDO candidates (from SapBERT labels if available)
        candidates_file = Path("models/sapbert_mondo_labels.npy")
        if candidates_file.exists():
            all_candidates = np.load(str(candidates_file)).tolist()
        else:
            # Fallback: use unique MONDO IDs from training data
            train_df = pd.read_csv("data/mondo_train.csv")
            all_candidates = sorted(train_df['mondo_id'].unique())
        
        print(f"Loaded {len(all_candidates)} MONDO candidates")
        print(f"Using {max_workers} parallel threads for evaluation")
        
        def evaluate_single_mention(mention_data):
            """Evaluate a single mention against all candidates."""
            mention, gt_mondo = mention_data
            
            def evaluate_pair(candidate):
                try:
                    is_equivalent, explanation = llm_link(mention, candidate, model)
                    return candidate, 1.0 if is_equivalent else 0.0
                except Exception as e:
                    print(f"Error evaluating {mention} -> {candidate}: {e}")
                    return candidate, 0.0
            
            # Use thread pool for each mention's candidates
            with ThreadPoolExecutor(max_workers=min(max_workers, len(all_candidates))) as executor:
                future_to_candidate = {executor.submit(evaluate_pair, candidate): candidate 
                                     for candidate in all_candidates}
                
                scores = {}
                for future in as_completed(future_to_candidate):
                    candidate, score = future.result()
                    scores[candidate] = score
            
            # Rank by scores (descending)
            ranked_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranked_list = [candidate for candidate, score in ranked_candidates[:top_k]]
            
            return ranked_list
        
        # Prepare mention data
        mention_data = list(zip(test_df['mention'], test_df['mondo_id']))
        
        # Process all mentions
        ranked_lists = []
        with tqdm(total=len(mention_data), desc=f"{model.upper()} mentions") as pbar:
            for mention_item in mention_data:
                ranked_list = evaluate_single_mention(mention_item)
                ranked_lists.append(ranked_list)
                pbar.update(1)
        
        metrics = evaluate_hits_and_mrr(test_df['mondo_id'].tolist(), ranked_lists)
        
        print(f"{model.upper()} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"{model.upper()} evaluation failed: {e}")
        return {}

def save_results(results: Dict[str, Dict[str, float]], output_file: str):
    """Save evaluation results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted comparison table of all results."""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE EVALUATION RESULTS")
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
    print(f"{'Model':<20}", end="")
    for metric in all_metrics:
        print(f"{metric:>12}", end="")
    print()
    print("-" * 80)
    
    # Print results for each model
    for model_name, metrics in results.items():
        print(f"{model_name:<20}", end="")
        for metric in all_metrics:
            value = metrics.get(metric, 0.0)
            print(f"{value:>12.4f}", end="")
        print()
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Unified Entity Linking Evaluation")
    parser.add_argument("--test", default="data/mondo_test.csv", help="Test file")
    parser.add_argument("--models", default="models", help="Models directory")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--max_samples", type=int, help="Limit test samples")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel threads for LLM evaluation")
    parser.add_argument("--skip_sapbert", action="store_true", help="Skip SapBERT evaluation")
    parser.add_argument("--skip_biomegatron", action="store_true", help="Skip BioMegatron evaluation")
    parser.add_argument("--skip_gpt", action="store_true", help="Skip GPT-4o evaluation")
    parser.add_argument("--skip_gemini", action="store_true", help="Skip Gemini evaluation")
    
    args = parser.parse_args()
    
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
    
    # Initialize LLM clients if needed
    if not args.skip_gpt or not args.skip_gemini:
        try:
            print("Initializing LLM clients...")
            initialize_clients()
        except Exception as e:
            print(f"Failed to initialize LLM clients: {e}")
            print("Skipping LLM evaluations")
            args.skip_gpt = True
            args.skip_gemini = True
    
    # Run evaluations
    results = {}
    
    # 1. SapBERT + FAISS
    if not args.skip_sapbert:
        sapbert_results = evaluate_sapbert(test_df, args.models)
        if sapbert_results:
            results["SapBERT + FAISS"] = sapbert_results
    
    # 2. BioMegatron Classifier
    if not args.skip_biomegatron:
        biomegatron_results = evaluate_biomegatron(test_df, args.models)
        if biomegatron_results:
            results["BioMegatron"] = biomegatron_results
    
    # 3. GPT-4o Zero-shot
    if not args.skip_gpt:
        gpt_results = evaluate_llm_ranking_parallel(test_df, "gpt", max_workers=args.max_workers)
        if gpt_results:
            results["GPT-4o"] = gpt_results
    
    # 4. Gemini Search-grounded
    if not args.skip_gemini:
        gemini_results = evaluate_llm_ranking_parallel(test_df, "gemini", max_workers=args.max_workers)
        if gemini_results:
            results["Gemini"] = gemini_results
    
    # Display and save results
    print_comparison_table(results)
    
    if args.output:
        save_results(results, args.output)
    else:
        timestamp = int(time.time())
        save_results(results, f"evaluation_results_{timestamp}.json")

if __name__ == "__main__":
    main() 