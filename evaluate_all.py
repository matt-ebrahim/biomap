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
    from inference_biomegatron import BiomedicalEntityLinker, find_available_models
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
        
        return metrics
        
    except Exception as e:
        print(f"Biomedical model evaluation failed: {e}")
        return {}

def evaluate_llm_ranking_parallel(test_df: pd.DataFrame, 
                                 model: str = "gpt", 
                                 top_k: int = 10,
                                 max_workers: int = None,
                                 batch_size: int = 100,
                                 batch_delay: float = 3.0) -> Dict[str, float]:
    """
    Evaluate LLM-based ranking approach with full candidate set and batched parallel processing.
    
    Args:
        test_df: Test dataframe
        model: "gpt" or "gemini"
        top_k: Number of top predictions to return
        max_workers: Number of parallel threads (None = use all available cores)
        batch_size: Number of candidates to process per batch
        batch_delay: Seconds to wait between batches
    """
    print(f"\n" + "="*50)
    print(f"Evaluating {model.upper()} Zero-shot Ranking")
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
        
        # Create MONDO ID to representative mention text mapping
        train_df = pd.read_csv("data/mondo_train.csv")
        mondo_to_mentions = train_df.groupby('mondo_id')['mention'].apply(list).to_dict()
        mondo_to_text = {}
        for mondo_id, mentions in mondo_to_mentions.items():
            # Use most frequent mention as representative
            mention_counts = pd.Series(mentions).value_counts()
            representative_mention = mention_counts.index[0]
            mondo_to_text[mondo_id] = representative_mention
        
        # Use all available CPU cores if not specified
        if max_workers is None:
            import multiprocessing
            max_workers = multiprocessing.cpu_count()
        
        print(f"Using all {len(all_candidates)} MONDO candidates")
        print(f"Parallel processing with {max_workers} workers")
        print(f"Batch processing: {batch_size} candidates per batch, {batch_delay}s delay between batches")
        print(f"Using representative mention text for LLM evaluation")
        
        def evaluate_single_mention(mention_data):
            """Evaluate a single mention against all candidates."""
            mention, gt_mondo = mention_data
            
            def evaluate_pair(candidate):
                try:
                    # Use representative mention text instead of MONDO ID
                    candidate_text = mondo_to_text.get(candidate, candidate)
                    is_equivalent, explanation = llm_link(mention, candidate_text, model)
                    return candidate, 1.0 if is_equivalent else 0.0
                except Exception as e:
                    print(f"Error evaluating {mention} -> {candidate}: {e}")
                    return candidate, 0.0
            
            scores = {}
            
            # Process candidates in batches to manage rate limits
            total_batches = (len(all_candidates) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(all_candidates), batch_size):
                batch_candidates = all_candidates[batch_idx:batch_idx + batch_size]
                current_batch_num = (batch_idx // batch_size) + 1
                
                print(f"  Processing batch {current_batch_num}/{total_batches} ({len(batch_candidates)} candidates)")
                
                # Process this batch with parallel workers
                with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_candidates))) as executor:
                    future_to_candidate = {executor.submit(evaluate_pair, candidate): candidate 
                                         for candidate in batch_candidates}
                    
                    # Collect results from this batch
                    batch_progress = tqdm(as_completed(future_to_candidate), 
                                        total=len(batch_candidates),
                                        desc=f"Batch {current_batch_num}")
                    
                    for future in batch_progress:
                        candidate, score = future.result()
                        scores[candidate] = score
                
                # Pause between batches to respect rate limits (except for last batch)
                if batch_idx + batch_size < len(all_candidates):
                    print(f"  Waiting {batch_delay} seconds before next batch...")
                    time.sleep(batch_delay)
            
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
    print("COMPREHENSIVE EVALUATION RESULTS")
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
    parser.add_argument("--batch_size", type=int, default=100, help="Number of candidates to process per batch for LLM evaluation")
    parser.add_argument("--batch_delay", type=float, default=3.0, help="Seconds to wait between batches for LLM evaluation")
    parser.add_argument("--skip_sapbert", action="store_true", help="Skip SapBERT evaluation")
    parser.add_argument("--skip_biomedical", action="store_true", help="Skip biomedical model evaluation")
    parser.add_argument("--biomedical_model", type=str, help="Specific biomedical model to use (e.g., models_345m_biomegatron345muncased)")
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
    
    # 2. Biomedical Classifier
    if not args.skip_biomedical:
        biomedical_results = evaluate_biomedical_model(test_df, args.models, args.biomedical_model)
        if biomedical_results:
            # Use model-specific name for results
            available_models = find_available_models()
            model_key = args.biomedical_model if args.biomedical_model and args.biomedical_model in available_models else list(available_models.keys())[0] if available_models else "Biomedical"
            if available_models and model_key in available_models:
                model_name = f"{available_models[model_key]['display_name']} ({available_models[model_key]['parameters']})"
            else:
                model_name = "Biomedical Model"
            results[model_name] = biomedical_results
    
    # 3. GPT-4o Zero-shot
    if not args.skip_gpt:
        gpt_results = evaluate_llm_ranking_parallel(
            test_df, "gpt", 
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            batch_delay=args.batch_delay
        )
        if gpt_results:
            results["GPT-4o"] = gpt_results
    
    # 4. Gemini Search-grounded
    if not args.skip_gemini:
        gemini_results = evaluate_llm_ranking_parallel(
            test_df, "gemini", 
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            batch_delay=args.batch_delay
        )
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