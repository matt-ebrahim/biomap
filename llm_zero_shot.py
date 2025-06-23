#!/usr/bin/env python3
"""
Zero-shot LLM evaluation for entity linking.

This script provides a unified interface for both GPT-4o and Gemini models
to evaluate entity linking accuracy using natural language understanding
with optimized parallel processing for faster evaluation.
"""

import time
import os
import openai
from google import genai
from google.genai import types
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Dict, Tuple, List
import json
import concurrent.futures
from threading import Lock
import threading

# Configuration - Updated to match your working examples
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", "sk-mN4RQgdN886QbidYh_-D_w"),
    "base_url": "https://llmgateway.experiment.trialspark.com",
    "model_name": "gpt-4o"
}

GEMINI_CONFIG = {
    "api_key": os.getenv("GOOGLE_API_KEY", "AIzaSyDQNKpdcH8RYDfy8WfBS3QOucRF4EwUIVk"),
    "model_name": "gemini-2.5-flash-preview-04-17"
}

# Enhanced parallel processing settings
MAX_WORKERS = 16
DELAY_BETWEEN_REQUESTS = 0.1  # Reduced delay for faster processing
CHECKPOINT_FREQUENCY = 100

# Global clients with thread-local storage
openai_client = None
gemini_client = None
request_lock = Lock()
thread_local = threading.local()

def get_openai_client():
    """Get thread-local OpenAI client."""
    if not hasattr(thread_local, 'openai_client'):
        thread_local.openai_client = openai
    return thread_local.openai_client

def get_gemini_client():
    """Get thread-local Gemini client (exactly matching your working pattern)."""
    if not hasattr(thread_local, 'gemini_client'):
        thread_local.gemini_client = genai.Client(api_key=GEMINI_CONFIG["api_key"])
    return thread_local.gemini_client

def initialize_clients():
    """Initialize OpenAI and Gemini clients exactly matching your working examples."""
    global openai_client, gemini_client
    
    # Validate API keys
    if not OPENAI_CONFIG["api_key"]:
        raise ValueError("OPENAI_API_KEY not configured")
    if not GEMINI_CONFIG["api_key"]:
        raise ValueError("GOOGLE_API_KEY not configured")
    
    # Initialize OpenAI (exactly matching your working example)
    openai.api_key = OPENAI_CONFIG["api_key"]
    openai.base_url = OPENAI_CONFIG["base_url"]
    openai_client = openai
    
    # Initialize Gemini (exactly matching your working example)
    gemini_client = genai.Client(api_key=GEMINI_CONFIG["api_key"])

def query_equivalence_gpt(mention: str, mondo_label: str, retries: int = 3) -> str:
    """Query GPT-4o for entity equivalence with optimized parallel processing."""
    prompt = f"""
Do the following two terms refer to the same disease concept?

Term 1 (Medical Mention): "{mention}"
Term 2 (MONDO Disease): "{mondo_label}"

Consider:
- Synonymous terms and clinical equivalents
- Different levels of specificity (general vs specific conditions)
- Alternative medical terminology for the same condition
- Abbreviations and full forms
- Common disease names vs formal medical terms

Answer only with "YES" if they refer to the same disease concept, or "NO" if they refer to different conditions.
Provide a brief justification in one sentence.

Format your response as: YES/NO [explanation]
"""
    
    client = get_openai_client()
    
    for attempt in range(retries):
        try:
            # Brief delay between API calls
            time.sleep(0.1)
            
            response = client.chat.completions.create(
                model=OPENAI_CONFIG["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
                wait_time = (2 ** attempt) * 2  # Reduced backoff
                time.sleep(wait_time)
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {str(e)}"
    
    return "ERROR: Max retries exceeded"

def query_equivalence_gemini(mention: str, mondo_label: str, retries: int = 3) -> str:
    """Query Gemini for entity equivalence (exactly matching your working example)."""
    prompt = f"""
You are a medical terminology expert. Please determine if these terms refer to the same disease concept.

Term 1 (Medical Mention): "{mention}"
Term 2 (MONDO Disease): "{mondo_label}"

Please consider:
1. Medical synonyms and alternative names
2. Different levels of specificity (general vs specific conditions)
3. Abbreviations vs full forms
4. Common names vs formal medical terminology
5. ICD classifications and medical ontologies

Answer with "YES" or "NO", followed by a brief medical explanation.

Format your response as: YES/NO [explanation]
"""
    
    client = get_gemini_client()
    
    for attempt in range(retries):
        try:
            # Brief delay between API calls
            time.sleep(0.1)
            
            # Exactly matching your working example
            response = client.models.generate_content(
                model=GEMINI_CONFIG["model_name"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT"],
                    temperature=0.1,
                    topK=40,
                    topP=0.95
                )
            )
            
            if not response.candidates or not response.candidates[0].content.parts:
                return "ERROR: No response generated"
            
            content = " ".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            
            if not content:
                return "ERROR: Empty response content"
            
            return content.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate" in error_msg or "limit" in error_msg or "429" in error_msg or "quota" in error_msg:
                wait_time = (2 ** attempt) * 5  # Matching your example timing
                time.sleep(wait_time)
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {str(e)}"
    
    return "ERROR: Max retries exceeded"

def llm_link(mention: str, mondo_label: str, model: str = "gpt") -> Tuple[bool, str]:
    """
    Unified function for entity linking using LLMs with parallel processing support.
    
    Args:
        mention: Medical mention text
        mondo_label: MONDO disease ID or label
        model: "gpt" for GPT-4o or "gemini" for Gemini
        
    Returns:
        Tuple of (is_equivalent: bool, explanation: str)
    """
    if model.lower() == "gpt":
        response = query_equivalence_gpt(mention, mondo_label)
    elif model.lower() == "gemini":
        response = query_equivalence_gemini(mention, mondo_label)
    else:
        raise ValueError(f"Unsupported model: {model}. Use 'gpt' or 'gemini'")
    
    # Parse response
    if response.startswith("ERROR"):
        return False, response
    
    # Extract YES/NO from response
    response_upper = response.upper()
    is_equivalent = response_upper.startswith("YES")
    
    return is_equivalent, response

def parallel_llm_evaluation(mentions: List[str], 
                           mondo_candidates: List[str], 
                           model: str = "gpt",
                           max_workers: int = 16) -> List[List[Tuple[str, float]]]:
    """
    Parallel evaluation of multiple mentions against MONDO candidates.
    
    Args:
        mentions: List of medical mentions
        mondo_candidates: List of MONDO IDs to evaluate against
        model: "gpt" or "gemini"
        max_workers: Number of parallel threads
        
    Returns:
        List of ranked results for each mention
    """
    print(f"Starting parallel {model.upper()} evaluation with {max_workers} workers")
    
    def evaluate_mention_batch(mention_idx):
        mention = mentions[mention_idx]
        results = []
        
        for candidate in mondo_candidates:
            try:
                is_equivalent, explanation = llm_link(mention, candidate, model)
                score = 1.0 if is_equivalent else 0.0
                results.append((candidate, score))
            except Exception as e:
                results.append((candidate, 0.0))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return mention_idx, results
    
    # Process all mentions in parallel
    all_results = [None] * len(mentions)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(evaluate_mention_batch, i): i 
                        for i in range(len(mentions))}
        
        # Collect results with progress bar
        with tqdm(total=len(mentions), desc=f"{model.upper()} parallel eval") as pbar:
            for future in concurrent.futures.as_completed(future_to_idx):
                mention_idx, results = future.result()
                all_results[mention_idx] = results
                pbar.update(1)
    
    return all_results

def evaluate_single_mention(args) -> Dict:
    """Evaluate a single mention (for parallel processing)."""
    mention, true_mondo, model, index = args
    
    try:
        is_equivalent, explanation = llm_link(mention, true_mondo, model)
        
        return {
            'index': index,
            'mention': mention,
            'true_mondo': true_mondo,
            'prediction': is_equivalent,
            'explanation': explanation,
            'correct': is_equivalent  # Since we're comparing against ground truth
        }
    except Exception as e:
        return {
            'index': index,
            'mention': mention,
            'true_mondo': true_mondo,
            'prediction': False,
            'explanation': f"ERROR: {str(e)}",
            'correct': False
        }

def evaluate_llm_linking(test_file: str = "data/mondo_test.csv", 
                        model: str = "gpt",
                        output_file: str = None,
                        max_samples: int = None) -> Dict[str, float]:
    """
    Evaluate LLM-based entity linking performance.
    
    Args:
        test_file: Path to test CSV file
        model: "gpt" or "gemini"
        output_file: Optional path to save detailed results
        max_samples: Optional limit on number of test samples
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading test data from {test_file}")
    test_df = pd.read_csv(test_file)
    
    if max_samples:
        test_df = test_df.head(max_samples)
        print(f"Limited to {max_samples} samples for evaluation")
    
    print(f"Evaluating {len(test_df)} examples using {model.upper()}")
    
    # Prepare arguments for parallel processing
    args_list = [
        (row['mention'], row['mondo_id'], model, idx)
        for idx, row in test_df.iterrows()
    ]
    
    results = []
    
    # Process with threading for I/O bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(evaluate_single_mention, args): args for args in args_list}
        
        # Collect results with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_args), 
                          total=len(args_list), 
                          desc=f"Evaluating with {model.upper()}"):
            result = future.result()
            results.append(result)
            
            # Checkpoint saving
            if len(results) % CHECKPOINT_FREQUENCY == 0:
                checkpoint_file = f"checkpoint_{model}_{len(results)}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nCheckpoint saved: {checkpoint_file}")
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])
    
    # Calculate metrics
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # Count errors
    errors = sum(1 for r in results if r['explanation'].startswith('ERROR'))
    error_rate = errors / total if total > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'errors': errors,
        'error_rate': error_rate,
        'model': model
    }
    
    print(f"\n{model.upper()} Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct: {correct}/{total}")
    print(f"Errors: {errors} ({error_rate:.4f})")
    
    # Save detailed results
    if output_file:
        detailed_results = {
            'metrics': metrics,
            'predictions': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"Detailed results saved to: {output_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Zero-shot LLM Entity Linking Evaluation")
    parser.add_argument("--test", default="data/mondo_test.csv", help="Test file for evaluation")
    parser.add_argument("--model", choices=["gpt", "gemini"], default="gpt", help="LLM model to use")
    parser.add_argument("--output", help="Output file for detailed results (JSON)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    
    args = parser.parse_args()
    
    # Initialize clients
    print("Initializing LLM clients...")
    initialize_clients()
    
    if args.demo:
        print("\n" + "="*50)
        print(f"Zero-shot LLM Entity Linking Demo ({args.model.upper()})")
        print("="*50)
        
        example_pairs = [
            ("cystic fibrosis", "MONDO:0009061"),
            ("diabetes", "MONDO:0005015"),
            ("heart attack", "MONDO:0005068"),
            ("breast cancer", "MONDO:0007254"),
            ("flu", "MONDO:0005812")
        ]
        
        print("\nExample evaluations:")
        for mention, mondo_id in example_pairs:
            is_equiv, explanation = llm_link(mention, mondo_id, args.model)
            result = "✓ MATCH" if is_equiv else "✗ NO MATCH"
            print(f"\nMention: '{mention}' → {mondo_id}")
            print(f"Result: {result}")
            print(f"Explanation: {explanation}")
        
        # Interactive mode
        print("\n" + "="*30)
        print("Interactive evaluation:")
        print("="*30)
        
        while True:
            try:
                mention = input("\nEnter medical mention (or 'quit'): ").strip()
                if mention.lower() in ['quit', 'exit', 'q']:
                    break
                
                mondo_id = input("Enter MONDO ID: ").strip()
                if not mondo_id:
                    continue
                
                print("Evaluating...")
                is_equiv, explanation = llm_link(mention, mondo_id, args.model)
                result = "✓ EQUIVALENT" if is_equiv else "✗ NOT EQUIVALENT"
                
                print(f"\nResult: {result}")
                print(f"Explanation: {explanation}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    else:
        # Run evaluation
        if not Path(args.test).exists():
            print(f"Error: Test file {args.test} not found")
            return
        
        output_file = args.output or f"llm_results_{args.model}_{int(time.time())}.json"
        
        metrics = evaluate_llm_linking(
            test_file=args.test,
            model=args.model,
            output_file=output_file,
            max_samples=args.max_samples
        )
        
        print(f"\nFinal Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

if __name__ == "__main__":
    main() 