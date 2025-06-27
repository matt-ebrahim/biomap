#!/usr/bin/env python3
"""
Inference script for biomedical entity linking models.

This script demonstrates how to:
1. Load trained biomedical language models
2. Rank MONDO candidates for a given mention
3. Return top-k predictions with optimized batch processing
4. Support multiple model types and sizes
"""

import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import glob

def find_available_models():
    """Find all available trained models in the models directory."""
    models_dir = Path("models")
    available_models = {}
    
    if not models_dir.exists():
        return available_models
    
    # Look for model directories with the pattern models_*
    for model_dir in models_dir.glob("models_*"):
        if model_dir.is_dir():
            final_model_path = model_dir / "final_model"
            model_info_path = model_dir / "model_info.json"
            
            if final_model_path.exists():
                # Load model info if available
                model_info = {}
                if model_info_path.exists():
                    try:
                        with open(model_info_path, 'r') as f:
                            model_info = json.load(f)
                    except:
                        pass
                
                available_models[model_dir.name] = {
                    "path": final_model_path,
                    "info": model_info,
                    "display_name": model_info.get("display_name", model_dir.name),
                    "parameters": model_info.get("formatted_parameters", "unknown"),
                    "model_key": model_info.get("model_key", "unknown")
                }
    
    return available_models

def list_available_models():
    """Print all available trained models."""
    models = find_available_models()
    
    if not models:
        print("No trained models found in the models/ directory.")
        print("Please run train_biomegatron_cls.py first to train a model.")
        return
    
    print("\n" + "="*80)
    print("AVAILABLE TRAINED MODELS")
    print("="*80)
    
    for model_key, model_data in models.items():
        print(f"\nModel Directory: {model_key}")
        print(f"  Display Name: {model_data['display_name']}")
        print(f"  Parameters: {model_data['parameters']}")
        print(f"  Model Type: {model_data['model_key']}")
        print(f"  Path: {model_data['path']}")
    
    print("\n" + "="*80)

class BiomedicalEntityLinker:
    def __init__(self, model_path, batch_size=64):
        """Initialize the entity linker with trained model."""
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        
        print(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Load MONDO candidates (in practice, you'd have a knowledge base)
        self.mondo_candidates = self._load_mondo_candidates()
        
    def _load_mondo_candidates(self):
        """Load available MONDO IDs and their representative mentions from training data."""
        try:
            train_df = pd.read_csv('data/mondo_train.csv')
            
            # Create MONDO ID -> representative mention mapping
            mondo_to_mentions = train_df.groupby('mondo_id')['mention'].apply(list).to_dict()
            
            # Use the most frequent mention as representative for each MONDO ID
            self.mondo_to_text = {}
            for mondo_id, mentions in mondo_to_mentions.items():
                mention_counts = pd.Series(mentions).value_counts()
                representative_mention = mention_counts.index[0]  # Most common
                self.mondo_to_text[mondo_id] = representative_mention
            
            candidates = sorted(self.mondo_to_text.keys())
            print(f"Loaded {len(candidates)} MONDO candidates")
            print("Sample MONDO ID -> mention mappings:")
            for i, mondo_id in enumerate(candidates[:3]):
                print(f"  {mondo_id} -> '{self.mondo_to_text[mondo_id]}'")
            
            return candidates
        except FileNotFoundError:
            print("Warning: Training data not found, using dummy candidates")
            self.mondo_to_text = {
                "MONDO:0000001": "disease",
                "MONDO:0000002": "disorder", 
                "MONDO:0000003": "condition"
            }
            return ["MONDO:0000001", "MONDO:0000002", "MONDO:0000003"]
    
    def score_candidates_batch(self, mention, candidates=None):
        """Score MONDO candidates using representative mention text for better semantic matching."""
        if candidates is None:
            candidates = self.mondo_candidates
        
        # Create mention-candidate pairs using representative mention text instead of MONDO IDs
        pairs = []
        for candidate in candidates:
            # Use representative mention text for this MONDO ID
            representative_text = self.mondo_to_text.get(candidate, candidate)
            pairs.append(f"{mention} [SEP] {representative_text}")
        
        all_scores = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_pairs,
                truncation=True,
                padding=True,
                max_length=64,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
            
            # Handle single item case
            if len(batch_pairs) == 1:
                batch_scores = [batch_scores]
            
            all_scores.extend(batch_scores)
        
        # Create results (return MONDO IDs, not the representative text)
        results = list(zip(candidates, all_scores))
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
        
        return results
    
    def score_candidates(self, mention, candidates=None):
        """Score MONDO candidates for a given mention (legacy method for compatibility)."""
        return self.score_candidates_batch(mention, candidates)
    
    def predict(self, mention, top_k=5):
        """Get top-k MONDO predictions for a mention."""
        scores = self.score_candidates_batch(mention)
        return scores[:top_k]
    
    def predict_batch_efficient(self, mentions, top_k=5):
        """Efficiently predict for multiple mentions with progress tracking."""
        results = {}
        
        print(f"Processing {len(mentions)} mentions with batch size {self.batch_size}")
        
        for mention in tqdm(mentions, desc="Biomedical model batch inference"):
            results[mention] = self.predict(mention, top_k)
        
        return results
    
    def predict_batch(self, mentions, top_k=5):
        """Predict for multiple mentions (legacy method)."""
        return self.predict_batch_efficient(mentions, top_k)

def main():
    """Demonstration of the entity linker."""
    parser = argparse.ArgumentParser(description='Biomedical entity linking inference')
    parser.add_argument('--model', type=str, 
                       help='Model directory to use (e.g., models_345m_biomegatron345muncased)')
    parser.add_argument('--list-models', action='store_true', 
                       help='List available models and exit')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--batch-size', type=int, default=64, 
                       help='Batch size for inference (default: 64)')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    # Find available models
    available_models = find_available_models()
    
    if not available_models:
        print("No trained models found in the models/ directory.")
        print("Please run train_biomegatron_cls.py first to train a model.")
        return
    
    # Select model
    if args.model:
        if args.model not in available_models:
            print(f"Error: Model '{args.model}' not found.")
            print("Available models:")
            for model_key in available_models.keys():
                print(f"  - {model_key}")
            return
        selected_model = args.model
    else:
        # Auto-select the first available model or prompt user
        if len(available_models) == 1:
            selected_model = list(available_models.keys())[0]
            print(f"Auto-selecting the only available model: {selected_model}")
        else:
            print("Multiple models available:")
            model_keys = list(available_models.keys())
            for i, model_key in enumerate(model_keys):
                model_data = available_models[model_key]
                print(f"  {i+1}. {model_key} ({model_data['display_name']}, {model_data['parameters']})")
            
            while True:
                try:
                    choice = input(f"Select a model (1-{len(model_keys)}): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(model_keys):
                        selected_model = model_keys[choice_idx]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(model_keys)}")
                except (ValueError, KeyboardInterrupt):
                    print("Invalid selection")
                    return
    
    model_data = available_models[selected_model]
    model_path = model_data["path"]
    
    print(f"\nSelected Model: {model_data['display_name']}")
    print(f"Parameters: {model_data['parameters']}")
    print(f"Model Path: {model_path}")
    
    # Initialize entity linker
    try:
        linker = BiomedicalEntityLinker(model_path, batch_size=args.batch_size)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Example mentions
    test_mentions = [
        "cystic fibrosis",
        "diabetes mellitus",
        "alzheimer disease",
        "breast cancer"
    ]
    
    print("\n" + "="*50)
    print("Biomedical Entity Linking Demo")
    print("="*50)
    
    for mention in test_mentions:
        print(f"\nMention: '{mention}'")
        predictions = linker.predict(mention, top_k=3)
        
        print("Top 3 MONDO candidates:")
        for i, (mondo_id, score) in enumerate(predictions, 1):
            print(f"  {i}. {mondo_id} (score: {score:.4f})")
    
    if args.interactive:
        print("\n" + "="*50)
        print("Interactive mode - Enter medical mentions (Ctrl+C to exit):")
        try:
            while True:
                custom_mention = input("\nEnter a medical mention: ").strip()
                if custom_mention:
                    predictions = linker.predict(custom_mention, top_k=5)
                    print(f"\nTop 5 predictions for '{custom_mention}':")
                    for i, (mondo_id, score) in enumerate(predictions, 1):
                        print(f"  {i}. {mondo_id} (score: {score:.4f})")
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")

if __name__ == "__main__":
    main() 