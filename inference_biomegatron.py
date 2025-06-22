#!/usr/bin/env python3
"""
Inference script for BioMegatron entity linking model.

This script demonstrates how to:
1. Load the trained BioMegatron model
2. Rank MONDO candidates for a given mention
3. Return top-k predictions with optimized batch processing
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

class BioMegatronEntityLinker:
    def __init__(self, model_path="models/biomegatron_mondo_cls_final", batch_size=64):
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
        """Load available MONDO IDs from training data."""
        try:
            train_df = pd.read_csv('data/mondo_train.csv')
            candidates = sorted(train_df['mondo_id'].unique())
            print(f"Loaded {len(candidates)} MONDO candidates")
            return candidates
        except FileNotFoundError:
            print("Warning: Training data not found, using dummy candidates")
            return ["MONDO:0000001", "MONDO:0000002", "MONDO:0000003"]
    
    def score_candidates_batch(self, mention, candidates=None):
        """Score MONDO candidates for a given mention using efficient batch processing."""
        if candidates is None:
            candidates = self.mondo_candidates
        
        # Create mention-candidate pairs
        pairs = [f"{mention} [SEP] {candidate}" for candidate in candidates]
        
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
        
        # Create results
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
        
        for mention in tqdm(mentions, desc="BioMegatron batch inference"):
            results[mention] = self.predict(mention, top_k)
        
        return results
    
    def predict_batch(self, mentions, top_k=5):
        """Predict for multiple mentions (legacy method)."""
        return self.predict_batch_efficient(mentions, top_k)

def main():
    """Demonstration of the entity linker."""
    
    # Check if model exists
    model_path = Path("models/biomegatron_mondo_cls_final")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run train_biomegatron_cls.py first to train the model")
        return
    
    # Initialize entity linker
    linker = BioMegatronEntityLinker()
    
    # Example mentions
    test_mentions = [
        "cystic fibrosis",
        "diabetes mellitus",
        "alzheimer disease",
        "breast cancer"
    ]
    
    print("\n" + "="*50)
    print("BioMegatron Entity Linking Demo")
    print("="*50)
    
    for mention in test_mentions:
        print(f"\nMention: '{mention}'")
        predictions = linker.predict(mention, top_k=3)
        
        print("Top 3 MONDO candidates:")
        for i, (mondo_id, score) in enumerate(predictions, 1):
            print(f"  {i}. {mondo_id} (score: {score:.4f})")
    
    print("\n" + "="*50)
    print("Custom mention prediction:")
    custom_mention = input("Enter a medical mention: ").strip()
    if custom_mention:
        predictions = linker.predict(custom_mention, top_k=5)
        print(f"\nTop 5 predictions for '{custom_mention}':")
        for i, (mondo_id, score) in enumerate(predictions, 1):
            print(f"  {i}. {mondo_id} (score: {score:.4f})")

if __name__ == "__main__":
    main() 