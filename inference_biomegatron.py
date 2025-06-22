#!/usr/bin/env python3
"""
Inference script for BioMegatron entity linking model.

This script demonstrates how to:
1. Load the trained BioMegatron model
2. Rank MONDO candidates for a given mention
3. Return top-k predictions
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from pathlib import Path

class BioMegatronEntityLinker:
    def __init__(self, model_path="models/biomegatron_mondo_cls_final"):
        """Initialize the entity linker with trained model."""
        self.model_path = Path(model_path)
        
        print(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        
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
    
    def score_candidates(self, mention, candidates=None):
        """Score MONDO candidates for a given mention."""
        if candidates is None:
            candidates = self.mondo_candidates
        
        # Create mention-candidate pairs
        pairs = [f"{mention} [SEP] {candidate}" for candidate in candidates]
        
        # Tokenize all pairs
        inputs = self.tokenizer(
            pairs,
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt"
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        
        # Handle single candidate case
        if len(candidates) == 1:
            scores = [scores]
        
        # Create results
        results = list(zip(candidates, scores))
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
        
        return results
    
    def predict(self, mention, top_k=5):
        """Get top-k MONDO predictions for a mention."""
        scores = self.score_candidates(mention)
        return scores[:top_k]
    
    def predict_batch(self, mentions, top_k=5):
        """Predict for multiple mentions."""
        results = {}
        for mention in mentions:
            results[mention] = self.predict(mention, top_k)
        return results

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