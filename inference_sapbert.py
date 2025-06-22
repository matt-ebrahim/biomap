#!/usr/bin/env python3
"""
SapBERT + FAISS inference for entity linking.

This script:
1. Loads trained SapBERT model and FAISS index
2. Encodes mentions and searches for similar MONDO entities
3. Evaluates performance using Hits@K and MRR metrics
4. Provides interactive inference interface
"""

from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import faiss
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict

def setup_device():
    """Setup computing device (CUDA/MPS/CPU)."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple MPS")  
    else:
        device = 'cpu'
        print("Using CPU")
    return device

class SapBERTEntityLinker:
    def __init__(self, model_dir="models"):
        """Initialize SapBERT entity linker."""
        self.model_dir = Path(model_dir)
        
        # Load model
        self.device = setup_device()
        self.tokenizer, self.model = self._load_model()
        
        # Load FAISS index and labels
        self.index, self.labels = self._load_index()
        
        print(f"Loaded {len(self.labels)} MONDO entities")
        print(f"FAISS index has {self.index.ntotal} entries")
    
    def _load_model(self):
        """Load SapBERT model and tokenizer."""
        print("Loading SapBERT model...")
        model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        
        return tokenizer, model
    
    def _load_index(self):
        """Load FAISS index and labels."""
        index_file = self.model_dir / "sapbert_mondo.faiss"
        labels_file = self.model_dir / "sapbert_mondo_labels.npy"
        
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        print(f"Loading FAISS index from {index_file}")
        index = faiss.read_index(str(index_file))
        
        print(f"Loading labels from {labels_file}")
        labels = np.load(str(labels_file))
        
        return index, labels
    
    def encode_mention(self, mention: str) -> np.ndarray:
        """Encode a single mention to embedding."""
        inputs = self.tokenizer(
            mention,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=25
        ).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(**inputs).pooler_output
        
        # Normalize for cosine similarity
        embedding = embedding.cpu().numpy()
        faiss.normalize_L2(embedding)
        
        return embedding
    
    def encode_mentions(self, mentions: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple mentions to embeddings."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(mentions), batch_size), desc="Encoding mentions"):
            batch_mentions = mentions[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_mentions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=25
            ).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(**inputs).pooler_output
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        faiss.normalize_L2(embeddings)
        
        return embeddings
    
    def search(self, mention: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for top-k similar MONDO entities."""
        # Encode mention
        mention_embedding = self.encode_mention(mention)
        
        # Search in FAISS index
        scores, indices = self.index.search(mention_embedding, top_k)
        
        # Get results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                mondo_id = str(self.labels[idx])  # Convert numpy string to Python string
                results.append((mondo_id, float(score)))
        
        return results
    
    def predict_batch(self, mentions: List[str], top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Predict for multiple mentions."""
        # Encode all mentions
        mention_embeddings = self.encode_mentions(mentions)
        
        # Search in batch
        scores, indices = self.index.search(mention_embeddings, top_k)
        
        # Format results
        results = {}
        for i, mention in enumerate(mentions):
            mention_results = []
            for score, idx in zip(scores[i], indices[i]):
                if idx != -1:
                    mondo_id = str(self.labels[idx])  # Convert numpy string to Python string
                    mention_results.append((mondo_id, float(score)))
            results[mention] = mention_results
        
        return results

def evaluate_entity_linking(linker: SapBERTEntityLinker, 
                          test_file: str = "data/mondo_test.csv",
                          top_k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """Evaluate entity linking performance."""
    print(f"Loading test data from {test_file}")
    test_df = pd.read_csv(test_file)
    
    print(f"Evaluating on {len(test_df)} test examples")
    
    # Get predictions for all mentions
    mentions = test_df['mention'].tolist()
    true_labels = test_df['mondo_id'].tolist()
    
    max_k = max(top_k_values)
    predictions = linker.predict_batch(mentions, top_k=max_k)
    
    # Calculate metrics
    metrics = {}
    
    # Hits@K
    for k in top_k_values:
        hits = 0
        for mention, true_label in zip(mentions, true_labels):
            if mention in predictions:
                pred_labels = [pred[0] for pred in predictions[mention][:k]]
                if true_label in pred_labels:
                    hits += 1
        
        hits_at_k = hits / len(test_df)
        metrics[f'Hits@{k}'] = hits_at_k
        print(f"Hits@{k}: {hits_at_k:.4f}")
    
    # MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for mention, true_label in zip(mentions, true_labels):
        if mention in predictions:
            pred_labels = [pred[0] for pred in predictions[mention]]
            if true_label in pred_labels:
                rank = pred_labels.index(true_label) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks)
    metrics['MRR'] = mrr
    print(f"MRR: {mrr:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="SapBERT + FAISS Entity Linking Inference")
    parser.add_argument("--models", default="models", help="Directory with model files")
    parser.add_argument("--test", default="data/mondo_test.csv", help="Test file for evaluation")
    parser.add_argument("--eval", action="store_true", help="Run evaluation on test set")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top predictions to return")
    
    args = parser.parse_args()
    
    # Check if model files exist
    model_dir = Path(args.models)
    if not (model_dir / "sapbert_mondo.faiss").exists():
        print(f"Error: FAISS index not found in {model_dir}")
        print("Please run build_sapbert_index.py first to create the index")
        return
    
    # Initialize entity linker
    try:
        linker = SapBERTEntityLinker(args.models)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run evaluation
    if args.eval:
        if not Path(args.test).exists():
            print(f"Error: Test file {args.test} not found")
            return
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        metrics = evaluate_entity_linking(linker, args.test)
        
        print("\nFinal Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Run interactive demo
    if args.demo:
        print("\n" + "="*50)
        print("SapBERT + FAISS Entity Linking Demo")
        print("="*50)
        
        example_mentions = [
            "cystic fibrosis",
            "diabetes mellitus",
            "alzheimer disease",
            "breast cancer",
            "heart attack"
        ]
        
        print("\nExample predictions:")
        for mention in example_mentions:
            results = linker.search(mention, top_k=3)
            print(f"\nMention: '{mention}'")
            print("Top 3 MONDO predictions:")
            for i, (mondo_id, score) in enumerate(results, 1):
                print(f"  {i}. {mondo_id} (score: {score:.4f})")
        
        # Interactive input
        print("\n" + "="*30)
        print("Enter your own mentions:")
        print("="*30)
        
        while True:
            try:
                mention = input("\nEnter a medical mention (or 'quit' to exit): ").strip()
                if mention.lower() in ['quit', 'exit', 'q']:
                    break
                
                if mention:
                    results = linker.search(mention, top_k=args.top_k)
                    print(f"\nTop {args.top_k} predictions for '{mention}':")
                    for i, (mondo_id, score) in enumerate(results, 1):
                        print(f"  {i}. {mondo_id} (score: {score:.4f})")
                        
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    # Default behavior if no flags
    if not args.eval and not args.demo:
        print("Use --eval to run evaluation or --demo for interactive demo")
        print("Example: python inference_sapbert.py --eval --demo")

if __name__ == "__main__":
    main() 