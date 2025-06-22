#!/usr/bin/env python3
"""
Build SapBERT + FAISS index for entity linking.

This script:
1. Loads SapBERT model for biomedical entity embeddings
2. Encodes all MONDO IDs from the training data
3. Builds a FAISS index for fast similarity search
4. Saves the index and label mappings for inference
"""

from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import faiss
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

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

def load_model(device='cuda'):
    """Load SapBERT model and tokenizer."""
    print("Loading SapBERT model...")
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    
    print(f"Model loaded on {device}")
    return tokenizer, model, device

def encode_texts(texts, tokenizer, model, device, batch_size=32, max_length=25):
    """Encode texts to embeddings using SapBERT."""
    all_embeddings = []
    
    print(f"Encoding {len(texts)} texts...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt",
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = model(**inputs).pooler_output
        
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

def build_mondo_index(data_file="data/mondo_train.csv", output_dir="models"):
    """Build FAISS index for MONDO entities using representative mention text."""
    # Setup
    device = setup_device()
    tokenizer, model, device = load_model(device)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load training data and create MONDO ID -> representative mention mapping
    print(f"Loading MONDO IDs from {data_file}")
    df = pd.read_csv(data_file)
    
    # For each MONDO ID, find the most representative mention (e.g., most common)
    mondo_to_mentions = df.groupby('mondo_id')['mention'].apply(list).to_dict()
    
    # Use the first mention as representative (could be improved with frequency analysis)
    mondo_to_text = {}
    for mondo_id, mentions in mondo_to_mentions.items():
        # Use the most frequent mention for this MONDO ID
        mention_counts = pd.Series(mentions).value_counts()
        representative_mention = mention_counts.index[0]  # Most common
        mondo_to_text[mondo_id] = representative_mention
    
    mondo_ids = sorted(mondo_to_text.keys())
    mondo_texts = [mondo_to_text[mondo_id] for mondo_id in mondo_ids]
    
    print(f"Found {len(mondo_ids)} unique MONDO IDs")
    print("Sample mappings:")
    for i in range(min(5, len(mondo_ids))):
        print(f"  {mondo_ids[i]} -> '{mondo_texts[i]}'")
    
    # Encode MONDO texts (using representative mentions)
    mondo_embeddings = encode_texts(mondo_texts, tokenizer, model, device)
    print(f"Generated embeddings shape: {mondo_embeddings.shape}")
    
    # Build FAISS index
    print("Building FAISS index...")
    embedding_dim = mondo_embeddings.shape[1]
    
    # Use Inner Product for similarity (SapBERT embeddings are normalized)
    index = faiss.IndexFlatIP(embedding_dim)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(mondo_embeddings)
    index.add(mondo_embeddings.astype(np.float32))
    
    print(f"FAISS index built with {index.ntotal} entries")
    
    # Save index and labels
    index_file = output_path / "sapbert_mondo.faiss"
    labels_file = output_path / "sapbert_mondo_labels.npy"
    
    print(f"Saving FAISS index to {index_file}")
    faiss.write_index(index, str(index_file))
    
    print(f"Saving MONDO labels to {labels_file}")
    np.save(str(labels_file), np.array(mondo_ids))
    
    # Save additional metadata including the mention mappings
    metadata = {
        'model_name': "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        'embedding_dim': embedding_dim,
        'num_entities': len(mondo_ids),
        'max_length': 25,
        'normalization': 'L2',
        'indexing_method': 'representative_mentions'  # New field to indicate the fix
    }
    
    metadata_file = output_path / "sapbert_metadata.txt"
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Saved metadata to {metadata_file}")
    print("\n" + "="*50)
    print("SapBERT FAISS index building complete!")
    print("="*50)
    print(f"Index file: {index_file}")
    print(f"Labels file: {labels_file}")
    print(f"Entities indexed: {len(mondo_ids)}")
    print(f"Embedding dimension: {embedding_dim}")
    print("🔧 FIXED: Using representative mention text instead of MONDO IDs for embeddings")
    
    return index_file, labels_file

def main():
    parser = argparse.ArgumentParser(description="Build SapBERT + FAISS index for MONDO entities")
    parser.add_argument("--data", default="data/mondo_train.csv", 
                       help="CSV file with MONDO IDs (default: data/mondo_train.csv)")
    parser.add_argument("--output", default="models",
                       help="Output directory for index files (default: models)")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data).exists():
        print(f"Error: Data file {args.data} not found")
        print("Please run prepare_dataset.py first to create the dataset")
        return
    
    # Build index
    build_mondo_index(args.data, args.output)

if __name__ == "__main__":
    main() 