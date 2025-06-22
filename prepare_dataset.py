#!/usr/bin/env python3
"""
Prepare benchmark splits from MedMentions dataset with MONDO ID mappings.

This script:
1. Loads UMLS→MONDO mappings
2. Processes MedMentions corpus
3. Filters for entities with MONDO mappings
4. Creates train/dev/test splits (80/10/10)
5. Saves as CSV files
"""

import pandas as pd
import json
import pathlib
import random
import re
import gzip

# Load UMLS to MONDO mapping
print("Loading UMLS→MONDO mappings...")
umls2mondo = dict(line.strip().split('\t') for line in open('umls2mondo.tsv'))
print(f"Loaded {len(umls2mondo)} UMLS→MONDO mappings")

# Process MedMentions corpus
print("Processing MedMentions corpus...")
records = []
total_entities = 0
mondo_entities = 0

with open('medmentions/st21pv/data/corpus_pubtator.txt') as f:
    for line in f:
        # Skip title/abstract lines and empty lines
        if line.startswith('###') or '\t' not in line or '|t|' in line or '|a|' in line:
            continue
        
        parts = line.strip().split('\t')
        if len(parts) < 6:
            continue
            
        pmid, start, end, text, semantic_type, entity_id = parts[:6]
        
        # Extract CUI from entity_id (format: UMLS:CXXXXXXX)
        if entity_id.startswith('UMLS:'):
            cui = entity_id.replace('UMLS:', '')
            total_entities += 1
            
            if cui in umls2mondo:  # Only include entities with MONDO mappings
                records.append({
                    'pmid': pmid,
                    'mention': text,
                    'cui': cui,
                    'mondo_id': umls2mondo[cui]
                })
                mondo_entities += 1

print(f"Total entities processed: {total_entities}")
print(f"Entities with MONDO mappings: {mondo_entities}")
print(f"Coverage: {mondo_entities/total_entities*100:.1f}%")

# Create DataFrame and remove duplicates
df = pd.DataFrame(records).drop_duplicates()
print(f"Unique mention-MONDO pairs: {len(df)}")

# Shuffle deterministically
df = df.sample(frac=1, random_state=42)

# Create splits (80% train, 10% dev, 10% test)
n = len(df)
train_end = int(0.8 * n)
dev_end = int(0.9 * n)

train = df.iloc[:train_end]
dev = df.iloc[train_end:dev_end]
test = df.iloc[dev_end:]

print(f"\nDataset splits:")
print(f"Train: {len(train)} examples ({len(train)/n*100:.1f}%)")
print(f"Dev:   {len(dev)} examples ({len(dev)/n*100:.1f}%)")
print(f"Test:  {len(test)} examples ({len(test)/n*100:.1f}%)")

# Save splits to CSV files
for split_name, split_df in [('train', train), ('dev', dev), ('test', test)]:
    filename = f'mondo_{split_name}.csv'
    split_df.to_csv(filename, index=False)
    print(f"Saved {filename}")

print("\nDataset preparation complete!")
print("\nOutput files:")
print("- mondo_train.csv  (~80%)")
print("- mondo_dev.csv    (~10%)")
print("- mondo_test.csv   (~10%)")
print("\nEach row contains: pmid, mention, cui, mondo_id") 