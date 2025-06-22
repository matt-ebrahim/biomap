# Entity Linking Benchmark

This repository contains tools and benchmarks for entity linking tasks using modern machine learning approaches.

## Environment Setup

We use conda for environment management instead of traditional Python virtual environments. This ensures better dependency management and easier GPU support configuration.

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.8 or higher

### Environment Files

We provide three conda environment configurations in the `environments/` directory:

- **`environments/environment.yaml`** - GPU-enabled version with CUDA support (for NVIDIA GPUs)
- **`environments/environment-cpu.yaml`** - CPU-only version (also supports Apple Silicon MPS)
- **`environments/environment-mac.yaml`** - Mac-optimized version that avoids common FAISS dependency issues

### Installation Instructions

#### For NVIDIA GPU Systems (Linux/Windows with CUDA)
```bash
conda env create -f environments/environment.yaml
conda activate biomap-env
```

#### For CPU-Only Systems
```bash
conda env create -f environments/environment-cpu.yaml
conda activate biomap-env
```

#### For Mac Users (Recommended)
```bash
conda env create -f environments/environment-mac.yaml
conda activate biomap-env
```

> **Note for Mac Users**: We recommend using `environment-mac.yaml` as it's specifically optimized to avoid common FAISS dependency issues on macOS. If you have an Apple Silicon Mac (M1/M2/M3), this environment will automatically include MPS (Metal Performance Shaders) support for GPU acceleration.

### Verifying Your Installation

After creating and activating the environment, verify your setup:

#### Check PyTorch Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

#### Check GPU Support
```bash
# For NVIDIA CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# For Apple MPS
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### Check Other Dependencies
```bash
python -c "import transformers, datasets, faiss; print('All core dependencies loaded successfully')"
```

### Key Dependencies

- **PyTorch** (≥2.0) - Deep learning framework
- **Transformers** (≥4.40) - Hugging Face transformers library
- **Datasets** - Hugging Face datasets library
- **FAISS** - Efficient similarity search and clustering
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning utilities
- **OpenAI** - OpenAI API client
- **Google Generative AI** - Google's generative AI API client

### Updating the Environment

To update the environment with new dependencies:

1. Modify the appropriate YAML file
2. Update the environment:
   ```bash
   conda env update -f environments/environment.yaml --prune
   ```

### Removing the Environment

To completely remove the environment:
```bash
conda env remove -n biomap-env
```

## Troubleshooting

### FAISS ImportError on Mac

If you encounter an error like:
```
ImportError: dlopen(.../_swigfaiss.so, 0x0002): Library not loaded: @rpath/libmkl_intel_lp64.1.dylib
```

This is a common issue with FAISS on Mac. Try these solutions:

#### Solution 1: Use the Mac-optimized environment
```bash
conda env remove -n biomap-env
conda env create -f environments/environment-mac.yaml
conda activate biomap-env
```

#### Solution 2: Fix existing environment
```bash
conda activate biomap-env
conda install mkl
```

#### Solution 3: Use pip for FAISS
```bash
conda activate biomap-env
pip uninstall faiss-cpu
pip install faiss-cpu
```

## Dataset: MedMentions ST21pv

This project uses the MedMentions ST21pv corpus, a high-quality biomedical entity linking dataset with UMLS annotations.

### Dataset Overview

- **Source**: Chan Zuckerberg Initiative MedMentions corpus
- **Subset**: ST21pv (21 Semantic Types and Preferred Vocabularies)
- **Format**: PubTator format
- **Entity IDs**: UMLS concepts (can be mapped to MONDO IDs)
- **Content**: 4,392 PubMed abstracts with exhaustive entity annotations

### Download Instructions

```bash
# Clone the official MedMentions repository
git clone https://github.com/chanzuckerberg/MedMentions.git

# Rename to match project structure
mv MedMentions medmentions

# Extract the ST21pv corpus data
cd medmentions/st21pv/data
gunzip corpus_pubtator.txt.gz
```

### Dataset Structure

The corpus is in PubTator format with the following structure:

```
PMID|t|Title text
PMID|a|Abstract text
PMID    StartIndex    EndIndex    MentionText    SemanticType    EntityID
...
[blank line separating documents]
```

#### Example Entry:
```
25763772|t|DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection in cystic fibrosis
25763772|a|Pseudomonas aeruginosa (Pa) infection in cystic fibrosis (CF) patients is associated with...
25763772    0    5    DCTN4    T103    UMLS:C4308010
25763772    23   63   chronic Pseudomonas aeruginosa infection    T038    UMLS:C0854135
25763772    67   82   cystic fibrosis    T038    UMLS:C0010674
```

### Field Descriptions

- **PMID**: PubMed ID of the paper
- **StartIndex/EndIndex**: Character positions in the concatenated title + " " + abstract
- **MentionText**: The actual text span of the entity mention
- **SemanticType**: UMLS semantic type (e.g., T103 = "Amino Acid, Peptide, or Protein")
- **EntityID**: UMLS concept identifier (format: UMLS:CXXXXXXX)

### Dataset Statistics

- **Total lines**: ~216,458
- **Documents**: 4,392 PubMed abstracts
- **Entity mentions**: Exhaustively annotated biomedical entities
- **Entity types**: 21 UMLS semantic types relevant to biomedical information retrieval

### Citation

If you use the MedMentions dataset, please cite:

```bibtex
@inproceedings{mohan2019medmentions,
  title={MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts},
  author={Mohan, Sunil and Li, Donghui},
  booktitle={Proceedings of the 2019 Conference on Automated Knowledge Base Construction (AKBC 2019)},
  year={2019},
  address={Amherst, Massachusetts, USA}
}
```

### Notes

- The original S3 download link (`https://s3.amazonaws.com/medmentions/data/pubtator-mentions.tar.gz`) is no longer available
- Use the GitHub repository as the official source
- For MONDO ID mapping, see the next section for creating UMLS → MONDO mappings

## UMLS → MONDO Crosswalk Mapping

To enable entity linking with MONDO disease ontology IDs, you need to create a mapping from UMLS concepts (used in MedMentions) to MONDO IDs.

### Why This Mapping is Needed

The MedMentions dataset uses UMLS concept identifiers (e.g., `UMLS:C0010674`), but many biomedical applications prefer MONDO IDs for disease concepts. This crosswalk enables conversion between the two systems.

### Download and Process MONDO Equivalencies

```bash
# Download the latest MONDO equivalencies file
curl -L -o equivalencies.json "https://github.com/monarch-initiative/mondo/releases/download/v2025-06-03/equivalencies.json"

# Extract UMLS to MONDO mappings using Python
python3 -c "
import json
import re

# Load the equivalencies JSON
with open('equivalencies.json', 'r') as f:
    data = json.load(f)

# Extract UMLS to MONDO mappings
umls_to_mondo = {}

# Look through the graphs for equivalent node sets
for graph in data['graphs']:
    if 'equivalentNodesSets' in graph:
        for equiv_set in graph['equivalentNodesSets']:
            if 'nodeIds' in equiv_set:
                umls_ids = []
                mondo_ids = []
                
                # Find UMLS and MONDO IDs in the equivalent set
                for node_id in equiv_set['nodeIds']:
                    if 'umls/id/' in node_id:
                        umls_match = re.search(r'umls/id/([A-Z0-9]+)', node_id)
                        if umls_match:
                            umls_ids.append(umls_match.group(1))
                    elif 'MONDO_' in node_id:
                        mondo_match = re.search(r'MONDO_([0-9]+)', node_id)
                        if mondo_match:
                            mondo_ids.append(f'MONDO:{mondo_match.group(1)}')
                
                # Create mappings for all combinations
                for umls_id in umls_ids:
                    for mondo_id in mondo_ids:
                        umls_to_mondo[umls_id] = mondo_id

print(f'Found {len(umls_to_mondo)} UMLS to MONDO mappings')

# Write to TSV file
with open('umls2mondo.tsv', 'w') as f:
    for umls_cui, mondo_id in sorted(umls_to_mondo.items()):
        f.write(f'{umls_cui}\t{mondo_id}\n')

print('Saved to umls2mondo.tsv')
"

# Clean up the large JSON file
rm equivalencies.json
```

### Mapping File Format

The resulting `umls2mondo.tsv` file contains tab-separated mappings:

```
C0000744    MONDO:0008692
C0000774    MONDO:0001770
C0000832    MONDO:0004846
C0010674    MONDO:0009061
```

**Format**: `UMLS_CUI <TAB> MONDO_ID`

### Mapping Statistics

- **Total mappings**: ~11,116 UMLS CUI → MONDO ID pairs
- **File size**: ~250KB
- **Coverage**: Verified overlap with MedMentions dataset
- **Source**: Latest MONDO release equivalencies

### Alternative: Download Pre-processed Mapping

If the Python processing fails, you can try alternative sources:

```bash
# Note: The original monarch-dumps URL may be outdated
# curl -L -O https://github.com/monarch-initiative/monarch-dumps/raw/master/monarch-umlscui-mappings.tsv
# cut -f1,2 monarch-umlscui-mappings.tsv > umls2mondo.tsv
```

### Usage in Entity Linking

```python
# Example: Convert UMLS IDs from MedMentions to MONDO IDs
umls_to_mondo = {}
with open('umls2mondo.tsv', 'r') as f:
    for line in f:
        umls_cui, mondo_id = line.strip().split('\t')
        umls_to_mondo[umls_cui] = mondo_id

# Convert UMLS:C0010674 to MONDO:0009061
umls_id = "C0010674"  # From MedMentions
mondo_id = umls_to_mondo.get(umls_id)
print(f"{umls_id} -> {mondo_id}")
```

## Dataset Preparation

After downloading the MedMentions corpus and creating the UMLS→MONDO mapping, you need to prepare benchmark splits for training and evaluation.

### Preparation Script

The `prepare_dataset.py` script processes the raw MedMentions data and creates train/dev/test splits with MONDO IDs:

```bash
python3 prepare_dataset.py
```

### What the Script Does

1. **Loads UMLS→MONDO mappings** from `umls2mondo.tsv`
2. **Processes MedMentions corpus** from `medmentions/st21pv/data/corpus_pubtator.txt`
3. **Filters entities** to include only those with MONDO mappings
4. **Removes duplicates** to ensure unique mention-MONDO pairs
5. **Creates deterministic splits** using `random_state=42`:
   - **Train**: 80% of data
   - **Dev**: 10% of data  
   - **Test**: 10% of data
6. **Saves CSV files** in the `data/` directory

### Output Files

```
data/
├── mondo_train.csv    # Training set (~80%)
├── mondo_dev.csv      # Development set (~10%)
└── mondo_test.csv     # Test set (~10%)
```

### CSV Format

Each CSV file contains the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `pmid` | PubMed ID of the paper | `25763772` |
| `mention` | Text span of the entity mention | `cystic fibrosis` |
| `cui` | UMLS Concept Unique Identifier | `C0010674` |
| `mondo_id` | Corresponding MONDO disease ID | `MONDO:0009061` |

### Example Data

```csv
pmid,mention,cui,mondo_id
25763772,cystic fibrosis,C0010674,MONDO:0009061
25763772,chronic Pseudomonas aeruginosa infection,C0854135,MONDO:0005709
```

### Statistics

The script provides detailed statistics during processing:

- **Total entities processed**: Number of UMLS entities in MedMentions
- **Entities with MONDO mappings**: Subset that can be mapped to MONDO
- **Coverage percentage**: Proportion of entities with MONDO mappings
- **Unique mention-MONDO pairs**: Final dataset size after deduplication
- **Split sizes**: Exact counts and percentages for each split

### Usage Notes

- **Deterministic splits**: Using `random_state=42` ensures reproducible results
- **Deduplication**: Removes duplicate mention-MONDO pairs across the corpus
- **MONDO-only filtering**: Only includes entities that have valid MONDO mappings
- **Directory creation**: Automatically creates `data/` directory if it doesn't exist

### Prerequisites

Before running the script, ensure you have:

1. ✅ Downloaded MedMentions dataset (`medmentions/st21pv/data/corpus_pubtator.txt`)
2. ✅ Created UMLS→MONDO mapping (`umls2mondo.tsv`)
3. ✅ Installed required Python packages:
   ```bash
   pip install pandas
   ```

## Baseline Models

### BioMegatron Fine-tuned Classifier

We provide a baseline implementation using NVIDIA's BioMegatron model fine-tuned for entity linking classification.

#### 🔧 **Critical Fixes Applied**

**Problem Identified**: The original implementation had a critical training bug where:
- Negative sampling used random permutation, creating false negatives (correct pairs labeled as incorrect)
- Model learned inverted associations (malaria→mumps scored higher than malaria→malaria)
- Used meaningless MONDO codes instead of medical terms for training

**Solutions Implemented**:
1. **Proper Negative Sampling**: Ensures negative pairs are truly incorrect (never valid mention-MONDO combinations)
2. **Semantic Text Matching**: Uses representative mention text instead of MONDO codes for meaningful semantic learning
3. **Improved Training Parameters**: Reduced learning rate, added warmup, increased epochs for better convergence

**Performance Impact**: Fixed training should achieve ~60-80% accuracy instead of inverted/poor performance.

#### Training

The training script fine-tunes BioMegatron to rank MONDO candidates for given mentions:

```bash
python3 train_biomegatron_cls.py
```

**What the training script does:**

1. **Loads the pre-trained model**: Uses `EMBO/BioMegatron345mUncased` (community-uploaded BioMegatron)
2. **Creates representative mention mappings**: Maps each MONDO ID to its most frequent mention text
3. **Creates training pairs with proper semantic matching**: 
   - **Positive pairs**: mention + [SEP] + representative_mention_text (label=1.0)
   - **Negative pairs**: mention + [SEP] + different_representative_text (label=0.0)
   - **✅ FIXED**: Uses proper negative sampling to ensure negatives are truly incorrect
4. **Fine-tunes for regression**: Model learns to score mention-text pairs semantically
5. **Saves trained model**: Stores in `models/biomegatron_mondo_cls_final/`

**Training configuration:**
- **Epochs**: 5 (increased for better learning)
- **Batch size**: 16 per device
- **Learning rate**: 1e-5 (reduced for stability)
- **Max sequence length**: 64 tokens
- **Evaluation**: Every epoch with early stopping
- **Improvements**: Added warmup steps and weight decay for better convergence

**Note**: Trained models are not included in the git repository due to size constraints. You need to train the model first using the training script above.

#### Inference

Use the trained model to rank MONDO candidates:

```bash
python3 inference_biomegatron.py
```

**Programmatic usage:**

```python
from inference_biomegatron import BioMegatronEntityLinker

# Initialize entity linker
linker = BioMegatronEntityLinker()

# Get top-5 MONDO predictions for a mention
predictions = linker.predict("cystic fibrosis", top_k=5)

# Results: [(mondo_id, confidence_score), ...]
for mondo_id, score in predictions:
    print(f"{mondo_id}: {score:.4f}")
```

#### Model Architecture

- **Base model**: BioMegatron-BERT (345M parameters)
- **Task**: Binary classification/regression for mention-text pairs
- **Input format**: `"mention [SEP] representative_mention_text"`
- **Output**: Confidence score (0-1) for semantic similarity
- **✅ FIXED**: Now uses meaningful medical terms instead of MONDO codes for better semantic matching

#### Requirements

Additional packages needed for training:

```bash
# Install transformers and torch
pip install torch transformers datasets
```

For CUDA support, install the appropriate PyTorch version for your system.

### SapBERT + FAISS Vector Search

We provide a second baseline using SapBERT embeddings with FAISS for fast similarity search.

#### Building the Index

First, build the FAISS index from MONDO entities:

```bash
python3 build_sapbert_index.py
```

**What the indexing script does:**

1. **Loads SapBERT model**: Uses `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
2. **Creates representative mention mappings**: Maps each MONDO ID to its most frequent mention text
3. **Encodes medical terms**: Converts representative mention text (not MONDO codes) to embeddings
4. **Builds FAISS index**: Creates IndexFlatIP for cosine similarity search
5. **Saves artifacts**: Stores index and label mappings in `models/`
6. **✅ FIXED**: Uses meaningful medical terms for embeddings instead of meaningless MONDO codes

**Generated files:**
- `models/sapbert_mondo.faiss` - FAISS index file
- `models/sapbert_mondo_labels.npy` - MONDO ID labels
- `models/sapbert_metadata.txt` - Index metadata

#### Inference and Evaluation

Run inference with automatic evaluation:

```bash
# Evaluation only
python3 inference_sapbert.py --eval

# Interactive demo only  
python3 inference_sapbert.py --demo

# Both evaluation and demo
python3 inference_sapbert.py --eval --demo
```

**Evaluation metrics:**
- **Hits@K**: Percentage of correct predictions in top-K results
- **MRR**: Mean Reciprocal Rank of the correct prediction

**Programmatic usage:**

```python
from inference_sapbert import SapBERTEntityLinker

# Initialize entity linker
linker = SapBERTEntityLinker(model_dir="models")

# Search for similar MONDO entities
results = linker.search("cystic fibrosis", top_k=5)

# Results: [(mondo_id, similarity_score), ...]
for mondo_id, score in results:
    print(f"{mondo_id}: {score:.4f}")
```

#### Model Architecture

- **Base model**: SapBERT (PubMedBERT-fulltext)
- **Embedding dimension**: 768
- **Similarity metric**: Cosine similarity (normalized L2)
- **Search method**: FAISS IndexFlatIP for exact search
- **Max sequence length**: 25 tokens

#### Performance Characteristics

- **Speed**: Very fast inference (~1ms per query)
- **Memory**: Requires loading full MONDO embedding matrix
- **Accuracy**: Depends on semantic similarity of embeddings
- **Scalability**: Easily scales to large knowledge bases

#### Requirements

Additional packages needed:

```bash
# Install FAISS (CPU version)
pip install faiss-cpu

# Or for GPU version
pip install faiss-gpu

# Other dependencies
pip install torch transformers tqdm
```

### Zero-shot LLM Evaluation

We provide a third baseline using large language models (GPT-4o and Gemini) for zero-shot entity linking evaluation.

#### Setup

The LLM evaluation requires API keys for both OpenAI and Gemini. Set these as environment variables:

```bash
# Set API keys as environment variables
export OPENAI_API_KEY="your_openai_api_key_here"
export GOOGLE_API_KEY="your_google_api_key_here"

# Make environment variables permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="your_openai_api_key_here"' >> ~/.bashrc
echo 'export GOOGLE_API_KEY="your_google_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

The script automatically handles rate limiting and error recovery.

**Important**: The script will validate that both environment variables are set before running. If either is missing, it will show an error message.

#### Running LLM Evaluation

```bash
# Evaluate with GPT-4o
python3 llm_zero_shot.py --model gpt --test data/mondo_test.csv

# Evaluate with Gemini  
python3 llm_zero_shot.py --model gemini --test data/mondo_test.csv

# Interactive demo
python3 llm_zero_shot.py --demo --model gpt

# Limit evaluation to 100 samples
python3 llm_zero_shot.py --model gpt --max_samples 100
```

#### Core Function

The main evaluation function provides a unified interface:

```python
from llm_zero_shot import llm_link, initialize_clients

# Initialize API clients
initialize_clients()

# Evaluate a single mention-MONDO pair
is_equivalent, explanation = llm_link("cystic fibrosis", "MONDO:0009061", model="gpt")

print(f"Equivalent: {is_equivalent}")
print(f"Explanation: {explanation}")
```

#### Evaluation Features

- **Dual model support**: Both GPT-4o and Gemini models
- **Parallel processing**: Concurrent API calls with rate limiting
- **Error handling**: Automatic retry with exponential backoff
- **Checkpointing**: Saves progress every 50 evaluations
- **Detailed output**: JSON results with explanations

#### Evaluation Metrics

- **Accuracy**: Percentage of correct equivalence judgments
- **Error rate**: Proportion of API errors/failures
- **Detailed explanations**: Natural language reasoning for each decision

#### Prompt Engineering

The prompts are designed for medical entity linking:

**GPT-4o Prompt:**
- Focuses on clinical equivalents and medical terminology
- Considers different specificity levels
- Handles abbreviations and formal terms

**Gemini Prompt:**
- Emphasizes medical expertise
- Includes ICD classification considerations
- Leverages medical knowledge base

#### Parallel Processing

- **Concurrent requests**: Up to 16 workers (configurable)
- **Request delays**: Optimized 0.1 seconds between calls
- **Retry logic**: Exponential backoff for rate limits
- **Checkpointing**: Every 100 evaluations
- **Thread-local clients**: Efficient resource management

#### Output Format

Results are saved as JSON with detailed information:

```json
{
  "metrics": {
    "accuracy": 0.85,
    "total_samples": 100,
    "correct_predictions": 85,
    "errors": 2,
    "error_rate": 0.02,
    "model": "gpt"
  },
  "predictions": [
    {
      "mention": "cystic fibrosis",
      "true_mondo": "MONDO:0009061", 
      "prediction": true,
      "explanation": "YES - Both terms refer to the same genetic disorder...",
      "correct": true
    }
  ]
}
```

#### Requirements

```bash
pip install openai google-generativeai pandas tqdm
```

## Unified Evaluation

We provide a comprehensive evaluation script that tests all baseline models and provides direct comparison.

### Running Complete Evaluation

```bash
# Evaluate all models on full test set
python3 evaluate_all.py

# Quick evaluation with limited samples
python3 evaluate_all.py --max_samples 100

# Evaluate specific models only
python3 evaluate_all.py --skip_gpt --skip_gemini  # Skip LLM models
python3 evaluate_all.py --skip_sapbert --skip_biomegatron  # Skip ML models

# Control parallel processing threads  
python3 evaluate_all.py --max_workers 16

# Custom output file
python3 evaluate_all.py --output my_results.json
```

### Evaluation Features

- **Comprehensive metrics**: Hits@1, Hits@3, Hits@5, Hits@10, and MRR
- **All baselines**: SapBERT, BioMegatron, GPT-4o, and Gemini
- **High-performance parallel processing**: Optimized multi-threading for LLM evaluation
- **Flexible options**: Skip specific models or limit test samples
- **Formatted output**: Comparison table and JSON results

### Expected Output

The script produces a comprehensive comparison table:

```
================================================================================
📊 COMPREHENSIVE EVALUATION RESULTS
================================================================================
Model                   Hits@1      Hits@3      Hits@5     Hits@10         MRR
--------------------------------------------------------------------------------
SapBERT + FAISS         0.4250      0.6100      0.7200      0.8150      0.5375
BioMegatron             0.3900      0.5750      0.6850      0.7800      0.4925
GPT-4o                  0.4800      0.6400      0.7300      0.8200      0.5950
Gemini                  0.4600      0.6200      0.7100      0.8000      0.5650
================================================================================
```

### Evaluation Strategy

#### SapBERT + FAISS
- **Approach**: Fast similarity search using pre-computed embeddings
- **Speed**: Very fast (~1ms per query)
- **Coverage**: All MONDO entities in knowledge base

#### BioMegatron Classifier  
- **Approach**: Fine-tuned transformer for mention-entity classification
- **Speed**: Moderate (~100ms per query)
- **Coverage**: All MONDO entities, with learned ranking

#### GPT-4o Zero-shot
- **Approach**: Large language model with medical reasoning
- **Speed**: Optimized with parallel processing (up to 16 concurrent threads)
- **Coverage**: Complete MONDO entity evaluation

#### Gemini Search-grounded
- **Approach**: LLM with search capabilities for medical validation
- **Speed**: Optimized with parallel processing (up to 16 concurrent threads)
- **Coverage**: Complete MONDO entity evaluation

### Performance Optimization

LLM evaluations are optimized for high-throughput processing:

- **Parallel processing**: Multi-threaded evaluation with configurable worker count
- **Efficient API usage**: Optimized rate limiting and retry logic
- **Skip options**: Skip specific models during development
- **Sample limits**: Test on subset of data for quick validation

### JSON Output Format

Results are saved with detailed breakdown:

```json
{
  "SapBERT + FAISS": {
    "Hits@1": 0.4250,
    "Hits@3": 0.6100,
    "Hits@5": 0.7200,
    "Hits@10": 0.8150,
    "MRR": 0.5375
  },
  "BioMegatron": {
    "Hits@1": 0.3900,
    "Hits@3": 0.5750,
    "Hits@5": 0.6850,
    "Hits@10": 0.7800,
    "MRR": 0.4925
  }
}
```

## Project Structure

```
entity-linking-benchmark/
├── README.md                              # This documentation
├── environments/
│   ├── environment.yaml                   # CUDA-enabled environment
│   ├── environment-cpu.yaml              # CPU/MPS environment  
│   └── environment-mac.yaml              # Mac-optimized environment
├── medmentions/st21pv/data/
│   └── corpus_pubtator.txt               # MedMentions dataset
├── umls2mondo.tsv                        # UMLS→MONDO mappings (11,116 entries)
├── prepare_dataset.py                    # Dataset preparation script
├── train_biomegatron_cls.py              # BioMegatron training script
├── inference_biomegatron.py              # BioMegatron inference script
├── build_sapbert_index.py                # SapBERT + FAISS index builder
├── inference_sapbert.py                  # SapBERT + FAISS inference script
├── llm_zero_shot.py                      # Zero-shot LLM evaluation (GPT-4o/Gemini)
├── evaluate_all.py                       # Unified evaluation script for all models
├── data/                                 # Generated benchmark datasets
│   ├── mondo_train.csv                   # Training split
│   ├── mondo_dev.csv                     # Development split  
│   └── mondo_test.csv                    # Test split
└── models/                               # Trained models and indices
    ├── biomegatron_mondo_cls_final/      # Fine-tuned BioMegatron model
    ├── sapbert_mondo.faiss               # SapBERT FAISS index
    ├── sapbert_mondo_labels.npy          # MONDO ID labels for FAISS
    └── sapbert_metadata.txt              # SapBERT index metadata
```

## Usage

*This section will be updated with usage instructions as features are implemented.*

## Contributing

*This section will be updated with contribution guidelines.*

## License

*License information will be added.*

---

*This README will be updated as the project evolves and new features are added.* 