# Entity Linking Benchmark

This repository contains tools and benchmarks for entity linking tasks using modern machine learning approaches.

## Environment Setup

We use conda for environment management instead of traditional Python virtual environments. This ensures better dependency management and easier GPU support configuration.

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.8 or higher

### Environment Files

We provide three conda environment configurations:

- **`environment.yaml`** - GPU-enabled version with CUDA support (for NVIDIA GPUs)
- **`environment-cpu.yaml`** - CPU-only version (also supports Apple Silicon MPS)
- **`environment-mac.yaml`** - Mac-optimized version that avoids common FAISS dependency issues

### Installation Instructions

#### For NVIDIA GPU Systems (Linux/Windows with CUDA)
```bash
conda env create -f environment.yaml
conda activate biomap-env
```

#### For CPU-Only Systems
```bash
conda env create -f environment-cpu.yaml
conda activate biomap-env
```

#### For Mac Users (Recommended)
```bash
conda env create -f environment-mac.yaml
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
   conda env update -f environment.yaml --prune
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
conda env create -f environment-mac.yaml
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

## Project Structure

*This section will be updated as the project develops.*

## Usage

*This section will be updated with usage instructions as features are implemented.*

## Contributing

*This section will be updated with contribution guidelines.*

## License

*License information will be added.*

---

*This README will be updated as the project evolves and new features are added.* 