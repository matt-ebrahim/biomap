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
- For MONDO ID mapping, you'll need to create UMLS → MONDO mappings separately

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