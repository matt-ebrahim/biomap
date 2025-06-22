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