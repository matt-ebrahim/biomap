# Entity Linking Benchmark

This repository contains tools and benchmarks for entity linking tasks using modern machine learning approaches.

## Environment Setup

We use conda for environment management instead of traditional Python virtual environments. This ensures better dependency management and easier GPU support configuration.

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.8 or higher

### Environment Files

We provide two conda environment configurations:

- **`environment.yaml`** - GPU-enabled version with CUDA support (for NVIDIA GPUs)
- **`environment-cpu.yaml`** - CPU-only version (also supports Apple Silicon MPS)

### Installation Instructions

#### For NVIDIA GPU Systems (Linux/Windows with CUDA)
```bash
conda env create -f environment.yaml
conda activate biomap-env
```

#### For CPU-Only Systems or Apple Silicon Macs
```bash
conda env create -f environment-cpu.yaml
conda activate biomap-env
```

> **Note for Mac Users**: If you have an Apple Silicon Mac (M1/M2/M3), use the CPU environment file. It will automatically include MPS (Metal Performance Shaders) support for GPU acceleration on Apple hardware.

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