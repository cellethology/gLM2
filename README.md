# gLM2 Framework

gLM2 (Genomic Language Model 2) is a mixed-modality deep learning framework for extracting embeddings from biological sequences (amino acids and nucleotides). This framework allows you to extract high-dimensional embeddings from FASTA sequences using pre-trained gLM2 models from HuggingFace.

## Overview

The gLM2 framework provides tools to:

- Extract embeddings from FASTA sequences using pre-trained gLM2 models (gLM2_650M, gLM2_150M, etc.)
- Process sequences in batches for efficient inference
- Reduce embedding dimensionality using GPU-accelerated PCA (cuML)
- Support both padded and variable-length embeddings
- Handle mixed-modality sequences (amino acids and nucleotides)

## Prerequisites

- Python 3.10 or higher
- `uv` package manager (install from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv))
- CUDA-capable GPU (optional, but recommended for faster inference and PCA)
- CUDA 12.x (for cuML GPU acceleration)

## Environment Setup

### 1. Install uv (if not already installed)

```bash
# Using pip
pip install uv
```

### 2. Create Virtual Environment and Install Dependencies

Run the environment setup command:

```bash
uv sync
source .venv/bin/activate
```

This will:

- Create a virtual environment (`.venv/`)
- Install all required dependencies including:
  - PyTorch and Transformers (for model inference)
  - cuML and CuPy (for GPU-accelerated PCA)
  - BioPython (for FASTA file handling)
  - NumPy and other utilities

**Note:** The `cuml-cu12` package is installed from the RAPIDS PyPI index, which requires CUDA 12.x. If you don't have CUDA or want CPU-only PCA, you may need to modify the dependencies.

### 3. Install Package in Editable Mode (Recommended)

For development and easier imports:

```bash
pip install -e .
```

Or with uv:

```bash
uv pip install -e .
```

## Usage

### Extract Embeddings from FASTA Sequences

The main module for extracting embeddings is `retrieve_embeddings`.

#### Basic Usage

```bash
python -m retrieve_embeddings.retrieve_embeddings \
    test_files/test.fasta \
    embeddings.npz
```

#### Full Command with All Options

```bash
python -m retrieve_embeddings.retrieve_embeddings \
    <path-to-input.fasta> \
    <path-to-output.npz> \
    --model-name tattabio/gLM2_650M \
    --batch-size 1 \
    --device cuda \
    --no-average
```

#### Command-Line Arguments

- `input_file` (positional, required): Path to input FASTA file containing biological sequences
- `output_file` (positional, required): Path to output `.npz` file where embeddings will be saved
- `--model-name` (optional): HuggingFace model identifier (default: `tattabio/gLM2_650M`)
  - Available models: `tattabio/gLM2_650M`, `tattabio/gLM2_150M`
- `--batch-size` (optional): Batch size for processing sequences (default: 1)
- `--device` (optional): Device to run inference on - `cuda`, `cpu`, or `auto` (default: `auto`)
- `--no-average` (optional): Keep per-token embeddings (default is mean pooled)
- `--no-validate` (optional): Skip sequence validation (not recommended)

#### Output Format

The script outputs a compressed NumPy archive (`.npz`) file containing:

- `ids`: Array of sequence IDs from the FASTA file
- `embeddings`:
  - If mean pooled (default): Array with shape `(num_sequences, hidden_dim)`
  - If per-token (`--no-average`): Object array of variable-length embeddings

#### Example Commands

```bash
# Extract embeddings with default settings (mean pooled, GPU if available)
python -m retrieve_embeddings.retrieve_embeddings \
    test_files/test.fasta \
    embeddings.npz

# Extract per-token embeddings
python -m retrieve_embeddings.retrieve_embeddings \
    test_files/test.fasta \
    embeddings.npz \
    --no-average

# Use CPU and smaller batch size
python -m retrieve_embeddings.retrieve_embeddings \
    test_files/test.fasta \
    embeddings.npz \
    --device cpu \
    --batch-size 4

# Use a smaller model
python -m retrieve_embeddings.retrieve_embeddings \
    test_files/test.fasta \
    embeddings.npz \
    --model-name tattabio/gLM2_150M
```

### Python API Usage

You can also use the functions programmatically:

```python
from pathlib import Path
from retrieve_embeddings import process_fasta_and_save_embeddings

# Extract and save embeddings
process_fasta_and_save_embeddings(
    fasta_path=Path("test_files/test.fasta"),
    output_path=Path("embeddings.npz"),
    model_name="tattabio/gLM2_650M",
    batch_size=1,
    average_embeddings=True,
    device="cuda"
)
```

### Reduce Embedding Dimensionality with PCA

After extracting embeddings, you can reduce their dimensionality using GPU-accelerated PCA:

#### Basic Usage

```bash
python -m pca.pca \
    --input-file embeddings.npz \
    --output-file embeddings_pca512.npz \
    --n-components 512
```

#### Full Command with All Options

```bash
python -m pca.pca \
    --input-file <path-to-input.npz> \
    --output-file <path-to-output.npz> \
    --n-components 512 \
    --random-state 42
```

#### Command-Line Arguments

- `--input-file` (required): Path to input `.npz` file containing embeddings
- `--output-file` (required): Path to output `.npz` file for reduced embeddings
- `--n-components` (optional): Number of PCA components (default: 512)
- `--random-state` (optional): Random seed for reproducibility (default: 42)

#### Supported Input Formats

The PCA module supports:

- **Padded embeddings**: 3D array `(batch_size, seq_len, dimension)` - automatically flattened
- **Variable-length embeddings**: Object array of variable-length embeddings - each flattened individually

#### Example

```bash
# Reduce to 256 dimensions
python -m pca.pca \
    --input-file embeddings.npz \
    --output-file embeddings_pca256.npz \
    --n-components 256

# Reduce with custom random seed
python -m pca.pca \
    --input-file embeddings.npz \
    --output-file embeddings_pca512.npz \
    --n-components 512 \
    --random-state 123
```

### Python API for PCA

```python
from pathlib import Path
from pca import reduce_embeddings_pca

# Reduce embedding dimensionality
reduce_embeddings_pca(
    input_file=Path("embeddings.npz"),
    output_file=Path("embeddings_pca512.npz"),
    n_components=512,
    random_state=42
)
```

### Loading Embeddings

You can load the saved embeddings in Python:

```python
import numpy as np
from pca import load_embeddings

# Load embeddings
embeddings, ids = load_embeddings("embeddings.npz")

print(f"Loaded {len(ids)} sequences")
print(f"Embeddings shape: {embeddings.shape}")

# Or load directly with numpy
data = np.load("embeddings.npz", allow_pickle=True)
sequence_ids = data["ids"]
embeddings = data["embeddings"]
```

## Sequence Format

gLM2 supports mixed-modality sequences containing:

- **Amino acids** (uppercase): A-Z (standard 20 plus ambiguous B, J, O, U, X, Z)
- **Nucleotides** (lowercase): a, t, g, c, n (for ambiguous)
- **Special strand tokens**: `<+>` and `<->` to indicate positive/negative strand

### Example Sequences

```
>seq1
<+>MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK<+>aattttttaaggaa<->MLGIDNIERVKPGGLEKGGRAFGFSAIVVVGNED

>seq2
<+>aatttaaaaaggaa
```

## Project Structure

```
gLM2/
├── model/                    # gLM2 model architecture
│   ├── __init__.py
│   ├── configuration_glm2.py  # Model configuration
│   └── modeling_glm2.py      # Model implementation
├── retrieve_embeddings/       # Embedding extraction module
│   ├── __init__.py
│   ├── retrieve_embeddings.py # Main embedding extraction script
│   └── util.py                # Utility functions (FASTA reading, validation, saving)
├── pca/                       # PCA dimensionality reduction
│   ├── __init__.py
│   └── pca.py                 # PCA reduction script
├── test_files/                # Test data
│   └── test.fasta            # Example FASTA file
├── tests/                     # Unit tests
│   └── test_saving_loading.py
├── docs/                      # Documentation
│   └── images/
├── pyproject.toml             # Project dependencies and configuration
├── uv.lock                    # Locked dependency versions
└── README.md                  # This file
```

## Testing

Run the test suite to verify your installation:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_saving_loading.py -v

# Run with coverage
pytest tests/ --cov=retrieve_embeddings --cov=pca
```

## Troubleshooting

### Model Download Issues

The gLM2 models are automatically downloaded from HuggingFace on first use. If you encounter download issues:

1. Check your internet connection
2. Verify HuggingFace access (models are public, but you may need to accept terms)
3. Try downloading manually from: https://huggingface.co/tattabio/gLM2_650M

### CUDA/cuML Issues

If you encounter issues with cuML (GPU-accelerated PCA):

1. **CUDA not available**: The framework will automatically fall back to CPU for model inference, but PCA requires cuML which needs CUDA 12.x
2. **cuML installation fails**: This is expected - cuML must be installed via conda:
   ```bash
   conda install -c rapidsai -c conda-forge -c nvidia cuml
   ```
3. **CPU-only PCA**: You can modify the code to use scikit-learn's PCA instead of cuML for CPU-only operation

### Import Errors

If you encounter `ModuleNotFoundError`:

1. Make sure the package is installed in editable mode:
   ```bash
   pip install -e .
   ```
2. Verify the virtual environment is activated
3. Check that `pyproject.toml` includes all necessary packages
