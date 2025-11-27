#!/usr/bin/env python
"""Reduce embedding dimensionality using cuML PCA.

This script loads embeddings from a .npz file (output from retrieve_embeddings.py),
flattens them, and uses cuML PCA to reduce dimensionality.

The embeddings can be either:
- Padded: shape [batch_size, seq_len, dimension] - flattened to [batch_size, seq_len * dimension]
- Variable-length: object array of variable-length embeddings - each flattened individually

Command-line usage:
    # Basic usage
    python -m pca.pca --input-file embeddings.npz --output-file embeddings_pca512.npz --n-components 512

    # With custom number of components
    python -m pca.pca --input-file embeddings.npz --output-file embeddings_pca256.npz --n-components 256

Python API usage:
    >>> from pca.pca import reduce_embeddings_pca
    >>> reduce_embeddings_pca(
    ...     input_file="embeddings.npz",
    ...     output_file="embeddings_pca512.npz",
    ...     n_components=512
    ... )
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import cupy as cp
import numpy as np
from cuml import PCA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embeddings(input_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and IDs from .npz file.

    Args:
        input_file: Path to input .npz file.

    Returns:
        Tuple of (embeddings_array, ids_array).
        embeddings_array can be:
            - 3D array with shape (batch_size, seq_len, dimension) for padded embeddings
            - Object array for variable-length embeddings

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the .npz file doesn't contain required keys.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading embeddings from {input_path}...")
    data = np.load(str(input_path), allow_pickle=True)

    if "embeddings" not in data or "ids" not in data:
        raise ValueError(
            f"Input file must contain 'embeddings' and 'ids' keys. "
            f"Found keys: {list(data.keys())}"
        )

    embeddings = data["embeddings"]
    ids = data["ids"]

    logger.info(f"Loaded {len(ids)} sequences")
    logger.info(f"Original embeddings shape: {embeddings.shape}")
    logger.info(f"Embeddings dtype: {embeddings.dtype}")

    return embeddings, ids


def flatten_embeddings(
    embeddings: np.ndarray, use_mean_pooling: bool = False
) -> np.ndarray:
    """
    Flatten embeddings for PCA processing, optionally using mean pooling.

    Args:
        embeddings: Embeddings array. Can be:
            - 3D array with shape (batch_size, seq_len, dimension) for padded embeddings
            - Object array for variable-length embeddings
        use_mean_pooling: If True, average over the seq_len dimension instead of flattening.
            This reduces shape from (batch_size, seq_len, dimension) to (batch_size, dimension).
            Defaults to False.

    Returns:
        Processed embeddings with shape (batch_size, features).
        If use_mean_pooling=True: features = dimension (mean pooled over seq_len)
        If use_mean_pooling=False: features = seq_len * dimension (flattened)

    Raises:
        ValueError: If embeddings have unexpected shape.
    """
    if len(embeddings.shape) == 3:
        # Padded embeddings: [batch_size, seq_len, dimension]
        num_sequences, seq_len, dimension = embeddings.shape
        
        if use_mean_pooling:
            # Mean pool over seq_len dimension: [batch_size, seq_len, dimension] -> [batch_size, dimension]
            pooled = np.mean(embeddings, axis=1)
            logger.info(
                f"Mean pooled padded embeddings from {embeddings.shape} to {pooled.shape}"
            )
            return pooled
        else:
            # Flatten: [batch_size, seq_len, dimension] -> [batch_size, seq_len * dimension]
            flattened = embeddings.reshape(num_sequences, seq_len * dimension)
            logger.info(
                f"Flattened padded embeddings from {embeddings.shape} to {flattened.shape}"
            )
            return flattened

    if embeddings.dtype == object:
        # Variable-length embeddings: object array
        logger.info("Processing variable-length embeddings...")
        
        if use_mean_pooling:
            # Mean pool each sequence over its seq_len dimension
            pooled_list = []
            for i, emb in enumerate(embeddings):
                if len(emb.shape) == 2:
                    # [seq_len, dimension] -> [dimension] (mean over seq_len)
                    pooled_list.append(np.mean(emb, axis=0))
                elif len(emb.shape) == 1:
                    # Already 1D, treat as single token embedding
                    pooled_list.append(emb)
                else:
                    raise ValueError(
                        f"Unexpected embedding shape at index {i}: {emb.shape}"
                    )
            
            # Stack into array: all should have same dimension
            pooled = np.array(pooled_list)
            logger.info(
                f"Mean pooled variable-length embeddings to shape {pooled.shape}"
            )
            return pooled
        else:
            # Original flattening behavior
            flattened_list = []
            for i, emb in enumerate(embeddings):
                if len(emb.shape) == 2:
                    # [seq_len, dimension] -> [seq_len * dimension]
                    flattened_list.append(emb.flatten())
                elif len(emb.shape) == 1:
                    # Already flattened
                    flattened_list.append(emb)
                else:
                    raise ValueError(
                        f"Unexpected embedding shape at index {i}: {emb.shape}"
                    )

            # Find max length and pad to same length for PCA
            max_len = max(len(emb) for emb in flattened_list)
            dimension = flattened_list[0].shape[-1] if len(flattened_list[0].shape) > 0 else max_len

            # Pad all to same length
            padded_flat = np.zeros((len(flattened_list), max_len), dtype=flattened_list[0].dtype)
            for i, emb in enumerate(flattened_list):
                padded_flat[i, : len(emb)] = emb

            logger.info(
                f"Flattened and padded variable-length embeddings to shape {padded_flat.shape}"
            )
            return padded_flat

    if len(embeddings.shape) == 2:
        # Already 2D (already flattened or mean pooled)
        logger.info(f"Embeddings already 2D: {embeddings.shape}")
        if use_mean_pooling:
            logger.warning(
                "use_mean_pooling=True but embeddings are already 2D. "
                "No pooling applied."
            )
        return embeddings

    raise ValueError(f"Unexpected embeddings shape: {embeddings.shape}")


def reduce_dimensionality(
    embeddings: np.ndarray, n_components: int = 512, random_state: int = 42
) -> Tuple[np.ndarray, PCA]:
    """
    Reduce embedding dimensionality using cuML PCA.

    Args:
        embeddings: Flattened embeddings array with shape (num_sequences, features).
        n_components: Number of components for PCA. Defaults to 512.
        random_state: Random state for reproducibility. Defaults to 42.

    Returns:
        Tuple of (reduced_embeddings, pca_model).
        reduced_embeddings has shape (num_sequences, n_components).

    Raises:
        ValueError: If n_components is greater than number of features or sequences.
    """
    num_sequences, num_features = embeddings.shape

    if n_components > num_features:
        raise ValueError(
            f"n_components ({n_components}) cannot be greater than "
            f"number of features ({num_features})"
        )

    if n_components > num_sequences:
        logger.warning(
            f"n_components ({n_components}) is greater than number of sequences "
            f"({num_sequences}). Setting n_components to {num_sequences}."
        )
        n_components = num_sequences

    logger.info(
        f"Reducing dimensionality from {num_features} to {n_components} using cuML PCA..."
    )

    # Convert to cupy array for GPU processing
    embeddings_gpu = cp.asarray(embeddings.astype(np.float32))

    # Fit PCA and transform
    pca = PCA(n_components=n_components) # NOTE: cuml uses deterministic SVD
    reduced_embeddings_gpu = pca.fit_transform(embeddings_gpu)

    # Convert back to numpy array
    reduced_embeddings = cp.asnumpy(reduced_embeddings_gpu)

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    total_variance = float(cp.sum(explained_variance))
    logger.info(
        f"Total explained variance: {total_variance:.4f} ({total_variance*100:.2f}%)"
    )
    logger.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")

    return reduced_embeddings, pca


def save_reduced_embeddings(
    reduced_embeddings: np.ndarray, ids: np.ndarray, output_file: Path
) -> None:
    """
    Save reduced embeddings to .npz file.

    Args:
        reduced_embeddings: Reduced embeddings array with shape (num_sequences, n_components).
        ids: Array of sequence IDs.
        output_file: Path to output .npz file.

    Raises:
        ValueError: If shapes don't match.
    """
    if len(reduced_embeddings) != len(ids):
        raise ValueError(
            f"Number of embeddings ({len(reduced_embeddings)}) doesn't match "
            f"number of IDs ({len(ids)})"
        )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving reduced embeddings to {output_path}...")
    np.savez_compressed(str(output_path), ids=ids, embeddings=reduced_embeddings)
    logger.info(
        f"Successfully saved {len(ids)} reduced embeddings with shape "
        f"{reduced_embeddings.shape} to {output_path}"
    )


def reduce_embeddings_pca(
    input_file: Path,
    output_file: Path,
    n_components: int = 512,
    random_state: int = 42,
    use_mean_pooling: bool = False,
) -> None:
    """
    Main function to reduce embedding dimensionality using PCA.

    Args:
        input_file: Path to input .npz file containing embeddings.
        output_file: Path to output .npz file for reduced embeddings.
        n_components: Number of PCA components. Defaults to 512.
        random_state: Random state for reproducibility. Defaults to 42.
        use_mean_pooling: If True, average over the seq_len dimension before PCA.
            This reduces memory usage and is useful when sequences are long.
            Defaults to False.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If embeddings have unexpected format or n_components is invalid.
    """
    # Load embeddings
    embeddings, ids = load_embeddings(input_file)

    # Process embeddings (flatten or mean pool)
    processed_embeddings = flatten_embeddings(embeddings, use_mean_pooling=use_mean_pooling)

    # Reduce dimensionality
    reduced_embeddings, pca_model = reduce_dimensionality(
        processed_embeddings, n_components=n_components, random_state=random_state
    )

    # Save reduced embeddings
    save_reduced_embeddings(reduced_embeddings, ids, output_file)

    logger.info("PCA dimensionality reduction completed successfully!")


def main() -> None:
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Reduce embedding dimensionality using cuML PCA"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to input .npz file containing embeddings",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to output .npz file for reduced embeddings",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=512,
        help="Number of PCA components (default: 512)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--use-mean-pooling",
        action="store_true",
        help="Average over sequence length dimension before PCA. "
        "Reduces memory usage for long sequences.",
    )

    args = parser.parse_args()

    reduce_embeddings_pca(
        input_file=args.input_file,
        output_file=args.output_file,
        n_components=args.n_components,
        random_state=args.random_state,
        use_mean_pooling=args.use_mean_pooling,
    )


if __name__ == "__main__":
    main()

