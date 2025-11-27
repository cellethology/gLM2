"""
gLM2.pca module

Provides functions for reducing embedding dimensionality using PCA.

Core usage examples:
    from pca import reduce_embeddings_pca, load_embeddings
    
    # Reduce embeddings dimensionality
    reduce_embeddings_pca(
        input_file="embeddings.npz",
        output_file="embeddings_pca512.npz",
        n_components=512
    )
    
    # Load embeddings from .npz file
    embeddings, ids = load_embeddings("embeddings.npz")
"""

from .pca import (
    load_embeddings,
    reduce_embeddings_pca,
    save_reduced_embeddings,
)

__all__ = [
    "load_embeddings",
    "reduce_embeddings_pca",
    "save_reduced_embeddings",
]
