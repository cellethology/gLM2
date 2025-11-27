"""
gLM2.retrieve_embeddings module

Provides functions for retrieving and saving embeddings from biological sequences
using the gLM2 model.

Core usage examples:
    from retrieve_embeddings.retrieve_embeddings import process_fasta_and_save_embeddings, load_model_and_tokenizer

    # Process FASTA and save embeddings as .npz
    process_fasta_and_save_embeddings(
        fasta_path="my_sequences.fasta",
        output_path="my_embeddings.npz",
        batch_size=8,
        pad_embeddings=True
    )

Utility functions are available in:
    retrieve_embeddings.util

"""

from .retrieve_embeddings import (
    process_fasta_and_save_embeddings,
    load_model_and_tokenizer,
    get_embeddings_batch,
)
from .util import (
    read_fasta,
    save_embeddings_to_npz,
    validate_sequence,
)

__all__ = [
    "process_fasta_and_save_embeddings",
    "load_model_and_tokenizer",
    "get_embeddings_batch",
    "read_fasta",
    "save_embeddings_to_npz",
    "validate_sequence",
]
