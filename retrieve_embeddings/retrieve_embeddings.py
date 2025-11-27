"""Core functions for retrieving embeddings from sequences.

Command-line usage:
    # First, install the package in editable mode:
    pip install -e .
    
    # Then use as a module:
    python -m retrieve_embeddings.retrieve_embeddings ./retrieve_embeddings/test.fasta embeddings.npz

    # With custom batch size
    python -m retrieve_embeddings.retrieve_embeddings ./retrieve_embeddings/test.fasta embeddings.npz --batch-size 16

    # Without padding (variable-length embeddings)
    python -m retrieve_embeddings.retrieve_embeddings ./retrieve_embeddings/test.fasta embeddings.npz --no-pad

    # Use specific device
    python -m retrieve_embeddings.retrieve_embeddings ./retrieve_embeddings/test.fasta embeddings.npz --device cuda

    # Use different model
    python -m retrieve_embeddings.retrieve_embeddings ./retrieve_embeddings/test.fasta embeddings.npz --model-name tattabio/gLM2_150M

    # Combine options
    python -m retrieve_embeddings.retrieve_embeddings ./retrieve_embeddings/test.fasta embeddings.npz --batch-size 4 --no-pad --device cpu

Python API usage:
    >>> from pathlib import Path
    >>> from retrieve_embeddings import process_fasta_and_save_embeddings
    >>> 
    >>> process_fasta_and_save_embeddings(
    ...     fasta_path=Path("test.fasta"),
    ...     output_path=Path("embeddings.npz"),
    ...     batch_size=8,
    ...     pad_embeddings=True
    ... )
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from retrieve_embeddings.util import read_fasta, save_embeddings_to_npz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str = "tattabio/gLM2_650M",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load the gLM2 model and tokenizer.

    Args:
        model_name: HuggingFace model identifier. Defaults to 'tattabio/gLM2_650M'.
        device: Device to load the model on ('cuda', 'cpu', or None for auto-detect).
            Defaults to None (auto-detect).
        dtype: Data type for the model. Defaults to torch.bfloat16.

    Returns:
        Tuple of (model, tokenizer).

    Example:
        >>> model, tokenizer = load_model_and_tokenizer()
        >>> print(type(model))
        <class 'transformers.models.glm2.modeling_glm2.gLM2Model'>
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model {model_name} on device {device}")
    model = AutoModel.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model.eval()  # Set to evaluation mode
    logger.info("Model and tokenizer loaded successfully")

    return model, tokenizer


def get_embeddings_batch(
    sequences: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Get embeddings for a batch of sequences.

    Args:
        sequences: List of sequence strings to embed.
        model: Loaded gLM2 model.
        tokenizer: Tokenizer for the model.
        device: Device to run inference on. If None, uses model's device.

    Returns:
        Tensor of embeddings with shape (batch_size, seq_len, hidden_dim).

    Example:
        >>> model, tokenizer = load_model_and_tokenizer()
        >>> sequences = ["<+>MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK"]
        >>> embeddings = get_embeddings_batch(sequences, model, tokenizer)
        >>> print(embeddings.shape)
        torch.Size([1, seq_len, hidden_dim])
    """
    if not sequences:
        raise ValueError("sequences list cannot be empty")

    if device is None:
        device = next(model.parameters()).device

    # Tokenize all sequences
    encodings = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = encodings.input_ids.to(device)

    # Extract embeddings
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        embeddings = outputs.last_hidden_state

    return embeddings


def process_fasta_and_save_embeddings(
    fasta_path: Path,
    output_path: Path,
    model: Optional[AutoModel] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    model_name: str = "tattabio/gLM2_650M",
    batch_size: int = 8,
    device: Optional[str] = None,
    validate: bool = True,
    pad_embeddings: bool = True,
) -> None:
    """
    Process sequences from a FASTA file and save embeddings to .npz file.

    Args:
        fasta_path: Path to the FASTA file.
        output_path: Path to save the embeddings .npz file.
        model: Pre-loaded model. If None, will load from model_name.
        tokenizer: Pre-loaded tokenizer. If None, will load from model_name.
        model_name: HuggingFace model identifier. Only used if model/tokenizer are None.
        batch_size: Number of sequences to process in each batch. Defaults to 8.
        device: Device to run inference on. If None, auto-detects.
        validate: If True, validate sequences before processing. Invalid sequences will be
            skipped. Defaults to True.
        pad_embeddings: If True, pad all embeddings to the same max length (shape will be
            [batch_size, seq_len, dimension]). If False, keep variable-length embeddings
            (saved as object array). Defaults to True.

    Example:
        >>> process_fasta_and_save_embeddings(
        ...     Path("sequences.fasta"),
        ...     Path("embeddings.npz"),
        ...     batch_size=4,
        ...     pad_embeddings=True
        ... )
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_name=model_name, device=device)
        device = next(model.parameters()).device
    elif device is None:
        device = next(model.parameters()).device

    # Read sequences from FASTA
    sequences_data = read_fasta(fasta_path, validate=validate)
    sequence_ids = [seq_id for seq_id, _ in sequences_data]
    sequences = [seq for _, seq in sequences_data]

    logger.info(f"Processing {len(sequences)} sequences in batches of {batch_size}")

    # Find global max length if padding is enabled
    global_max_len = None
    hidden_dim = None
    if pad_embeddings:
        all_encodings = tokenizer(sequences, return_tensors="pt", padding=True)
        global_max_len = all_encodings.input_ids.shape[1]
        logger.info(f"Global max sequence length for padding: {global_max_len}")

    # Process in batches and collect embeddings
    all_embeddings_list: List[torch.Tensor] = []
    all_ids: List[str] = []

    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i : i + batch_size]
        batch_ids = sequence_ids[i : i + batch_size]

        logger.debug(
            f"Processing batch {i // batch_size + 1}/{(len(sequences) + batch_size - 1) // batch_size}"
        )

        # Get embeddings for this batch
        batch_embeddings = get_embeddings_batch(
            batch_sequences,
            model,
            tokenizer,
            device=device,
        )

        # Store hidden_dim from first batch
        if hidden_dim is None:
            hidden_dim = batch_embeddings.shape[2]

        # Process each sequence in the batch
        encodings = tokenizer(batch_sequences, return_tensors="pt", padding=True)
        attention_mask = encodings.attention_mask

        for j, seq_id in enumerate(batch_ids):
            seq_len = attention_mask[j].sum().item()
            
            if pad_embeddings:
                # Pad to global max length
                padded_emb = torch.zeros(
                    (global_max_len, hidden_dim),
                    dtype=batch_embeddings.dtype,
                    device=batch_embeddings.device,
                )
                # Copy actual sequence (up to batch max length)
                batch_max_len = batch_embeddings.shape[1]
                actual_len = min(seq_len, batch_max_len)
                padded_emb[:actual_len, :] = batch_embeddings[j, :actual_len, :]
                all_embeddings_list.append(padded_emb)
            else:
                # Keep variable-length embeddings (remove padding)
                all_embeddings_list.append(batch_embeddings[j, :seq_len, :])
            
            all_ids.append(seq_id)

    logger.info(f"Successfully processed {len(all_embeddings_list)} sequences")

    ids_array = np.array(all_ids)

    if pad_embeddings:
        # Stack all embeddings into a single array: [batch_size, seq_len, dimension]
        # Convert bfloat16 to float32 for numpy compatibility
        embeddings_array = torch.stack(all_embeddings_list).cpu().float().numpy()
        
        # Verify shape
        assert embeddings_array.shape == (
            len(all_ids),
            global_max_len,
            hidden_dim,
        ), f"Unexpected embeddings shape: {embeddings_array.shape}"

        logger.info(
            f"Embeddings shape: {embeddings_array.shape} = [batch_size={len(all_ids)}, "
            f"seq_len={global_max_len}, dimension={hidden_dim}]"
        )
        
        # Save padded embeddings
        save_embeddings_to_npz(embeddings_array, ids_array, output_path, padded=True)
    else:
        # Convert to numpy and save as variable-length (object array)
        # Convert bfloat16 to float32 for numpy compatibility
        embeddings_list = [emb.cpu().float().numpy() for emb in all_embeddings_list]
        embeddings_array = np.array(embeddings_list, dtype=object)
        
        logger.info(
            f"Saved {len(all_ids)} variable-length embeddings "
            f"(hidden_dim={hidden_dim})"
        )
        
        # Save variable-length embeddings
        save_embeddings_to_npz(embeddings_array, ids_array, output_path, padded=False)


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract embeddings from sequences in a FASTA file"
    )
    parser.add_argument(
        "fasta_path",
        type=Path,
        help="Path to input FASTA file",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to output .npz file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="tattabio/gLM2_650M",
        help="HuggingFace model identifier (default: tattabio/gLM2_650M)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', or None for auto-detect)",
    )
    parser.add_argument(
        "--no-pad",
        action="store_true",
        help="Don't pad embeddings to the same length (saves variable-length embeddings)",
    )

    args = parser.parse_args()

    process_fasta_and_save_embeddings(
        fasta_path=args.fasta_path,
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        validate=True,  # Always validate sequences
        pad_embeddings=not args.no_pad,
    )


if __name__ == "__main__":
    main()

