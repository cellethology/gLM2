"""Retrieve gLM2 embeddings from FASTA sequences.

CLI:
    python -m retrieve_embeddings.retrieve_embeddings input.fasta embeddings.npz
    python -m retrieve_embeddings.retrieve_embeddings input.fasta embeddings.npz --no-average

Python API:
    process_fasta_and_save_embeddings(
        fasta_path,
        output_path,
        batch_size=1,
        average_embeddings=True,
    )
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
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
    padding = len(sequences) > 1
    encodings = tokenizer(sequences, return_tensors="pt", padding=padding)
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
    batch_size: int = 1,
    device: Optional[str] = None,
    validate: bool = True,
    average_embeddings: bool = True,
) -> None:
    """
    Process sequences from a FASTA file and save embeddings to .npz file.

    Args:
        fasta_path: Path to the FASTA file.
        output_path: Path to save the embeddings .npz file.
        model: Pre-loaded model. If None, will load from model_name.
        tokenizer: Pre-loaded tokenizer. If None, will load from model_name.
        model_name: HuggingFace model identifier. Only used if model/tokenizer are None.
        batch_size: Number of sequences to process in each batch. Defaults to 1.
        device: Device to run inference on. If None, auto-detects.
        validate: If True, validate sequences before processing. Invalid sequences will be
            skipped. Defaults to True.
        average_embeddings: If True, average embeddings across the sequence length.
            Outputs [batch_size, hidden_dim]. If False, saves variable-length per-token
            embeddings (object array). Defaults to True.

    Example:
        >>> process_fasta_and_save_embeddings(
        ...     Path("sequences.fasta"),
        ...     Path("embeddings.npz"),
        ...     batch_size=1,
        ...     average_embeddings=True
        ... )
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name, device=device
        )
        device = next(model.parameters()).device
    elif device is None:
        device = next(model.parameters()).device

    # Read sequences from FASTA
    sequences_data = read_fasta(fasta_path, validate=validate)
    sequence_ids = [seq_id for seq_id, _ in sequences_data]
    sequences = [seq for _, seq in sequences_data]

    logger.info(f"Processing {len(sequences)} sequences in batches of {batch_size}")

    hidden_dim = None

    # Process in batches and collect embeddings
    all_embeddings_list: List[torch.Tensor] = []
    all_ids: List[str] = []

    for i in tqdm(
        range(0, len(sequences), batch_size),
        desc="Extracting embeddings",
        unit="batch",
    ):
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
        padding = len(batch_sequences) > 1
        encodings = tokenizer(batch_sequences, return_tensors="pt", padding=padding)

        if average_embeddings:
            if not padding:
                pooled_embeddings = batch_embeddings.mean(dim=1)
            else:
                attention_mask = encodings.attention_mask.to(batch_embeddings.device)
                mask = attention_mask.unsqueeze(-1)
                lengths = mask.sum(dim=1).clamp_min(1)
                pooled_embeddings = (batch_embeddings * mask).sum(dim=1) / lengths
            all_embeddings_list.extend(pooled_embeddings)
            all_ids.extend(batch_ids)
            continue

        if not padding:
            all_embeddings_list.append(batch_embeddings[0])
            all_ids.append(batch_ids[0])
            continue

        attention_mask = encodings.attention_mask.to(batch_embeddings.device)
        for j, seq_id in enumerate(batch_ids):
            seq_len = attention_mask[j].sum().item()
            # Keep variable-length embeddings (remove padding)
            all_embeddings_list.append(batch_embeddings[j, :seq_len, :])
            all_ids.append(seq_id)

    logger.info(f"Successfully processed {len(all_embeddings_list)} sequences")

    ids_array = np.array(all_ids)

    if average_embeddings:
        embeddings_array = torch.stack(all_embeddings_list).cpu().float().numpy()

        assert embeddings_array.shape == (
            len(all_ids),
            hidden_dim,
        ), f"Unexpected embeddings shape: {embeddings_array.shape}"

        logger.info(
            f"Embeddings shape: {embeddings_array.shape} = [batch_size={len(all_ids)}, "
            f"hidden_dim={hidden_dim}]"
        )

        save_embeddings_to_npz(
            embeddings_array,
            ids_array,
            output_path,
            pooled=True,
        )
    else:
        # Convert to numpy and save as variable-length (object array)
        # Convert bfloat16 to float32 for numpy compatibility
        embeddings_list = [emb.cpu().float().numpy() for emb in all_embeddings_list]
        embeddings_array = np.array(embeddings_list, dtype=object)

        logger.info(
            f"Saved {len(all_ids)} variable-length embeddings (hidden_dim={hidden_dim})"
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
        default=1,
        help="Batch size for processing (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', or None for auto-detect)",
    )
    parser.add_argument(
        "--no-average",
        action="store_true",
        help="Don't average embeddings across sequence length (keeps per-token embeddings)",
    )

    args = parser.parse_args()

    process_fasta_and_save_embeddings(
        fasta_path=args.fasta_path,
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        validate=False,
        average_embeddings=not args.no_average,
    )


if __name__ == "__main__":
    main()
