"""Utility functions for embedding retrieval."""

import logging
import re
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
from Bio import SeqIO

# Configure logging
logger = logging.getLogger(__name__)

# Validation constants
MIN_SEQUENCE_LENGTH: int = 1
# Valid amino acid characters (uppercase, including ambiguous IUPAC extended code https://www.bioinformatics.org/sms/iupac.html) 
VALID_AA_CHARS: Set[str] = set("ACDEFGHIKLMNPQRSTVWYBJOUXZ")
# Valid nucleotide characters (lowercase, including ambiguous)
VALID_NUC_CHARS: Set[str] = set("atgcn")
# All valid characters including special tokens
VALID_CHARS: Set[str] = VALID_AA_CHARS | VALID_NUC_CHARS | set("<>+-")


def validate_sequence(sequence: str, seq_id: str = "") -> Tuple[bool, str]:
    """
    Validate a gLM2 sequence for length and invalid characters.

    gLM2 sequences can contain:
    - Amino acids (uppercase): A-Z (standard 20 plus ambiguous B, J, O, U, X, Z)
    - Nucleotides (lowercase): a, t, g, c, n (for ambiguous)
    - Special strand tokens: `<+>` and `<->` to indicate positive/negative strand

    Args:
        sequence: Sequence string to validate (can contain AA, nucleotides, and strand tokens).
        seq_id: Optional sequence identifier for error messages.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty string.

    Example:
        >>> is_valid, error = validate_sequence("<+>MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK")
        >>> print(is_valid)
        True
    """
    if not sequence:
        return False, f"Empty sequence found{': ' + seq_id if seq_id else ''}"

    seq_len = len(sequence)
    if seq_len < MIN_SEQUENCE_LENGTH:
        return False, (
            f"Sequence too short (length {seq_len}, minimum {MIN_SEQUENCE_LENGTH})"
            f"{': ' + seq_id if seq_id else ''}"
        )

    # Check for invalid characters
    sequence_chars = set(sequence)
    invalid_chars = sequence_chars - VALID_CHARS
    if invalid_chars:
        return False, (
            f"Invalid characters found: {sorted(invalid_chars)}"
            f"{' in sequence: ' + seq_id if seq_id else ''}"
        )

    # Validate strand token format (must be complete <+> or <->, not partial)
    strand_pattern = r"<[+-]>"
    temp_seq = re.sub(strand_pattern, "", sequence)
    if "<" in temp_seq or ">" in temp_seq or "+" in temp_seq or "-" in temp_seq:
        return False, (
            f"Malformed strand tokens found (must be '<+>' or '<->')"
            f"{' in sequence: ' + seq_id if seq_id else ''}"
        )

    # Validate case requirements: 
    # Make sure that for characters A,G,N,T: 
    # - if uppercase, accepted as AA (valid)
    # - if lowercase, accepted as nucleotide (valid)
    # Ensure no lowercase amino acids (including a,g,n,t) and no uppercase nucleotides (including A,G,N,T)
    lowercase_aas = []
    for char in temp_seq:
        if char.islower():
            # If the uppercase form is in amino acids but the char is not a valid nucleotide letter
            # This catches e.g. 'm', 'p', etc. so need to distinguish
            if char.upper() in VALID_AA_CHARS and char not in VALID_NUC_CHARS:
                lowercase_aas.append(char)

    if lowercase_aas:
        return False, (
            f"Amino acids must be uppercase, found lowercase: {sorted(set(lowercase_aas))}"
            f"{' in sequence: ' + seq_id if seq_id else ''}"
        )

    uppercase_nucs = []
    for char in temp_seq:
        if char.isupper():
            # Only flag as error if this is a nucleotide letter in uppercase form
            if char.lower() in VALID_NUC_CHARS and char not in VALID_AA_CHARS:
                uppercase_nucs.append(char)
            # For A/G/N/T: if uppercase and present in both, treat as AA, don't error

    if uppercase_nucs:
        return False, (
            f"Nucleotides must be lowercase, found uppercase: {sorted(set(uppercase_nucs))}"
            f"{' in sequence: ' + seq_id if seq_id else ''}"
        )

    return True, ""


def read_fasta(
    file_path: Path, validate: bool = True
) -> List[Tuple[str, str]]:
    """
    Read sequences from a FASTA file using BioPython's SeqIO.

    Args:
        file_path: Path to the FASTA file.
        validate: If True, validate each sequence using validate_sequence().
            Invalid sequences will be skipped with a warning. Defaults to True.

    Returns:
        List of tuples containing (sequence_id, sequence) pairs.

    Raises:
        FileNotFoundError: If the FASTA file does not exist.
        ValueError: If the FASTA file is empty or malformed.

    Example:
        >>> sequences = read_fasta(Path("sequences.fasta"))
        >>> print(sequences[0])
        ('seq1', 'MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK')
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    sequences: List[Tuple[str, str]] = []
    skipped_count = 0

    try:
        for record in SeqIO.parse(str(file_path), "fasta"):
            seq_id = record.id
            sequence = str(record.seq)

            if not sequence:
                logger.warning(f"Sequence {seq_id} is empty, skipping")
                skipped_count += 1
                continue

            # Validate sequence if requested
            if validate:
                is_valid, error_msg = validate_sequence(sequence, seq_id)
                if not is_valid:
                    logger.warning(f"Skipping invalid sequence {seq_id}: {error_msg}")
                    skipped_count += 1
                    continue

            sequences.append((seq_id, sequence))

    except Exception as e:
        raise ValueError(f"Error parsing FASTA file {file_path}: {e}") from e

    if not sequences:
        raise ValueError(f"FASTA file is empty or all sequences were invalid: {file_path}")

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} invalid or empty sequences")

    logger.info(f"Read {len(sequences)} sequences from {file_path}")
    return sequences


def save_embeddings_to_npz(
    embeddings: np.ndarray,
    ids: np.ndarray,
    output_path: Path,
    padded: bool = True,
    pooled: bool = False,
) -> None:
    """
    Save embeddings to a compressed .npz file.

    Args:
        embeddings: Embeddings array. If pooled=True, should have shape [batch_size, dimension].
            If padded=True, should have shape [batch_size, seq_len, dimension]. If padded=False,
            can be a list/array of variable-length embeddings.
        ids: Array of sequence IDs with length batch_size.
        output_path: Path to save the .npz file.
        padded: If True, embeddings are padded to the same length (3D array).
            If False, embeddings have variable lengths (will be saved as object array).
            Defaults to True.
        pooled: If True, embeddings are averaged across sequence length (2D array).
            Defaults to False.

    Raises:
        ValueError: If embeddings or ids are empty, or shapes don't match.

    Example:
        >>> embeddings = np.random.randn(3, 100, 512)
        >>> ids = np.array(['seq1', 'seq2', 'seq3'])
        >>> save_embeddings_to_npz(embeddings, ids, Path("embeddings.npz"))
    """
    if len(embeddings) == 0:
        raise ValueError("embeddings array cannot be empty")

    if len(ids) == 0:
        raise ValueError("ids array cannot be empty")

    if len(embeddings) != len(ids):
        raise ValueError(
            f"Number of embeddings ({len(embeddings)}) doesn't match "
            f"number of IDs ({len(ids)})"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pooled:
        if isinstance(embeddings, list):
            embeddings_array = np.array(embeddings)
        else:
            embeddings_array = embeddings

        if len(embeddings_array.shape) != 2:
            raise ValueError(
                f"Expected 2D embeddings array with shape [batch_size, dimension] "
                f"when pooled=True, got shape {embeddings_array.shape}"
            )

        np.savez_compressed(str(output_path), ids=ids, embeddings=embeddings_array)
        logger.info(
            f"Successfully saved {len(ids)} pooled embeddings with shape {embeddings_array.shape} "
            f"to {output_path}"
        )
    elif padded:
        # Padded embeddings: should be 3D array
        if len(embeddings.shape) != 3:
            raise ValueError(
                f"Expected 3D embeddings array with shape [batch_size, seq_len, dimension] "
                f"when padded=True, got shape {embeddings.shape}"
            )
        # Save directly
        np.savez_compressed(str(output_path), ids=ids, embeddings=embeddings)
        logger.info(
            f"Successfully saved {len(ids)} padded embeddings with shape {embeddings.shape} to {output_path}"
        )
    else:
        # Variable-length embeddings: save as object array
        if isinstance(embeddings, list):
            embeddings_array = np.array(embeddings, dtype=object)
        else:
            embeddings_array = embeddings
        
        np.savez_compressed(str(output_path), ids=ids, embeddings=embeddings_array)
        logger.info(
            f"Successfully saved {len(ids)} variable-length embeddings to {output_path}"
        )
