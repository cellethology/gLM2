"""Tests for FASTA file loading and embedding generation.

This module tests:
- FASTA file loading
- Model-based embedding generation and saving
"""

import tempfile
from pathlib import Path

import numpy as np

from retrieve_embeddings.retrieve_embeddings import process_fasta_and_save_embeddings
from retrieve_embeddings.util import read_fasta
from pca.pca import load_embeddings


def test_read_fasta_loads_sequences_correctly() -> None:
    """Test that read_fasta correctly loads sequences from a FASTA file."""
    # Create a temporary FASTA file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        # Write test FASTA content
        f.write(">seq1\n")
        f.write("<+>MALTKVEKRNRIKRRVKISGTQASPRLSVYKSNK<+>aattttttaaggaa<->MLGIDNIERVKPGGLEKGGRAFGFSAIVVVGNED\n")
        f.write(">seq2\n")
        f.write("<+>MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK<+>aatttaaaaaggaa<->MLGIDNIERVKPGGLELVDRLVAVNRVTKVTKGGRAFGFSAIVVVGNED\n")
        f.write(">seq3\n")
        f.write("<+>MALTKVEKRNRIKRRVRRVRGKISGTQASPRLSVYKSNK<+>aatttaagggggaa<->MLGIDNIERVKPGGLELVDRLVAVNRVTKVTKGGRAFGFSAIVVVGNEDRAFGFSAIVVVGNED\n")
        temp_fasta_path = Path(f.name)

    try:
        # Load the FASTA file
        sequences = read_fasta(temp_fasta_path, validate=False)

        # Verify we got the expected number of sequences
        assert len(sequences) == 3, f"Expected 3 sequences, got {len(sequences)}"

        # Verify the sequences match
        assert sequences[0] == ("seq1", "<+>MALTKVEKRNRIKRRVKISGTQASPRLSVYKSNK<+>aattttttaaggaa<->MLGIDNIERVKPGGLEKGGRAFGFSAIVVVGNED")
        assert sequences[1] == ("seq2", "<+>MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK<+>aatttaaaaaggaa<->MLGIDNIERVKPGGLELVDRLVAVNRVTKVTKGGRAFGFSAIVVVGNED")
        assert sequences[2] == ("seq3", "<+>MALTKVEKRNRIKRRVRRVRGKISGTQASPRLSVYKSNK<+>aatttaagggggaa<->MLGIDNIERVKPGGLELVDRLVAVNRVTKVTKGGRAFGFSAIVVVGNEDRAFGFSAIVVVGNED")

        # Verify each entry is a tuple of (id, sequence)
        for seq_id, sequence in sequences:
            assert isinstance(seq_id, str), "Sequence ID should be a string"
            assert isinstance(sequence, str), "Sequence should be a string"
            assert len(sequence) > 0, "Sequence should not be empty"

    finally:
        # Clean up
        temp_fasta_path.unlink()


def test_embeddings_saved_with_actual_model_content() -> None:
    """Test that embeddings generated from the model contain actual content when saved.
    
    This test:
    1. Uses the test.fasta file from test_files directory
    2. Uses process_fasta_and_save_embeddings to generate and save embeddings
    3. Loads the saved embeddings and verifies they contain real content
    """
    # Use the existing test.fasta file
    test_fasta_path = Path(__file__).parent.parent / "test_files" / "test.fasta"
    assert test_fasta_path.exists(), f"Test FASTA file not found at {test_fasta_path}"
    
    # Create temporary output file for embeddings
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        temp_embeddings_path = Path(f.name)
    
    try:
        # Generate and save embeddings using process_fasta_and_save_embeddings
        # The function will load the model internally
        process_fasta_and_save_embeddings(
            fasta_path=test_fasta_path,
            output_path=temp_embeddings_path,
            model_name="tattabio/gLM2_650M",
            device="cpu",  # Use CPU for testing
            batch_size=2,
            validate=True,
            average_embeddings=True,  # Test with mean pooled embeddings
        )
        
        # Load the saved embeddings
        loaded_embeddings, loaded_ids = load_embeddings(temp_embeddings_path)
        
        # Verify basic structure (test.fasta has 3 sequences)
        assert len(loaded_embeddings) == 3, f"Expected 3 embeddings, got {len(loaded_embeddings)}"
        assert len(loaded_ids) == 3, f"Expected 3 IDs, got {len(loaded_ids)}"
        assert len(loaded_embeddings.shape) == 2, (
            f"Expected 2D array [batch, hidden_dim], got shape {loaded_embeddings.shape}"
        )
        
        # Verify embeddings contain actual content (not all zeros)
        for i, embedding in enumerate(loaded_embeddings):
            assert embedding.ndim == 1, f"Embedding {i} should be 1D, got {embedding.ndim}D"

            # Check that embedding is not all zeros
            assert not np.allclose(embedding, 0.0), (
                f"Embedding {i} should not be all zeros"
            )

            # Check that embedding has non-zero variance (contains variation)
            assert np.var(embedding) > 1e-6, (
                f"Embedding {i} should have non-zero variance, got {np.var(embedding)}"
            )

            # Check that embedding values are reasonable (not NaN or Inf)
            assert np.all(np.isfinite(embedding)), (
                f"Embedding {i} should contain only finite values"
            )

            # Check that embedding has reasonable magnitude (not extremely large)
            assert np.abs(embedding).max() < 1e6, (
                f"Embedding {i} should not have extremely large values, max={np.abs(embedding).max()}"
            )
        
        # Verify IDs match the sequences in test.fasta
        assert "seq1" in loaded_ids, f"Expected 'seq1' in IDs, got {loaded_ids}"
        assert "seq2" in loaded_ids, f"Expected 'seq2' in IDs, got {loaded_ids}"
        assert "seq3" in loaded_ids, f"Expected 'seq3' in IDs, got {loaded_ids}"
        
        # Verify embeddings have different content (sequences are different)
        if loaded_embeddings.shape[0] >= 2:
            emb1 = loaded_embeddings[0]
            emb2 = loaded_embeddings[1]
            assert not np.allclose(emb1, emb2), (
                "Embeddings from different sequences should be different"
            )
        
        # Verify embedding dimensions are reasonable
        batch_size, hidden_dim = loaded_embeddings.shape
        assert hidden_dim > 0, f"Hidden dimension should be positive, got {hidden_dim}"
        assert batch_size == 3, f"Batch size should be 3, got {batch_size}"
        
    finally:
        # Clean up
        if temp_embeddings_path.exists():
            temp_embeddings_path.unlink()
