from dataclasses import dataclass

@dataclass
class Config:

    # Dev mode: if True, only a subset of the data will be used
    dev: bool = True

    # Preprocessing
    min_chars_per_author: int = 300 # All authors with less are eliminated
    max_sequence_length: int = 512

    # Splitting
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42

    # Tokenization
    padding: str = "max_length"

    # Embedding

    # Training

    # Paths