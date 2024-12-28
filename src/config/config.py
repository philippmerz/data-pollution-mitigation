from dataclasses import dataclass

@dataclass
class Config:

    # Dev mode: if True, only a subset of the data will be used
    dev: bool = True

    model: str = 'logistic-regression' # OR 'xgboost'

    # Preprocessing
    min_chars_per_author: int = 300 # All authors with less are eliminated
    max_sequence_length: int = 512
    remove_contamination: bool = True

    # Splitting
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42

    # Tokenization
    padding: str = "max_length"

    # Embedding

    # Training
    learning_rate: float = 0.001

    # Paths