from dataclasses import dataclass

@dataclass
class Config:
    # Preprocessing
    max_sequence_length: int = 512
    remove_stopwords: bool = True

    # Splitting
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42

    # Tokenization
    vocab_size: int = 30000
    padding: str = "max_length"

    # Embedding
    embedding_dim: int = 768

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10

    # Paths
    model_save_path: str = "models/saved/"
    logs_path: str = "logs/"