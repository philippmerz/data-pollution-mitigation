from dataclasses import dataclass, field
from ..utils.utils import Models

@dataclass
class Config:

    # Dev mode: if True, only a subset of the data will be used
    dev: bool = False

    model: Models = 'logistic-regression'

    # Raw data
    raw_data_path: str = "data/raw/gender.csv"

    # Preprocessing
    min_chars_per_author: int = 300 # All authors with less are eliminated
    max_sequence_length: int = 512
    remove_contamination: bool = False

    # adds artificial self-ID gender to every single post
    add_full_contamination: bool = True

    discarded_data_path: str = "data/preprocessed/discarded.csv"
    train_data_path: str = "data/preprocessed/train.csv"
    test_data_path: str = "data/preprocessed/test.csv"
    val_data_path: str = "data/preprocessed/val.csv"

    if not remove_contamination:
        discarded_data_path = "data/preprocessed/discarded_contaminated.csv"
        train_data_path = "data/preprocessed/train_contaminated.csv"
        test_data_path = "data/preprocessed/test_contaminated.csv"
        val_data_path = "data/preprocessed/val_contaminated.csv"

    if add_full_contamination:
        discarded_data_path = "data/preprocessed/discarded_100contaminated.csv"
        train_data_path = "data/preprocessed/train_100contaminated.csv"
        test_data_path = "data/preprocessed/test_100contaminated.csv"
        val_data_path = "data/preprocessed/val_100contaminated.csv"

    # Splitting
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42

    # Tokenization
    padding: str = "max_length"

    # Embedding
    train_cls_path: str = "data/embedded/embedded_train_data.csv"
    test_cls_path: str = "data/embedded/embedded_test_data.csv"
    val_cls_path: str = "data/embedded/embedded_val_data.csv"

    if not remove_contamination:
        train_cls_path = "data/embedded/embedded_train_data_contaminated.csv"
        test_cls_path = "data/embedded/embedded_test_data_contaminated.csv"
        val_cls_path = "data/embedded/embedded_val_data_contaminated.csv"

    embedding_batch_size: int = 64

    # Training
    learning_rate: float = 0.001

    # Logistic Regression
    max_iter: int = 1000
    random_seed: int = 42

    # Neural Network 
    nn_hidden_layers: list = field(default_factory=lambda: [256, 128, 128, 64, 16])
    nn_dropout: float = 0.0
    nn_epochs: int = 10
    nn_batch_size: int = 32
    nn_learning_rate: float = 0.001
