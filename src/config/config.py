from dataclasses import dataclass

@dataclass
class Config:

    # Dev mode: if True, only a subset of the data will be used
    dev: bool = True

    model: str = 'xgboost' # OR 'logistic-regression'

    # Raw data
    raw_data_path: str = "data/raw/gender.csv"

    # Preprocessing
    min_chars_per_author: int = 300 # All authors with less are eliminated
    max_sequence_length: int = 512
    remove_contamination: bool = True

    discarded_data_path: str = "data/preprocessed/discarded.csv"
    train_data_path: str = "data/preprocessed/train.csv"
    test_data_path: str = "data/preprocessed/test.csv"
    val_data_path: str = "data/preprocessed/val.csv"


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

    # Training
    learning_rate: float = 0.001

    # Logistic Regression
    max_iter: int = 1000
    random_seed: int = 42

    # Paths