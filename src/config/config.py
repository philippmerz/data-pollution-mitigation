from dataclasses import dataclass

@dataclass
class Config:

    # Dev mode: if True, only a subset of the data will be used
    dev: bool = True

    model: str = 'logistic-regression' # OR 'xgboost'

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

    train_token_path: str = "data/tokenized/tokenized_train_data.pkl"
    test_token_path: str = "data/tokenized/tokenized_test_data.pkl"
    val_token_path: str = "data/tokenized/tokenized_val_data.pkl"

    # Embedding
    train_embedded_path: str = "data/embedded/embedded_train_data.pkl"
    test_embedded_path: str = "data/embedded/embedded_test_data.pkl"
    val_embedded_path: str = "data/embedded/embedded_val_data.pkl"

    # Training
    learning_rate: float = 0.001

    # Paths