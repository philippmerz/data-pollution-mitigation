from dataclasses import dataclass, field

@dataclass
class Config:

    # Dev mode: if True, only a subset of the data will be used
    dev: bool = True
    #dev: bool = False

    model: str = 'neural-network' # xgboost OR 'logistic-regression' OR 'neural-network'

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

    embedding_batch_size: int = 64

    # Training
    learning_rate: float = 0.001

    # Logistic Regression
    max_iter: int = 1000
    random_seed: int = 42

    # Neural Network 
    nn_hidden_layers: list = field(default_factory=lambda: [1536, 768, 512, 768, 256, 128, 64, 64, 64])
    nn_dropout: float = 0
    nn_epochs: int = 10
    nn_batch_size: int = 32
    nn_learning_rate: float = 0.00001
