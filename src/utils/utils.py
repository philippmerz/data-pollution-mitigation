from pathlib import Path
from typing import Literal, Any

PipelineStage = Literal['raw', 'preprocessed', 'tokenized', 'classifier_tokens']

def load_data(path: Path, start_from: PipelineStage) -> Any:
    """Load data for the pipeline based on the starting stage."""
    if start_from == 'raw':
        return path

    elif start_from == 'preprocessed':
        import pandas as pd
        return pd.read_csv(path)

    elif start_from == 'split':
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    elif start_from == 'tokenized':
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    raise ValueError(f"Unknown starting point: {start_from}")
