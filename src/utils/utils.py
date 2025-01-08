from pathlib import Path
from typing import Literal, Any, List

PipelineStage = Literal['raw', 'preprocessed', 'classifier_tokens', 'trained_model']
Models = Literal['neural-network', 'xgboost', 'logistic-regression']

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

def make_should_run(stage_order: List[PipelineStage]):
    def should_run(start_from: PipelineStage, current_stage: PipelineStage) -> bool:
        return stage_order.index(current_stage) >= stage_order.index(start_from)
    return should_run
