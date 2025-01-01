from pathlib import Path
from typing import Literal, Any, List
import torch
import pandas as pd
import ast

PipelineStage = Literal['raw', 'preprocessed', 'classifier_tokens']

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

def load_tokens(file_path: str) -> dict: 
    # Load the DataFrame 
    data = pd.read_csv(file_path)
    
    # Convert the string representations of lists to lists
    if isinstance(data['post'].iloc[0], str): 
        data['post'] = data['post'].apply(ast.literal_eval)

    if isinstance(data['attention_mask'].iloc[0], str): 
        data['attention_mask'] = data['attention_mask'].apply(ast.literal_eval)

    # Convert the lists to tensors
    input_ids = torch.tensor(data['post'].tolist()) 
    attention_mask = torch.tensor(data['attention_mask'].tolist()) 
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def make_should_run(stage_order: List[PipelineStage]):
    def should_run(start_from: PipelineStage, current_stage: PipelineStage) -> bool:
        return stage_order.index(current_stage) >= stage_order.index(start_from)
    return should_run
