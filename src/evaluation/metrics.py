from typing import Dict, Any
from src.config.config import Config
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, model: Any, data: Any) -> Dict[str, float]:
        X_test = np.stack(data['post'].values)
        y_test = data['female'].values

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Return metrics as a dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
