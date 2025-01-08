from typing import Dict, Any
from src.config.config import Config
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from  src.data.tokenization import Tokenizer


class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, model: Any, data: Any) -> Dict[str, float]:
        X_test = data.cls
        vector_df = pd.DataFrame(X_test.tolist(), index=X_test.index)
        vector_df.columns = [f'cls_{i}' for i in range(768)]
        X_test = vector_df
        y_test = data.female

        if self.config.model == 'neural-network':
            # Evaluation for Neural Network
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

                predictions = model(X_test_tensor).squeeze()
                y_pred = (predictions >= 0.5).int().numpy()  # Threshold at 0.5 for binary classification
        else:
            # Evaluation for other scikit-learn models
            y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        embedder = Tokenizer(self.config)
        qualitative_analysis = pd.DataFrame({
            'post': embedder.tokens_to_string(data.post.tolist()),
            'true_label': y_test,
            'predicted_label': y_pred
        })

        qualitative_analysis.to_csv(self.config.qualitative_analysis_path, index=False)

        # Return metrics as a dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
