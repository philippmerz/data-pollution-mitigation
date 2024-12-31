from src.config.config import Config
from typing import Any
import torch
import pandas as pd
from transformers import DistilBertModel

class Embedder:
    def __init__(self, config: Config):
        self.config = config
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model.eval()

    def embed(self, tokens: Any, save_path: str) -> pd.DataFrame:
        embeddings = []
        with torch.no_grad():
            for i in range(len(tokens['input_ids'])):
                input_ids = tokens['input_ids'][i].clone().detach().unsqueeze(0)
                attention_mask = tokens['attention_mask'][i].clone().detach().unsqueeze(0)
                
                # Generate embeddings
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = output.last_hidden_state[:, 0, :].squeeze(0)
                embeddings.append(cls_embedding.numpy())
        
        # Create a DataFrame with only embeddings
        df = pd.DataFrame({'embeddings': embeddings})
        
        try:
            df.to_csv(save_path, index=False)
        except Exception as e:
            print(f"Failed to save embeddings: {e}")
        
        return df
