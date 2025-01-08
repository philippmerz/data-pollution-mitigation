from src.config.config import Config
import torch
import pandas as pd
from transformers import DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

class ClassifierTokenizer:
    def __init__(self, config: Config):
        self.config = config
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model.eval()

    def get_classifier_tokens(self, data: pd.DataFrame, save_path: str) -> pd.DataFrame:
        print('Generating classifier tokens for path ', save_path)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        self.model = self.model.to(device)

        self.model = self.model.half()

        CHUNK_SIZE = 5000
        total_embeddings = []

        for chunk_start in range(0, len(data), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(data))
            chunk_data = data.iloc[chunk_start:chunk_end]

            # Convert chunk to tensors
            input_tensor = torch.tensor(np.array(chunk_data['post'].tolist()), dtype=torch.long)
            mask_tensor = torch.tensor(np.array(chunk_data['attention_mask'].tolist()), dtype=torch.long)
            dataset = TensorDataset(input_tensor, mask_tensor)

            dataloader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                pin_memory=False,
                num_workers=0
            )

            chunk_embeddings = []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Processing chunk {chunk_start//CHUNK_SIZE + 1}"):
                    input_ids, attention_mask = [t.to(device) for t in batch]
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    cls_embedding = output.last_hidden_state[:, 0, :]
                    chunk_embeddings.extend(cls_embedding.cpu().float().numpy())

            # Save chunk results immediately
            temp_df = chunk_data.copy()
            temp_df['cls'] = chunk_embeddings
            temp_df.to_csv(f"{save_path}.chunk{chunk_start//CHUNK_SIZE}", index=False)

            # Clear memory
            del chunk_embeddings, dataset, input_tensor, mask_tensor
            torch.cuda.empty_cache()  # Clear CUDA cache
            import gc
            gc.collect()  # Run garbage collection

            total_embeddings.extend(chunk_embeddings)

        # Combine all chunks
        data['cls'] = total_embeddings
        data.to_csv(save_path, index=False)
        return data