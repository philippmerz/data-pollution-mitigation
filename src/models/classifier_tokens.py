from src.config.config import Config
import torch
import pandas as pd
from transformers import DistilBertModel
from torch.utils.data import DataLoader, TensorDataset


class ClassifierTokenizer:
    def __init__(self, config: Config):
        self.config = config
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model.eval()

    def get_classifier_tokens(self, data: pd.DataFrame, save_path: str) -> pd.DataFrame:
        print('Generating classifier tokens for path ', save_path)
        embeddings = []

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        self.model = self.model.to(device)

        dataset = TensorDataset(torch.tensor(data['post'].tolist()),
                                torch.tensor(data['attention_mask'].tolist()))
        dataloader = DataLoader(dataset,
                                batch_size=self.config.embedding_batch_size,
                                pin_memory=True,
                                num_workers=4,
                                prefetch_factor=2)

        batch_number = 0

        with torch.no_grad():
            for batch in dataloader:
                print(f'Batch {batch_number}')
                input_ids, attention_mask = [t.to(device) for t in batch]

                # Generate embeddings
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = output.last_hidden_state[:, 0, :].squeeze(0).cpu()
                embeddings.extend(cls_embedding.numpy())

        data['cls'] = embeddings

        print('generated:')
        print(data.head())

        try:
            data.to_csv(save_path, index=False)
        except Exception as e:
            print(f"Failed to save embeddings: {e}")

        return data
