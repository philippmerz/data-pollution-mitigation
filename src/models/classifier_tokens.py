from src.config.config import Config
import torch
import pandas as pd
from transformers import DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import psutil
import gc

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    gb_used = mem_info.rss / (1024 * 1024 * 1024)  # Convert to GB
    swap = psutil.swap_memory()
    swap_used = swap.used / (1024 * 1024 * 1024)  # Convert to GB
    return gb_used, swap_used


class ClassifierTokenizer:
    def __init__(self, config: Config):
        self.config = config
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model.eval()

    def get_classifier_tokens(self, data: pd.DataFrame, save_path: str) -> pd.DataFrame:
        device = torch.device("mps")

        gb_used, swap_used = get_memory_usage()
        print(f"\nInitial Memory State:")
        print(f"RAM Usage: {gb_used:.2f} GB")
        print(f"Swap Usage: {swap_used:.2f} GB")

        self.model = self.model.to(device)
        self.model = self.model.half()

        total_embeddings = []
        CHUNK_SIZE = 5000

        for chunk_start in range(0, len(data), CHUNK_SIZE):
            print(f"\nProcessing chunk starting at {chunk_start}")
            gb_used, swap_used = get_memory_usage()
            print(f"Before chunk RAM: {gb_used:.2f} GB, Swap: {swap_used:.2f} GB")

            chunk_end = min(chunk_start + CHUNK_SIZE, len(data))
            chunk_data = data.iloc[chunk_start:chunk_end]

            input_tensor = torch.tensor(np.array(chunk_data['post'].tolist()), dtype=torch.long)
            mask_tensor = torch.tensor(np.array(chunk_data['attention_mask'].tolist()), dtype=torch.long)
            print(f"Input tensor shape: {input_tensor.shape}")

            gb_used, swap_used = get_memory_usage()
            print(f"After tensor creation RAM: {gb_used:.2f} GB, Swap: {swap_used:.2f} GB")

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
                for batch in tqdm(dataloader, desc=f"Processing chunk {chunk_start // CHUNK_SIZE + 1}"):
                    input_ids, attention_mask = [t.to(device) for t in batch]
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    cls_embedding = output.last_hidden_state[:, 0, :]
                    chunk_embeddings.extend(cls_embedding.cpu().float().numpy())

            total_embeddings.extend(chunk_embeddings)

            # Save chunk results
            temp_df = chunk_data.copy()
            temp_df['cls'] = chunk_embeddings
            temp_df.to_csv(f"{save_path}.chunk{chunk_start // CHUNK_SIZE}", index=False)

            gb_used, swap_used = get_memory_usage()
            print(f"Before cleanup RAM: {gb_used:.2f} GB, Swap: {swap_used:.2f} GB")

            # Clean up
            del chunk_embeddings, dataset, input_tensor, mask_tensor
            torch.cuda.empty_cache()
            gc.collect()

            gb_used, swap_used = get_memory_usage()
            print(f"After cleanup RAM: {gb_used:.2f} GB, Swap: {swap_used:.2f} GB\n")
            print("-" * 80)

        print("\nProcessing complete!")
        gb_used, swap_used = get_memory_usage()
        print(f"Final RAM: {gb_used:.2f} GB, Swap: {swap_used:.2f} GB")

        # Final save
        data['cls'] = total_embeddings
        data.to_csv(save_path, index=False)
        return data
