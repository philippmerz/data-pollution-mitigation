from src.config.config import Config
from typing import Any
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast

class Tokenizer:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenizes data and saves test, val and train tokens to file via torch.save
    def tokenize_datasets(self, data: list[pd.DataFrame], data_paths: list[str] ) :
        for i, dataframe in enumerate(data):
            print("Tokenizing training set")

            tokens = self.tokenize(dataframe)

            print("Saving tokenized data")

            torch.save(tokens, data_paths[i])

    def tokenize(self, data: list[str]) -> list[list[int]]:

        # The posts are not truncated to 512 tokens yet only padded to 512 tokens
        tokens = self.tokenizer(data, truncation=False, add_special_tokens=False)

        return tokens['input_ids']
