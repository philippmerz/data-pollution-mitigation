from src.config.config import Config
from typing import Any
import torch
import pandas as pd
import transformers
from transformers import DistilBertTokenizerFast

class Tokenizer:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize the training validation and test data set
    def tokenize_datasets(self, data: list[pd.DataFrame], data_paths: list[str] ) :
        for i, dataframe in enumerate(data):
            print("Tokenizing training set")

            tokens = self.tokenize(dataframe)

            print("Saving tokenized data")

            torch.save(tokens, data_paths[i])

    def tokenize(self, data: pd.DataFrame) -> Any:
        # Tokenize the data
        posts = data['post'].tolist()

        tokens = self.tokenizer(posts, padding=True, truncation=True, return_tensors="pt")

        return tokens
