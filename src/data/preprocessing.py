from pathlib import Path
from typing import Any
import pandas as pd
from src.config.config import Config

class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config

    def process(self, data_path: Path) -> Any:
        df = pd.read_csv(data_path).rename(columns={"auhtor_ID": "author_ID"})

        if self.config.dev:
            df = df.sample(1000)

        print(f"Loaded {len(df)} rows of data")

        # Eliminate all authors with less than 300 characters
        df = df.groupby('author_ID').filter(lambda x: len(''.join(x['post'])) > self.config.min_chars_per_author)

        return df
