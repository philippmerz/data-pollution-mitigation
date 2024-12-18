from pathlib import Path
from typing import Any
import pandas as pd
from src.config.config import Config

class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config

    def process(self, data_path: Path) -> Any:
        df = pd.read_csv(data_path).rename(columns={"auhtor_ID": "author_ID"})
        return df

