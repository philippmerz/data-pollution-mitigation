from src.config.config import Config
from typing import Any

class Embedder:
    def __init__(self, config: Config):
        self.config = config

    def embed(self, tokens: Any) -> Any:
        # Implementation
        pass
