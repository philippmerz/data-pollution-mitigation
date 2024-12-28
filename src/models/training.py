from src.config.config import Config
from typing import Any

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self, train_data: Any, val_data) -> Any:

        # XGBoost
        if self.config.model == 'xgboost':
            return train_xgboost(self.config)

        # Logistic Regression
        elif self.config.model == 'logistic-regression':
            return train_logistic_regression(self.config)

def train_xgboost(config: Config):
    pass

def train_logistic_regression(config: Config):
    pass
