from src.config.config import Config
from typing import Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self, train_data: Any, val_data) -> Any:

        # XGBoost
        if self.config.model == 'xgboost':
            return train_xgboost(self.config)

        # Logistic Regression
        elif self.config.model == 'logistic-regression':
            return train_logistic_regression(self.config, train_data, val_data)

def train_xgboost(config: Config):
    pass

def train_logistic_regression(config: Config, train_data: Any, val_data: Any) -> Any:
    X_train = np.stack(train_data['post'].values)
    y_train = train_data['female'].values

    # Define a pipeline with scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=config.max_iter, random_state=config.random_seed))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    return pipeline
