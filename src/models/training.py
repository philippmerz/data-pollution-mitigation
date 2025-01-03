from src.config.config import Config
from typing import Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self, train_data: Any, val_data) -> Any:

        # XGBoost
        if self.config.model == 'xgboost':
            return train_xgboost(self.config, train_data, val_data)

        # Logistic Regression
        elif self.config.model == 'logistic-regression':
            return train_logistic_regression(self.config, train_data, val_data)

def train_xgboost(config: Config, train_data: Any, val_data: Any):

    X_train = train_data.cls
    vector_df = pd.DataFrame(X_train.tolist(), index=X_train.index)
    vector_df.columns = [f'cls_{i}' for i in range(768)]
    X_train = vector_df
    y_train = train_data.female

    model = XGBClassifier()

    param_grid = {'n_estimators': [50,100,200]}

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=420)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=kfold,
        verbose=1,
        n_jobs=-1)

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def train_logistic_regression(config: Config, train_data: Any, val_data: Any) -> Any:

    X_train = train_data.cls
    vector_df = pd.DataFrame(X_train.tolist(), index=X_train.index)
    vector_df.columns = [f'cls_{i}' for i in range(768)]
    X_train = vector_df
    y_train = train_data.female

    model = LogisticRegression()

    param_grid = {'C': [0.75, 1, 1.25]}

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=420)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=kfold,
        verbose=1,
        n_jobs=-1)

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_   