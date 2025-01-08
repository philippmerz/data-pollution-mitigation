from src.config.config import Config
from typing import Any
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self, train_data: Any, val_data: Any) -> Any:
        print('training ', self.config.model)
        if self.config.model == 'xgboost':
            return train_xgboost(self.config, train_data, val_data)

        elif self.config.model == 'logistic-regression':
            return train_logistic_regression(self.config, train_data, val_data)

        elif self.config.model == 'neural-network':
            return train_neural_network(self.config, train_data, val_data)

    def load_model(self) -> Any:
        if self.config.model == 'xgboost':
            model = XGBClassifier()
            model.load_model(self.config.xgboost_model_path)
            return model

        elif self.config.model == 'logistic-regression':
            return joblib.load(self.config.lr_model_path)

        elif self.config.model == 'neural-network':
            return torch.load(self.config.nn_model_path)


def train_xgboost(config: Config, train_data: Any, val_data: Any):
    # Prepare training data
    X_train = pd.DataFrame(
        train_data.cls.tolist(),
        index=train_data.cls.index,
        columns=[f'cls_{i}' for i in range(768)]
    )
    y_train = train_data.female

    # Prepare validation data
    X_val = pd.DataFrame(
        val_data.cls.tolist(),
        index=val_data.cls.index,
        columns=[f'cls_{i}' for i in range(768)]
    )
    y_val = val_data.female

    model = XGBClassifier(
        eval_metric=['error', 'auc'],
        early_stopping_rounds=10
    )

    param_grid = {'n_estimators': [50, 100, 200]}

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)]
    )

    # Save best model
    grid_search.best_estimator_.save_model(config.xgboost_model_path)

    return grid_search.best_estimator_


def train_logistic_regression(config: Config, train_data: Any, val_data: Any) -> Any:
    X_train = train_data.cls
    vector_df = pd.DataFrame(X_train.tolist(), index=X_train.index)
    vector_df.columns = [f'cls_{i}' for i in range(768)]
    X_train = vector_df
    y_train = train_data.female

    model = LogisticRegression(max_iter=config.max_iter, random_state=config.random_seed)

    param_grid = {'C': [0.75, 1, 1.25]}

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=kfold,
        n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Save best model
    joblib.dump(grid_search.best_estimator_, config.lr_model_path)

    return grid_search.best_estimator_


# Neural Network Training Function
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_neural_network(config: Config, train_data: Any, val_data: Any) -> Any:
    # Prepare training data
    X_train = pd.DataFrame(train_data.cls.tolist(), index=train_data.index, columns=[f'cls_{i}' for i in range(768)])
    y_train = train_data.female

    X_val = pd.DataFrame(val_data.cls.tolist(), index=val_data.index, columns=[f'cls_{i}' for i in range(768)])
    y_val = val_data.female

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.nn_batch_size, shuffle=True)

    # Initialize the model
    model = NeuralNetwork(
        input_dim=768,
        hidden_layers=config.nn_hidden_layers,
        dropout=config.nn_dropout
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.nn_learning_rate)

    # Tracking best model
    best_accuracy = 0.0
    best_epoch = 0
    best_model_state = None

    # Training loop
    for epoch in range(config.nn_epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.nn_epochs}"):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation step (accuracy)
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(X_val.values, dtype=torch.float32))
            val_preds = (val_outputs.squeeze() >= 0.5).int().numpy()
            val_accuracy = accuracy_score(y_val, val_preds)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Check if current model is the best
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1  # epoch is zero-indexed, add 1 for display
            best_model_state = model.state_dict()

    print(f"\n✅ Best model kept from Epoch {best_epoch} with Validation Accuracy: {best_accuracy:.4f}")

    # Load best model before returning
    model.load_state_dict(best_model_state)

    # Save model to file
    torch.save(model, config.nn_model_path)
    return model
