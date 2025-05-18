import gzip
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.pruners import MedianPruner
from optuna.study import StudyDirection
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score

from xmc.classifiers.base import BaseMalwareClassifier
from xmc.explainers.mlp import MalwareExplainerMLP
from xmc.settings import MODELS_DIR_PATH
from xmc.utils import timer


"""
Architecture: max_features: int = 1000, ngram_range: tuple[int, int] = (1, 2), num_layers: int = 1, hidden_dim: int = 256, activation: str = 'relu', dropout_rate: float = 0.2, batch_size: int = 64, learning_rate: float = 0.0025897653095650194, epochs: int = 500, patience: int | None = 50, device: str | None = None, num_workers: int = -1, {'layer_0': 'Linear(in_features=1000, out_features=256, bias=True)', 'layer_1': 'ReLU()', 'layer_2': 'Dropout(p=0.2, inplace=False)', 'layer_3': 'Linear(in_features=256, out_features=8, bias=True)'}
Cross-Validation Results Summary:
Metric                   Mean      SD
accuracy                 0.7726    0.0078
precision_macro          0.6988    0.0120
recall_macro             0.6823    0.0166
f1_macro                 0.6865    0.0126

Classification Report:
               precision    recall  f1-score   support

      adware      0.710     0.677     0.693       232
    backdoor      0.721     0.670     0.695       282
  downloader      0.762     0.655     0.704       220
     dropper      0.585     0.637     0.610       168
     spyware      0.517     0.475     0.495       158
      trojan      0.885     0.829     0.856      1788
       virus      0.674     0.878     0.763       655
        worm      0.623     0.565     0.593       283

    accuracy                          0.763      3786
   macro avg      0.685     0.673     0.676      3786
weighted avg      0.770     0.763     0.763      3786
"""


class MalwareClassifierMLP(BaseMalwareClassifier):
    model_name = "mlp_1k"
    explainer_class = MalwareExplainerMLP

    @classmethod
    def model_path(cls) -> Path:
        return MODELS_DIR_PATH / f"{cls.model_name}.pt.gz"

    def __init__(
        self,
        max_features: int = 1_000,
        ngram_range: tuple[int, int] = (1, 2),
        num_layers: int = 1,
        hidden_dim: int = 256,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        batch_size: int = 64,
        learning_rate: float = 0.0025897653095650194,
        epochs: int = 400,
        patience: int | None = 50,
        device: str | None = None,
        num_workers: int = -1,
    ) -> None:
        super().__init__(
            max_features=max_features, ngram_range=ngram_range, use_scaler=True
        )
        self.net_params = {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "activation": activation,
            "dropout_rate": dropout_rate,
        }
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()
        self.set_random_seed()
        self.model = None  # set after fit_transform

    class MalwareNet(nn.Module):
        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            num_layers: int,
            hidden_dim: int,
            activation: str,
            dropout_rate: float,
            *,
            epochs: int | None = None,
            patience: int | None = None,
            batch_size: int | None = None,
            learning_rate: float | None = None,
        ) -> None:
            super().__init__()
            layers = []
            in_dim = input_dim
            for i in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                if activation == "relu":
                    layers.append(nn.ReLU())
                if activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
                if activation == "elu":
                    layers.append(nn.ELU())
                layers.append(nn.Dropout(dropout_rate))
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.net = nn.Sequential(*layers)
            # required only for training
            self.epochs = epochs
            self.patience = patience
            self.batch_size = batch_size
            self.learning_rate = learning_rate

        def forward(self, x):
            return self.net(x)

    def set_random_seed(self) -> None:
        import random

        torch.manual_seed(self.random_state)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)

    def load_and_transform_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = super().load_and_transform_data()
        X = X.toarray()  # convert to dense for PyTorch
        self.net_params["input_dim"] = X.shape[1]
        self.net_params["num_classes"] = len(self.label_encoder.classes_)
        self.model = self.MalwareNet(
            **self.net_params,
            epochs=self.epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        ).to(self.device)
        self.init_params += f", {self.get_model_layers()}"
        return X, y

    def _tune_hyperparameters(self, *, n_trials: int | None = 10):
        """Hyperparameter tuning, for development only"""
        self.log_write("Starting hyperparameter tuning (k=5)...\n")
        storage_path = "sqlite:///mlp_1k_study.db"
        study_name = "mlp_1k_optimization"
        X, y = self.load_and_transform_data()
        input_dim = X.shape[1]
        num_classes = len(self.label_encoder.classes_)
        basic_params = {
            "max_features": 1000,
            "ngram_range": (1, 2),
        }

        def objective(trial: optuna.Trial):
            model_params = {
                "num_layers": 1,
                "hidden_dim": 256,
                "activation": "relu",
                "dropout_rate": 0.2,
                "batch_size": 64,
                "learning_rate": 0.0025897653095650194,
                "epochs": 300,
                "patience": 20,
            }
            new_model_params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [64, 128, 256, 512]
                ),
                "hidden_dim": trial.suggest_categorical(
                    "hidden_dim", [32, 64, 128, 256, 512]
                ),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.7, step=0.1),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "activation": trial.suggest_categorical(
                    "activation", ["relu", "leaky_relu", "elu"]
                ),
            }
            model_params.update(new_model_params)
            self.reset_score_metrics()
            kfold = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            )
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                fold_model = self.MalwareNet(input_dim, num_classes, **model_params).to(
                    self.device
                )
                y_val, y_pred = self._train_and_evaluate(
                    fold_model, X[train_idx], y[train_idx], X[val_idx], y[val_idx]
                )
                self.calc_score_metrics(y_val, y_pred)

                trial.report(np.mean(self.score_metrics["f1_macro"]), step=fold_idx)
                if trial.should_prune():
                    self.log_score_metrics({**basic_params, **model_params})
                    raise optuna.TrialPruned()
            self.log_score_metrics({**basic_params, **model_params})
            return np.mean(self.score_metrics["f1_macro"])

        study = optuna.create_study(
            direction=StudyDirection.MAXIMIZE,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            storage=storage_path,
            study_name=study_name,
            load_if_exists=True,
        )
        study.optimize(
            objective, n_trials=n_trials, gc_after_trial=False, show_progress_bar=True
        )
        self.log_write(
            f"Hyperparameter tuning finished. Best hyperparameters: {study.best_trial.params}\n"
        )

    def _prepare_train_test_data(
        self,
        model: MalwareNet,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[DataLoader, DataLoader, nn.Module, Optimizer]:
        # convert train and test folds to DataLoaders for batching, shuffle and performance
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=model.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=model.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)
        return train_loader, test_loader, criterion, optimizer

    def _train_one_epoch(
        self,
        model: MalwareNet,
        train_loader: DataLoader,
        criterion: Any,
        optimizer: Any,
    ) -> float:
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            outputs = model(batch_x.float())
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        return train_loss

    def _evaluate(
        self, model: MalwareNet, test_loader: DataLoader
    ) -> tuple[list, list]:
        model.eval()
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x.float())
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                pred_labels.extend(batch_preds)
                true_labels.extend(batch_y.numpy())
        return true_labels, pred_labels

    def _train(
        self,
        model: MalwareNet,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
    ) -> None:
        self.set_random_seed()
        best_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        for epoch in range(model.epochs):
            train_loss = self._train_one_epoch(
                model, train_loader, criterion, optimizer
            )
            val_true, val_pred = self._evaluate(model, test_loader)
            curr_f1 = f1_score(val_true, val_pred, average="macro")
            print(
                f"Epoch {epoch + 1}/{model.epochs}, Train Loss: {train_loss:.4f}, Val Macro F1: {curr_f1:.4f}"
            )
            if model.patience is None:
                continue
            if curr_f1 > best_f1:
                best_f1 = curr_f1
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epoch(s).")
                if patience_counter >= model.patience:
                    print("Early stopping triggered.")
                    break
        if best_model_state:
            model.load_state_dict(best_model_state)
        print("Training complete.")

    def _train_and_evaluate(
        self,
        model: MalwareNet,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        train_loader, test_loader, criterion, optimizer = self._prepare_train_test_data(
            model, X_train, y_train, X_test, y_test
        )
        self._train(model, train_loader, test_loader, criterion, optimizer)
        y_true, y_pred = self._evaluate(model, test_loader)
        return y_true, y_pred

    @timer
    def cross_validate(self, X: np.ndarray, y: np.ndarray, *, cv_splits) -> None:
        kfold = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=self.random_state
        )
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            # re-init model for each fold
            fold_model = self.MalwareNet(
                **self.net_params,
                epochs=self.epochs,
                patience=self.patience,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
            ).to(self.device)
            y_val, y_pred = self._train_and_evaluate(
                fold_model, X[train_idx], y[train_idx], X[val_idx], y[val_idx]
            )
            self.calc_score_metrics(y_val, y_pred)
            print(f"Fold {fold_idx}, f1_macro={self.score_metrics['f1_macro'][-1]:.4f}")
        self.log_score_metrics()

    @timer
    def train_and_evaluate(
        self, X: np.ndarray, y: np.ndarray, *, test_size: float = 0.2
    ) -> None:
        """
        Train on the training split, then evaluate on the test split.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        y_true_enc, y_pred_enc = self._train_and_evaluate(
            self.model, X_train, y_train, X_test, y_test
        )
        y_true = self.label_encoder.inverse_transform(y_true_enc)
        y_pred = self.label_encoder.inverse_transform(y_pred_enc)
        self.plot_confusion_matrix(y_true, y_pred, disp_model_name="MLP")
        print(
            "Classification Report:\n", classification_report(y_true, y_pred, digits=3)
        )
        print("-" * 50)
        self.save_model_artifacts(X_train, X_test, y_train, y_test)

    def get_model_layers(self) -> dict[str, str]:
        architecture = {}
        for i, layer in enumerate(self.model.net.children()):
            layer_name = f"layer_{i}"
            architecture[layer_name] = str(layer)
        return architecture

    def get_model_artifacts(self) -> dict[str, Any]:
        artifacts = super().get_model_artifacts()
        artifacts.update(
            {"model_state_dict": self.model.state_dict(), "net_params": self.net_params}
        )
        return artifacts

    def _save_model_artifacts(self, artifacts: dict[str, Any], path: Path) -> None:
        with gzip.open(path, "wb", compresslevel=6) as f:
            torch.save(artifacts, f)

    @classmethod
    def _load_model_artifacts(cls) -> dict[str, Any]:
        with gzip.open(cls.model_path(), "rb") as f:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.load(f, map_location=device, weights_only=False)

    @classmethod
    def load_model_artifacts(cls) -> dict[str, Any]:
        artifacts = super().load_model_artifacts()
        # reconstruct the model
        model = MalwareClassifierMLP.MalwareNet(**artifacts["net_params"])
        model.load_state_dict(artifacts["model_state_dict"])
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        artifacts["model"] = model
        return artifacts
