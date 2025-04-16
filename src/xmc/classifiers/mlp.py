import os
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score

from xmc.classifiers.base import BaseMalwareClassifier
from xmc.explainers.mlp import MalwareExplainerMLP
from xmc.utils import timer

# Cross-Validation f1_macro scores: [0.738, 0.7376, 0.7482, 0.7501, 0.7404, 0.7417, 0.742, 0.7612, 0.7519, 0.7439]
# Cross-Validation f1_macro mean:   0.7455
# Cross-Validation f1_macro std:    0.0074
# --------------------------------------------------
# Finished MalwareClassifierMLP.cross_validate() in 2362.65 secs
# Classification Report:
#                precision    recall  f1-score   support
#
#       adware       0.78      0.67      0.72       279
#     backdoor       0.85      0.75      0.80       366
#   downloader       0.70      0.78      0.74       225
#      dropper       0.50      0.66      0.56       169
#      spyware       0.55      0.50      0.52       160
#       trojan       0.94      0.95      0.95      2913
#        virus       0.91      0.93      0.92      1097
#        worms       0.77      0.65      0.70       359
#
#     accuracy                           0.87      5568
#    macro avg       0.75      0.74      0.74      5568
# weighted avg       0.87      0.87      0.87      5568
#
# --------------------------------------------------
# Finished MalwareClassifierMLP.train_and_evaluate() in 238.10 secs


class MalwareClassifierMLP(BaseMalwareClassifier):
    model_name = "mlp"
    explainer_class = MalwareExplainerMLP

    class MalwareNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(
        self,
        max_features: int = 1_000,
        ngram_range: tuple[int, int] = (1, 2),
        epochs: int = 300,
        patience: int | None = 50,
        batch_size: int = 256,
        hidden_dim: int = 256,
        learning_rate: float = 0.001,
        device: str | None = None,
        num_workers: int = -1,
        random_state: int = 69,
    ) -> None:
        super().__init__(
            max_features=max_features,
            ngram_range=ngram_range,
            use_scaler=True,
            random_state=random_state,
        )
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        # set after fit_transform
        self.model = None
        self.input_dim = None
        self.num_classes = None
        self.set_random_seed()

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
        self.input_dim = X.shape[1]
        self.num_classes = len(self.label_encoder.classes_)
        self.model = self.MalwareNet(self.input_dim, self.hidden_dim, self.num_classes)
        self.model.to(self.device)
        return X, y

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
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
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
        best_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        for epoch in range(self.epochs):
            train_loss = self._train_one_epoch(
                model, train_loader, criterion, optimizer
            )
            val_true, val_pred = self._evaluate(model, test_loader)
            curr_f1 = f1_score(val_true, val_pred, average="macro")
            print(
                f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Macro F1: {curr_f1:.4f}"
            )
            if self.patience is None:
                continue
            if curr_f1 > best_f1:
                best_f1 = curr_f1
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epoch(s).")
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break
        if best_model_state:
            model.load_state_dict(best_model_state)
        print("Training complete.")

    @timer
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cv_splits: int = 10,
        scoring: str = "f1_macro",
    ) -> None:
        """
        Performs cross-validation in a manner similar to scikit-learn.
        """
        kfold = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=self.random_state
        )
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            # re-init model for each fold
            fold_model = self.MalwareNet(
                self.input_dim, self.hidden_dim, self.num_classes
            ).to(self.device)
            X_train = self.scaler.fit_transform(X[train_idx])
            X_val = self.scaler.transform(X[val_idx])
            train_loader, val_loader, criterion, optimizer = (
                self._prepare_train_test_data(
                    fold_model, X_train, y[train_idx], X_val, y[val_idx]
                )
            )
            self._train(fold_model, train_loader, val_loader, criterion, optimizer)
            val_true, val_pred = self._evaluate(fold_model, val_loader)
            fold_score = f1_score(val_true, val_pred, average=scoring.split("_")[1])
            scores.append(float(fold_score))
            print(f"Fold {fold_idx}, {scoring}={fold_score:.4f}")

        self.display_cv_results(scoring, scores)

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
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        train_loader, test_loader, criterion, optimizer = self._prepare_train_test_data(
            self.model, X_train, y_train, X_test, y_test
        )
        self._train(self.model, train_loader, test_loader, criterion, optimizer)
        y_true_encoded, y_pred_encoded = self._evaluate(self.model, test_loader)
        y_true = self.label_encoder.inverse_transform(y_true_encoded)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        self.plot_confusion_matrix(y_true, y_pred)
        print("Classification Report:\n", classification_report(y_true, y_pred))
        print("-" * 50)
        self.save_model_artifacts(X_train, X_test, y_train, y_test)

    def get_model_artifacts(self) -> dict[str, Any]:
        artifacts = super().get_model_artifacts()
        artifacts.update(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": next(self.model.parameters()).shape[1],
                "hidden_dim": self.hidden_dim,
            }
        )
        return artifacts

    @classmethod
    def load_model_artifacts(cls) -> dict[str, Any]:
        artifacts = super().load_model_artifacts()
        # reconstruct the model
        num_classes = len(artifacts["label_encoder"].classes_)
        model = MalwareClassifierMLP.MalwareNet(
            artifacts["input_dim"], artifacts["hidden_dim"], num_classes
        )
        model.load_state_dict(artifacts["model_state_dict"])
        model.eval()
        artifacts["model"] = model
        return artifacts
