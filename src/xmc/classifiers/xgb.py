from typing import Any

import numpy as np
import optuna
import torch
from optuna import Trial, TrialPruned
from optuna.pruners import MedianPruner
from optuna.study import StudyDirection
from scipy.sparse import csc_matrix
from sklearn import clone
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from xmc.classifiers.base import BaseMalwareClassifier
from xmc.explainers.xgb import MalwareExplainerXGB
from xmc.utils import timer

# Architecture: max_features: int = 1000, ngram_range: tuple[int, int] = (1, 2), patience: int | None = 50, max_depth: int = 8, min_child_weight: float = 1.0, subsample: float = 0.7, colsample_bytree: float = 0.45, learning_rate: float = 0.11, n_estimators: int = 350, gamma: float = 0.0, verbosity: int = 2, device: str | None = None, n_jobs: int | None = None
# Cross-Validation Results Summary:
# Metric                   Mean      SD
# accuracy                 0.8033    0.0080
# precision_macro          0.7495    0.0095
# recall_macro             0.7259    0.0146
# f1_macro                 0.7330    0.0121


class MalwareClassifierXGB(BaseMalwareClassifier):
    model_name = "xgb_1k"
    explainer_class = MalwareExplainerXGB

    def __init__(
        self,
        max_features: int = 1_000,
        ngram_range: tuple[int, int] = (1, 2),
        patience: int | None = 50,
        max_depth: int = 8,
        min_child_weight: float = 1.0,
        subsample: float = 0.7,
        colsample_bytree: float = 0.45,
        learning_rate: float = 0.11,
        n_estimators: int = 350,
        gamma: float = 0.0,
        verbosity: int = 2,
        device: str | None = None,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(
            max_features=max_features, ngram_range=ngram_range, use_scaler=False
        )
        # perform dirty check, if torch can use cuda so should xgboost
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.patience = patience
        callbacks = []
        if patience:
            callbacks.append(
                EarlyStopping(
                    rounds=self.patience,
                    save_best=True,
                    maximize=True,
                    metric_name="f1_macro",
                )
            )
        self.classifier = XGBClassifier(
            objective="multi:softmax",
            eval_metric=self.f1_macro,
            callbacks=callbacks,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            gamma=gamma,
            booster="gbtree",
            tree_method="hist",
            random_state=self.random_state,
            verbosity=verbosity,
            n_jobs=n_jobs,
            device=device,
            disable_default_eval_metric=True,
        )

    @staticmethod
    def f1_macro(y_pred: np.ndarray, y_true: np.ndarray):
        score = f1_score(y_true, y_pred, average="macro")
        return score

    def _tune_hyperparameters(self, *, n_trials: int | None = 10):
        """Hyperparameter tuning, for development only"""
        self.log_write("Starting hyperparameter tuning (k=5)...\n")
        storage_path = "sqlite:///xgb_study.db"
        study_name = "xgb_optimization"
        X, y = self.load_and_transform_data()
        self.patience = 20
        basic_params = {
            "max_features": 1000,
            "ngram_range": (1, 2),
            "patience": self.patience,
        }

        def objective(trial: Trial):
            # default_model_params = {
            #     "max_depth": 6,
            #     "min_child_weight": 1,
            #     "subsample": 1,
            #     "colsample_bytree": 1,
            #     "learning_rate": 0.3,
            #     "n_estimators": 100,
            #     "gamma": 0,
            # }
            model_params = {
                "max_depth": 8,
                "min_child_weight": 1,
                "subsample": 0.7,
                "colsample_bytree": 0.45,
                "learning_rate": 0.11,
                "n_estimators": 350,
                "gamma": 0,
            }
            new_model_params = {
                "max_depth": trial.suggest_int("max_depth", 6, 30, step=2),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 0.01, 1.0, step=0.01
                ),
                "subsample": trial.suggest_float("subsample", 0.3, 1.0, step=0.05),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.3, 1.0, step=0.05
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.3, step=0.001
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=25),
                "gamma": trial.suggest_float("gamma", 0, 1, step=0.01),
            }
            model_params.update(new_model_params)
            self.reset_score_metrics()
            kfold = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            )
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                model: XGBClassifier = clone(self.classifier)
                model.set_params(**model_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
                self.calc_score_metrics(y_val, y_pred)

                trial.report(np.mean(self.score_metrics["f1_macro"]), step=fold_idx)
                if trial.should_prune():
                    self.log_score_metrics({**basic_params, **model_params})
                    raise TrialPruned()
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

    @timer
    def cross_validate(self, X: csc_matrix, y: np.ndarray, *, cv_splits: int):
        kfold = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=self.random_state
        )
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            fold_model: XGBClassifier = clone(self.classifier)
            fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            y_pred = fold_model.predict(X_val)
            self.calc_score_metrics(y_val, y_pred)
            print(f"Fold {fold_idx}, f1_macro={self.score_metrics['f1_macro'][-1]:.4f}")
        self.log_score_metrics()

    @timer
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, *, test_size) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        self.classifier.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        y_pred_encoded = self.classifier.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_true = self.label_encoder.inverse_transform(y_test)
        self.plot_confusion_matrix(y_true, y_pred)
        print("Classification Report:\n", classification_report(y_true, y_pred))
        print("-" * 50)
        self.save_model_artifacts(X_train, X_test, y_train, y_test)

    def get_model_artifacts(self) -> dict[str, Any]:
        artifacts = super().get_model_artifacts()
        self.classifier.set_params(verbosity=1)
        artifacts["model"] = self.classifier
        return artifacts
