from typing import Any

import numpy as np
import optuna
from imblearn.ensemble import BalancedRandomForestClassifier
from optuna.pruners import MedianPruner
from optuna.study import StudyDirection
from scipy.sparse import csc_matrix
from sklearn import clone
from sklearn.metrics import (
    classification_report,
    make_scorer,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate as sklearn_cross_validate,
)

from xmc.classifiers.base import BaseMalwareClassifier
from xmc.explainers.brf import MalwareExplainerBRF
from xmc.utils import timer


class MalwareClassifierBRF(BaseMalwareClassifier):
    model_name = "brf_1k"
    explainer_class = MalwareExplainerBRF

    def __init__(
        self,
        max_features: int = 1_000,
        ngram_range: tuple[int, int] = (1, 2),
        use_scaler: bool = False,  # counterfactuals work better with scaler
        n_estimators: int = 200,
        max_depth: int = 27,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        rf_max_features: str | None = 0.33,
        replacement: bool = True,
        bootstrap: bool = False,
        sampling_strategy: str = "not majority",
        verbose: int = 2,
        n_jobs: int = -1,
    ):
        super().__init__(
            max_features=max_features,
            ngram_range=ngram_range,
            use_scaler=use_scaler,
        )
        self.classifier = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            replacement=replacement,
            bootstrap=bootstrap,
            sampling_strategy=sampling_strategy,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=rf_max_features,
            random_state=self.random_state,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    @timer
    def _tune_hyperparameters(self, *, n_trials: int | None = 20):
        """Hyperparameter tuning, for development only"""
        self.log_write("Starting hyperparameter tuning (k=5)...\n")
        storage_path = "sqlite:///brf_study.db"
        study_name = "brf_optimization"
        X, y = self.load_and_transform_data()

        def objective(trial: optuna.Trial):
            model_params = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            }
            new_model_params = {
                "max_depth": trial.suggest_int("max_depth", 11, 31, step=2),
                "max_features": trial.suggest_float(
                    "max_features", 0.01, 1.0, step=0.01
                ),
            }
            model_params.update(new_model_params)
            self.reset_score_metrics()
            kfold = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            )
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                model: BalancedRandomForestClassifier = clone(self.classifier)
                model.set_params(**model_params, verbose=0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                self.calc_score_metrics(y_val, y_pred)

                trial.report(np.mean(self.score_metrics["f1_macro"]), step=fold_idx)
                if trial.should_prune():
                    rf_max_features = model_params.pop("max_features")
                    self.log_score_metrics(
                        {
                            "max_features": 1_000,
                            "n_gram": (1, 2),
                            **model_params,
                            "rf_max_features": rf_max_features,
                        },
                    )
                    raise optuna.TrialPruned()
            rf_max_features = model_params.pop("max_features")
            self.log_score_metrics(
                {
                    "max_features": 1_000,
                    "n_gram": (1, 2),
                    **model_params,
                    "rf_max_features": rf_max_features,
                },
            )
            return np.mean(self.score_metrics["f1_macro"])

        study = optuna.create_study(
            direction=StudyDirection.MAXIMIZE,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True,
        )
        study.optimize(
            objective, n_trials=n_trials, gc_after_trial=False, show_progress_bar=True
        )
        self.log_write(
            f"Hyperparameter tuning finished. Best hyperparameters: {study.best_trial.params}\n"
        )

    @timer
    def cross_validate(self, X: csc_matrix, y: np.ndarray, *, cv_splits: int) -> None:
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "precision_macro": make_scorer(precision_score, average="macro"),
            "precision_weighted": make_scorer(precision_score, average="weighted"),
            "recall_macro": make_scorer(recall_score, average="macro"),
            "recall_weighted": make_scorer(recall_score, average="weighted"),
            "f1_macro": make_scorer(f1_score, average="macro"),
            "f1_weighted": make_scorer(f1_score, average="weighted"),
        }
        kfold = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=self.random_state
        )
        cv_results = sklearn_cross_validate(
            self.classifier, X, y, cv=kfold, scoring=scoring, n_jobs=-1, verbose=1
        )
        for key, scores in cv_results.items():
            if key.startswith("test_"):
                metric = key[5:]
                self.score_metrics[metric] = scores
        self.log_score_metrics(weighted=True)

    @timer
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, *, test_size) -> None:
        """
        Splits data into train/test, fits the classifier on training set,
        predicts on the test set, decodes the labels, and prints a classification report.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        self.classifier.fit(X_train, y_train)
        y_pred_encoded = self.classifier.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_true = self.label_encoder.inverse_transform(y_test)
        print("Classification Report:\n", classification_report(y_true, y_pred))
        self.plot_confusion_matrix(y_true, y_pred)
        print("-" * 50)
        self.save_model_artifacts(X_train, X_test, y_train, y_test)

    def get_model_artifacts(self) -> dict[str, Any]:
        artifacts = super().get_model_artifacts()
        self.classifier.set_params(verbose=0)
        artifacts["model"] = self.classifier
        return artifacts
