from typing import Any

import numpy as np
import optuna
import torch
from optuna import Trial, TrialPruned
from optuna.pruners import MedianPruner
from optuna.study import StudyDirection
from scipy.sparse import csc_matrix
from sklearn import clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from xmc.classifiers.base import BaseMalwareClassifier
from xmc.explainers.xgb import MalwareExplainerXGB
from xmc.utils import timer

# Cross-Validation f1_macro scores: [0.7763, 0.772, 0.7914, 0.7917, 0.7762, 0.7826, 0.7781, 0.7804, 0.7922, 0.7565]
# Cross-Validation f1_macro mean:   0.7797
# Cross-Validation f1_macro std:    0.0109
# --------------------------------------------------
# Finished MalwareClassifierXGB.cross_validate() in 272.61 secs
# Classification Report:
#                precision    recall  f1-score   support
#
#       adware       0.83      0.67      0.74       279
#     backdoor       0.86      0.80      0.83       366
#   downloader       0.72      0.79      0.75       225
#      dropper       0.66      0.69      0.68       169
#      spyware       0.66      0.53      0.58       160
#       trojan       0.93      0.96      0.94      2913
#        virus       0.94      0.94      0.94      1097
#        worms       0.75      0.70      0.72       359
#
#     accuracy                           0.89      5568
#    macro avg       0.79      0.76      0.77      5568
# weighted avg       0.88      0.89      0.88      5568
#
# --------------------------------------------------
# Finished MalwareClassifierXGB.train_and_evaluate() in 31.82 secs


class MalwareClassifierXGB(BaseMalwareClassifier):
    model_name = "xgb"
    explainer_class = MalwareExplainerXGB

    def __init__(
        self,
        max_features: int = 1_000,
        ngram_range: tuple[int, int] = (1, 2),
        patience: int | None = 50,
        max_depth: int = 20,
        learning_rate: float = 0.1837,
        n_estimators: int = 400,
        subsample: float = 0.7692,
        colsample_bytree: float = 0.6295,
        min_child_weight: float = 3,
        gamma: float = 0.00003,
        tree_method: str = "hist",
        random_state: int = 69,
        verbosity: int = 2,
        device: str | None = None,
        n_jobs=None,
    ):
        self.random_state = random_state
        # perform dirty check, if torch can use cuda so should xgboost
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.patience = patience
        self.vectorizer = CountVectorizer(
            tokenizer=self.comma_tokenizer,
            token_pattern=None,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.label_encoder = LabelEncoder()
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
            tree_method=tree_method,
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

    def tune_hyperparameters(
        self, X: csc_matrix, y: np.ndarray, *, n_trials: int | None
    ):
        # TODO: needs more tuning
        # Best hyperparameters: {'max_depth': 10, 'learning_rate': 0.1836571052713742, 'n_estimators': 400, 'subsample': 0.7692127413124975, 'colsample_bytree': 0.6294841326490792, 'min_child_weight': 3, 'gamma': 3.743302045781916e-05}
        # Best hyperparameters: {'max_depth': 10, 'min_child_weight': 0.0006939313862091159}
        # (1000 f) {"max_depth": 19,"min_child_weight": 2.9711383954026384e-10,"subsample": 0.8787367855578032,"colsample_bytree": 0.6359235255750173,"learning_rate": 0.1072635743723312,"n_estimators": 763}
        def objective(trial: Trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 1, 30),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1e-10, 1e10, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0, 1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "gamma": trial.suggest_float("gamma", 0, 0.01),
            }
            kfold = StratifiedKFold(
                n_splits=4, shuffle=True, random_state=self.random_state
            )
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                model: XGBClassifier = clone(self.classifier)
                model.set_params(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average="macro")
                scores.append(score)

                trial.report(np.mean(scores), step=fold_idx)
                if trial.should_prune():
                    raise TrialPruned()
            return np.mean(scores)

        study = optuna.create_study(
            direction=StudyDirection.MAXIMIZE,
            pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        )
        study.optimize(
            objective, n_trials=n_trials, gc_after_trial=False, show_progress_bar=True
        )
        print("Best hyperparameters:", study.best_trial.params)
        self.classifier.set_params(**study.best_trial.params)

    @timer
    def cross_validate(
        self,
        X: csc_matrix,
        y: np.ndarray,
        *,
        cv_splits: int,
        scoring: str = "f1_macro",
    ):
        kfold = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=self.random_state
        )
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            fold_model: XGBClassifier = clone(self.classifier)
            fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            y_pred = fold_model.predict(X_val)
            fold_score = f1_score(y_val, y_pred, average=scoring.split("_")[1])
            scores.append(float(fold_score))
            print(f"Fold {fold_idx}, {scoring}={fold_score:.4f}")

        self.display_cv_results(scoring, scores)

    @timer
    def train_and_evaluate(
        self, X: np.ndarray, y: np.ndarray, *, test_size: float = 0.2
    ) -> None:
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
