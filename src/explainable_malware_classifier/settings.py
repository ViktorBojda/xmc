import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
DATASETS_DIR_PATH = Path(DATASETS_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "classifiers/models")
MODELS_DIR_PATH = Path(MODELS_DIR)
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
PLOTS_DIR_PATH = Path(PLOTS_DIR)
EXPLANATIONS_DIR = os.path.join(ROOT_DIR, "explainers/explanations")
EXPLANATIONS_DIR_PATH = Path(EXPLANATIONS_DIR)
