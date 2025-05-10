import logging
import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # alibi requires this
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
DATASETS_DIR_PATH = Path(DATASETS_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "classifiers/models")
MODELS_DIR_PATH = Path(MODELS_DIR)
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
PLOTS_DIR_PATH = Path(PLOTS_DIR)
EXPLANATIONS_DIR = os.path.join(ROOT_DIR, "explainers/explanations")
EXPLANATIONS_DIR_PATH = Path(EXPLANATIONS_DIR)

tf.get_logger().setLevel(logging.ERROR)  # suppress deprecation messages
tf.compat.v1.disable_v2_behavior()  # disable TF2 behaviour as alibi code still relies on TF1 constructs

PAGE_WIDTH = 6.10356  # inches
PAGE_HEIGHT = 9.33253  # inches

SLOVAK_TRANS_MAP = {
    "virus": "vírus",
    "worm": "červ",
    "adware": "advér",
    "spyware": "spajvér",
}
