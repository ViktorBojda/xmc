import functools
import sys
import time
from pathlib import Path
from typing import Callable, Any, NoReturn

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

from xmc.settings import (
    PLOTS_DIR_PATH,
    SLOVAK_TRANS_MAP,
    PAGE_WIDTH,
    PAGE_HEIGHT,
    DATASETS_DIR_PATH,
)


def prompt_overwrite(file_path: Path) -> bool:
    """If file exists, prompt user whether to overwrite it."""
    if not file_path.exists():
        return True
    while True:
        user_input = input(
            f"File '{file_path}' already exists. Overwrite? (y/n): "
        ).lower()
        if user_input == "y":
            return True
        if user_input == "n":
            print(f"Skipping overwrite. File '{file_path}' remains unchanged.")
            return False
        print("Please enter 'y' or 'n'.")


def timer(func: Callable) -> Callable:
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__qualname__}() in {run_time:.2f} secs")
        return value

    return wrapper


def prompt_options(
    options: dict[str, Any], *, multi_select: bool = False
) -> tuple[Any, str] | tuple[list[Any], list[str]]:
    option_names = list(options.keys())
    for i, option_name in enumerate(option_names):
        print(f"{i + 1}.", option_name)
    input_text = (
        "Enter the option number(s) separated by comma (e.g., 1,3) or 'all': "
        if multi_select
        else "Enter the option number: "
    )
    while True:
        user_input = input(input_text)
        try:
            if multi_select:
                if user_input.lower() == "all":
                    selected_option_names = option_names
                else:
                    indices = [int(i.strip()) - 1 for i in user_input.split(",")]
                    selected_option_names = [option_names[i] for i in indices]
            else:
                user_input = int(user_input) - 1
                selected_option_names = [option_names[user_input]]
            selected_options = [options[name] for name in selected_option_names]
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
            continue
        break
    return (
        (selected_options, selected_option_names)
        if multi_select
        else (selected_options[0], selected_option_names[0])
    )


def load_dataset(name: str) -> pd.DataFrame:
    dataset_path = DATASETS_DIR_PATH / name
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found in the datasets directory '{DATASETS_DIR_PATH}'."
        )
    return pd.read_csv(dataset_path)


def stratified_sample(
    X: np.ndarray, y: np.ndarray, *, size: int, random_state: int
) -> np.ndarray:
    sss = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=random_state)
    return X[next(sss.split(X, y))[0]]


class ShapUnavailable:
    def __getattr__(self, name: str) -> NoReturn:
        print(
            f"ERROR: SHAP is not installed. Attempted to access 'shap.{name}'.\n"
            "Please build it locally or install it with:\n"
            "pip install .[shap]\n"
        )
        sys.exit(1)


def try_import_shap():
    try:
        import shap

        return shap
    except ImportError:
        print("WARNING: SHAP is not installed. Some features may not work.")
        return ShapUnavailable()


def round_values(values: list[float], decimals: int = 4) -> list[float]:
    return [round(value, decimals) for value in values]


def set_plt_style():
    try:
        plt.style.use("xmc.thesis")
    except Exception:
        print(
            "WARNING: Failed to set style for Matplotlib, make sure LaTeX is installed."
        )
        plt.style.use("default")


def save_plot(title: str | None, save_as: str) -> None:
    if title:
        plt.title(title)
    plt.tight_layout()
    save_path = PLOTS_DIR_PATH / save_as
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=600)


def slovak_trans(name: str) -> str:
    return SLOVAK_TRANS_MAP.get(name, name)


def page_figsize(
    w_frac: float,
    h_frac: float,
    page_w_in: float = PAGE_WIDTH,
    page_h_in: float = PAGE_HEIGHT,
) -> tuple[float, float]:
    """
    Compute a Matplotlib figsize (in inches) relative to page size.
    """
    width = page_w_in * w_frac
    height = page_h_in * h_frac
    return width, height
