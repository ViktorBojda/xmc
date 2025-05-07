from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xmc.settings import DATASETS_DIR_PATH
from xmc.utils import prompt_overwrite, timer


class Preprocessor:
    """
    Preprocess malware datasets by cleaning API call data, removing outliers,
    dropping invalid classes, and merging multiple datasets into one.
    """

    MIN_CLASS_SIZE = 500
    DRZEHRA_DATASET_DIR = DATASETS_DIR_PATH / "drzehra"
    PREPROCESSED_DRZEHRA_FILE = DRZEHRA_DATASET_DIR / "preprocessed.csv"
    OCATAK_DATASET_DIR = DATASETS_DIR_PATH / "ocatak"
    PREPROCESSED_OCATAK_FILE = OCATAK_DATASET_DIR / "preprocessed.csv"
    KHAS_DATASET_DIR_PATH = DATASETS_DIR_PATH / "khas_ccip"
    KHAS_VIRUS_SAMPLE_FILE = KHAS_DATASET_DIR_PATH / "VirusSample.csv"
    KHAS_VIRUS_SHARE_FILE = KHAS_DATASET_DIR_PATH / "VirusShare.csv"
    PREPROCESSED_KHAS_VIRUS_SHARE_FILE = KHAS_DATASET_DIR_PATH / "preprocessed.csv"
    PREPROCESSED_MERGED_FILE = DATASETS_DIR_PATH / "preprocessed_merged.csv"

    removed_mangled_api_calls = 0

    def _aggregate_api_calls(self, row: pd.Series, data: dict):
        api_calls = []
        for col_name, value in row.items():
            if col_name == "sha256":
                continue
            api_calls.extend([col_name] * value)
        sha = row["sha256"]
        if not api_calls:
            print(f"INFO: No API calls found for sample with hash '{sha}'.")
        if sha in data:
            raise ValueError("Duplicate file hash in dataset.")
        data[sha] = {"api": api_calls}

    def _assign_malware_class(self, row: pd.Series, data: dict[str, Any]):
        sha = row["sha256"]
        cls = row["class"]
        if sha not in data:
            return
        if "class" in data[sha]:
            raise ValueError("Duplicate file hash in dataset.")
        data[sha]["class"] = cls

    def _remove_mangled_api_calls(self, api_string: str):
        api_list = api_string.split(",")
        filtered_list = [api for api in api_list if not api.startswith("?")]
        self.removed_mangled_api_calls += len(api_list) - len(filtered_list)
        clean_api_calls = ",".join(filtered_list)
        if clean_api_calls:
            return clean_api_calls
        return np.nan

    def _is_within_limits(self, row: pd.Series, limits: dict[str, tuple[int, int]]):
        cls = row["class"]
        lower, upper = limits[cls]
        if row["api_count"] < lower or row["api_count"] > upper:
            return np.nan
        return row

    def _log_dataset_stats(self, df: pd.DataFrame, log_path: Path) -> None:
        all_api_calls = set()
        for api_list in df["api"]:
            if api_list:
                all_api_calls.update(api_list.split(","))
        res_api_calls = "\n".join(sorted(all_api_calls))
        res = (
            f"Number of samples: {df.shape[0]}\n"
            f"Class distribution:\n"
            f"{df['class'].value_counts().to_string()}\n\n"
            f"Unique API calls:\n"
            f"{res_api_calls}\n"
        )
        print(res)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(res)

    def preprocess_drzehra_dataset(self) -> None:
        # drzehra dataset is not usable due to missing sequences of calls
        # and many malformed samples
        if not prompt_overwrite(self.PREPROCESSED_DRZEHRA_FILE):
            return
        df = pd.read_csv(self.DRZEHRA_DATASET_DIR / "api_calls.csv")
        df.drop(columns=["malware"], inplace=True)  # all are malware
        print("Rows before removing missing values:" + str(len(df)))
        df = df[df.drop(columns="sha256").sum(axis=1) > 0]
        df.dropna(inplace=True)
        print("Rows after removing missing values:" + str(len(df)))
        data = {}
        df.apply(lambda row: self._aggregate_api_calls(row, data), axis=1)
        df = pd.read_csv(self.DRZEHRA_DATASET_DIR / "labels.csv")
        df.apply(lambda row: self._assign_malware_class(row, data), axis=1)
        csv_data = [
            {"api": ",".join(values["api"]), "class": values["class"]}
            for sha, values in data.items()
            if "class" in values
        ]
        df_out = pd.DataFrame(csv_data)
        df_out.to_csv(self.PREPROCESSED_DRZEHRA_FILE, index=False)
        self._log_dataset_stats(df_out, self.DRZEHRA_DATASET_DIR / "log.txt")
        print("Successfully preprocessed drzehra dataset.")

    def preprocess_ocatak_dataset(self) -> None:
        if not prompt_overwrite(self.PREPROCESSED_OCATAK_FILE):
            return
        df = pd.read_csv(
            self.OCATAK_DATASET_DIR / "api_calls.txt",
            sep="/n",
            header=None,
            names=["api"],
        )
        df["api"] = df["api"].str.strip().str.replace(" ", ",")
        labels_df = pd.read_csv(self.OCATAK_DATASET_DIR / "labels.csv")
        df["class"] = labels_df["class"].values
        df.to_csv(self.PREPROCESSED_OCATAK_FILE, index=False)
        self._log_dataset_stats(df, self.OCATAK_DATASET_DIR / "log.txt")
        print("Successfully preprocessed ocatak dataset.")

    def preprocess_khas_dataset(self) -> None:
        if not prompt_overwrite(self.PREPROCESSED_KHAS_VIRUS_SHARE_FILE):
            return
        df = pd.read_csv(self.KHAS_VIRUS_SHARE_FILE)
        df["file"] = df["file"].str.replace("VirusShare_", "", regex=False)
        num_rows = len(df)
        df = df.drop_duplicates(subset="file")
        print(f"Number of removed duplicate hash rows: {num_rows - len(df)}")
        df.drop(columns=["file"], inplace=True)
        df.to_csv(self.PREPROCESSED_KHAS_VIRUS_SHARE_FILE, index=False)
        print("Successfully preprocessed khas VirusShare dataset.")

    def preprocess_datasets(self) -> None:
        self.preprocess_ocatak_dataset()
        self.preprocess_khas_dataset()

        df1 = pd.read_csv(self.PREPROCESSED_KHAS_VIRUS_SHARE_FILE)
        df2 = pd.read_csv(self.PREPROCESSED_OCATAK_FILE)
        log_text = ""

        def print_log(x):
            nonlocal log_text
            print(x, end="")
            log_text += x

        df = pd.concat([df1, df2], ignore_index=True)
        print_log(f"Rows before preprocessing: {len(df)}\n")
        print_log(
            f"Class distribution before preprocessing:\n{df['class'].value_counts()}\n"
        )

        df = df[~df["api"].astype(str).str.strip().eq("")]
        df.dropna(inplace=True)
        print_log(f"\nRows after removing missing values: {len(df)}\n")

        df["api"] = df["api"].apply(self._remove_mangled_api_calls)
        print_log(
            f"Number of removed mangled API calls: {self.removed_mangled_api_calls}\n"
            f"Rows after removing mangled API calls: {len(df)}\n"
        )

        df["class"] = df["class"].str.lower()
        df = df[(df["class"] != "undefined") & (df["class"] != "unknown")]
        print_log(f"Rows after removing class 'undefined' and 'unknown': {len(df)}\n")

        df["api_count"] = df["api"].apply(lambda x: len(x.split(",")))
        print_log(
            f'Minimum number of API calls: {df["api_count"].min()}\n'
            f'Maximum number of API calls: {df["api_count"].max()}\n'
        )

        print_log(
            f"\nAPI count limits for each class:\n"
            f"{'Class':<20}{'Lower Limit':<20}{'Upper Limit':<20}\n"
        )
        limits = {}
        for cls in df["class"].unique():
            subset = df[df["class"] == cls]
            lower_limit = int(subset["api_count"].quantile(0.05))
            upper_limit = int(subset["api_count"].quantile(0.95))
            limits[cls] = (lower_limit, upper_limit)
            print_log(f"{cls:<20}{lower_limit:<20}{upper_limit:<20}\n")

        df = df.apply(lambda row: self._is_within_limits(row, limits), axis=1)
        df.dropna(inplace=True)
        print_log(f"Rows after removing class specific outliers: {len(df)}\n")

        class_counts = df["class"].value_counts()
        print_log(
            f"\nClass distribution before removing classes with less than {self.MIN_CLASS_SIZE} samples:\n"
            f"{class_counts}\n"
        )

        classes_to_drop = class_counts[
            class_counts < self.MIN_CLASS_SIZE
        ].index.tolist()
        df = df[~df["class"].isin(classes_to_drop)]
        print_log(
            f"\nRows after removing classes with less than {self.MIN_CLASS_SIZE} samples: {len(df)}\n"
        )

        print_log(
            f'Minimum number of API calls: {df["api_count"].min()}\n'
            f'Maximum number of API calls: {df["api_count"].max()}\n'
        )
        print_log(f"\nRows after preprocessing: {len(df)}\n")
        print_log(
            f"Class distribution after preprocessing:\n{df['class'].value_counts()}\n"
        )

        (DATASETS_DIR_PATH / "log.txt").write_text(log_text)
        if not prompt_overwrite(self.PREPROCESSED_MERGED_FILE):
            return
        df.to_csv(self.PREPROCESSED_MERGED_FILE, index=False)
        print(
            f"Successfully preprocessed datasets. Saved to {self.PREPROCESSED_MERGED_FILE}"
        )

    @timer
    def run(self) -> None:
        self.preprocess_datasets()


if __name__ == "__main__":
    Preprocessor().run()
