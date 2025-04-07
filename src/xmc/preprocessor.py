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
    KHAS_VIRUS_SAMPLE_FILE = DATASETS_DIR_PATH / "khas_ccip/VirusSample.csv"
    KHAS_VIRUS_SHARE_FILE = DATASETS_DIR_PATH / "khas_ccip/VirusShare.csv"
    PREPROCESSED_MERGED_FILE = DATASETS_DIR_PATH / "preprocessed_merged.csv"

    removed_mangled_api_calls = 0

    def _aggregate_api_calls(self, row: pd.Series, data: dict):
        api_calls = []
        count = 0
        for col_name, value in row.items():
            count += 1
            if col_name == "sha256":
                continue
            api_calls.extend([col_name] * value)
        if not api_calls:
            return
        sha = row["sha256"]
        if sha in data:
            raise ValueError("Duplicate sha256 in dataset")
        data[sha] = {"api": api_calls}

    def _assign_malware_class(self, row: pd.Series, data: dict):
        sha = row["sha256"]
        if sha in data:
            data[sha]["class"] = row["class"]

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

    def preprocess_drzehra_dataset(self) -> None:
        if not prompt_overwrite(self.PREPROCESSED_DRZEHRA_FILE):
            return
        df = pd.read_csv(self.DRZEHRA_DATASET_DIR / "api_calls.csv")
        df.drop(columns=["malware"], inplace=True)  # all are malware
        data = {}
        df.apply(lambda row: self._aggregate_api_calls(row, data), axis=1)

        df = pd.read_csv(self.DRZEHRA_DATASET_DIR / "labels.csv")
        df.apply(lambda row: self._assign_malware_class(row, data), axis=1)

        csv_data = [
            {"api": ",".join(values["api"]), "class": values["class"]}
            for sha, values in data.items()
            if "class" in values
        ]
        pd.DataFrame(csv_data).to_csv(self.PREPROCESSED_DRZEHRA_FILE, index=False)
        print("Successfully preprocessed drzehra dataset")

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
        print("Successfully preprocessed ocatak dataset.")

    def preprocess_datasets(self) -> None:
        self.preprocess_drzehra_dataset()
        self.preprocess_ocatak_dataset()

        df1 = pd.read_csv(self.KHAS_VIRUS_SAMPLE_FILE)
        df2 = pd.read_csv(self.KHAS_VIRUS_SHARE_FILE)
        df3 = pd.read_csv(self.PREPROCESSED_DRZEHRA_FILE)
        df4 = pd.read_csv(self.PREPROCESSED_OCATAK_FILE)

        df1.drop(columns=["file"], inplace=True)
        df2.drop(columns=["file"], inplace=True)

        df = pd.concat([df1, df2, df3, df4], ignore_index=True)
        print("Rows before preprocess: " + str(len(df)))

        df["api"] = df["api"].apply(self._remove_mangled_api_calls)
        df.dropna(inplace=True)
        print(
            "Number of mangled API calls removed: "
            + str(self.removed_mangled_api_calls)
        )
        print("Rows after removing missing values: " + str(len(df)))

        df["class"] = df["class"].str.lower()
        df = df[(df["class"] != "undefined") & (df["class"] != "unknown")]
        print("Rows after removing class 'undefined' and 'unknown': " + str(len(df)))

        df["api_count"] = df["api"].apply(lambda x: len(x.split(",")))
        print(f'Minimum number of API calls: {df["api_count"].min()}')
        print(f'Maximum number of API calls: {df["api_count"].max()}')

        limits = {}
        for cls in df["class"].unique():
            subset = df[df["class"] == cls]
            lower_limit = int(subset["api_count"].quantile(0.05))
            upper_limit = int(subset["api_count"].quantile(0.95))
            limits[cls] = (lower_limit, upper_limit)
            print(
                f"Class: {cls}\t Lower Limit: {lower_limit}\t Upper Limit: {upper_limit}"
            )
        df = df.apply(lambda row: self._is_within_limits(row, limits), axis=1)
        df.dropna(inplace=True)
        print(f"Rows after removing class specific outliers: {str(len(df))}")

        print("Class distribution before removal:")
        class_counts = df["class"].value_counts()
        print(class_counts)

        classes_to_drop = class_counts[
            class_counts < self.MIN_CLASS_SIZE
        ].index.tolist()
        df = df[~df["class"].isin(classes_to_drop)]
        print(
            f"Rows after removing classes with less than {self.MIN_CLASS_SIZE} samples: {len(df)}"
        )

        print(f'Minimum number of API calls: {df["api_count"].min()}')
        print(f'Maximum number of API calls: {df["api_count"].max()}')

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
