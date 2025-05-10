from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from xmc.classifiers.utils import save_plot, page_figsize, set_plt_style, slovak_trans
from xmc.settings import DATASETS_DIR_PATH
from xmc.utils import prompt_overwrite, timer


class Preprocessor:
    MIN_CLASS_SIZE = 700
    DRZEHRA_DATASET_DIR_PATH = DATASETS_DIR_PATH / "drzehra"
    DRZEHRA_PREPROC_DATASET = DRZEHRA_DATASET_DIR_PATH / "drzehra_preproc_dataset.csv"
    OCATAK_DATASET_DIR_PATH = DATASETS_DIR_PATH / "ocatak"
    OCATAK_PREPROC_DATASET = OCATAK_DATASET_DIR_PATH / "ocatak_preproc_dataset.csv"
    KHAS_DATASET_DIR_PATH = DATASETS_DIR_PATH / "khas_ccip"
    KHAS_VIRUS_SAMPLE_DATASET = KHAS_DATASET_DIR_PATH / "virus_sample.csv"
    KHAS_VIRUS_SHARE_DATASET = KHAS_DATASET_DIR_PATH / "virus_share.csv"
    KHAS_VIRUS_SHARE_PREPROC_DATASET = (
        KHAS_DATASET_DIR_PATH / "virus_share_preproc_dataset.csv"
    )
    FINAL_DATASET = DATASETS_DIR_PATH / "malware_dataset.csv"

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
        filtered_list = [
            api
            for api in api_list
            if (not api.startswith("?") and not api.startswith("_Z"))
        ]
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
        if not prompt_overwrite(self.DRZEHRA_PREPROC_DATASET):
            return
        df = pd.read_csv(self.DRZEHRA_DATASET_DIR_PATH / "api_calls.csv")
        df.drop(columns=["malware"], inplace=True)  # all are malware
        print("Rows before removing missing values:" + str(len(df)))
        df = df[df.drop(columns="sha256").sum(axis=1) > 0]
        df.dropna(inplace=True)
        print("Rows after removing missing values:" + str(len(df)))
        data = {}
        df.apply(lambda row: self._aggregate_api_calls(row, data), axis=1)
        df = pd.read_csv(self.DRZEHRA_DATASET_DIR_PATH / "labels.csv")
        df.apply(lambda row: self._assign_malware_class(row, data), axis=1)
        csv_data = [
            {"api": ",".join(values["api"]), "class": values["class"]}
            for sha, values in data.items()
            if "class" in values
        ]
        df_out = pd.DataFrame(csv_data)
        df_out.to_csv(self.DRZEHRA_PREPROC_DATASET, index=False)
        self._log_dataset_stats(
            df_out, self.DRZEHRA_DATASET_DIR_PATH / "drzehra_preproc_log.txt"
        )
        print("Successfully preprocessed drzehra dataset.")

    def preprocess_ocatak_dataset(self) -> None:
        if not prompt_overwrite(self.OCATAK_PREPROC_DATASET):
            return
        df = pd.read_csv(
            self.OCATAK_DATASET_DIR_PATH / "api_calls.txt",
            sep="/n",
            header=None,
            names=["api"],
        )
        df["api"] = df["api"].str.strip().str.replace(" ", ",")
        labels_df = pd.read_csv(self.OCATAK_DATASET_DIR_PATH / "labels.csv")
        df["class"] = labels_df["class"].values
        df.to_csv(self.OCATAK_PREPROC_DATASET, index=False)
        self._log_dataset_stats(
            df, self.OCATAK_DATASET_DIR_PATH / "ocatak_preproc_log.txt"
        )
        print("Successfully preprocessed ocatak dataset.")

    def preprocess_khas_dataset(self) -> None:
        if not prompt_overwrite(self.KHAS_VIRUS_SHARE_PREPROC_DATASET):
            return
        virus_share = pd.read_csv(self.KHAS_VIRUS_SHARE_DATASET)
        virus_sample = pd.read_csv(self.KHAS_VIRUS_SAMPLE_DATASET)
        virus_share_hashes = (
            virus_share["file"]
            .str.strip()
            .str.lower()
            .str.replace("virusshare_", "", regex=False)
        )
        virus_sample_hashes = virus_sample["file"].str.strip().str.lower()
        overlap = virus_sample_hashes.isin(virus_share_hashes)
        assert overlap.sum() == len(virus_sample)
        print(
            f"All VirusSample hashes are also in VirusShare. Skipping VirusSample dataset."
        )
        num_rows = len(virus_share)
        virus_share = virus_share.drop_duplicates(subset="file")
        print(
            f"Number of removed duplicate hash rows in VirusShare: {num_rows - len(virus_share)}"
        )
        virus_share.drop(columns=["file"], inplace=True)
        virus_share.to_csv(self.KHAS_VIRUS_SHARE_PREPROC_DATASET, index=False)
        self._log_dataset_stats(
            virus_share, self.KHAS_DATASET_DIR_PATH / "virus_share_preproc_log.txt"
        )
        print("Successfully preprocessed khas datasets.")

    def preprocess_datasets(self) -> None:
        if not prompt_overwrite(self.FINAL_DATASET):
            return

        self.preprocess_khas_dataset()
        self.preprocess_ocatak_dataset()

        df1 = pd.read_csv(self.KHAS_VIRUS_SHARE_PREPROC_DATASET)
        df2 = pd.read_csv(self.OCATAK_PREPROC_DATASET)
        log_text = ""

        def print_log(x):
            nonlocal log_text
            print(x, end="")
            log_text += x

        df = pd.concat([df1, df2], ignore_index=True)
        df["class"] = df["class"].str.lower()
        class_counts = df["class"].value_counts()
        unique_apis = set(
            api.strip().lower() for apis in df["api"] for api in apis.split(",")
        )
        print_log(
            f"Rows before preprocessing: {len(df)}\n"
            f"Number of unique API calls before preprocessing: {len(unique_apis)}\n"
            f"Class distribution before preprocessing:\n{class_counts}\n\n"
        )

        classes_to_drop = class_counts[
            class_counts < self.MIN_CLASS_SIZE
        ].index.tolist()
        df = df[~df["class"].isin(classes_to_drop)]
        print_log(
            f"Rows after removing classes with less than {self.MIN_CLASS_SIZE} samples: {len(df)}\n"
        )

        self.removed_mangled_api_calls = 0
        df["api"] = df["api"].apply(self._remove_mangled_api_calls)
        print_log(
            f"Number of removed mangled API calls: {self.removed_mangled_api_calls}\n"
        )
        # only possible after removing mangled names, since they are case-sensitive
        df["api"] = df["api"].str.lower()

        df = df[~df["api"].astype(str).str.strip().eq("")]
        df.dropna(inplace=True)
        print_log(f"Rows after removing missing values: {len(df)}\n")

        df["api_count"] = df["api"].apply(lambda x: len(x.split(",")))
        print_log(
            f'Global minimum number of API calls: {df["api_count"].min()}\n'
            f'Global maximum number of API calls: {df["api_count"].max()}\n'
        )

        classes = df["class"].value_counts().index.tolist()
        assert len(classes) == 8
        set_plt_style()
        fig, axes = plt.subplots(4, 2, figsize=page_figsize(1, 0.9), sharey=True)
        for ax, cls in zip(axes.flatten(), classes):
            class_df = df[df["class"] == cls]
            api_counts = class_df["api_count"]
            sns.kdeplot(
                data=class_df,
                x="api_count",
                fill=True,
                linewidth=1.5,
                alpha=0.5,
                log_scale=(True, False),
                ax=ax,
            )
            ax.axvline(
                x=api_counts.quantile(0.05),
                color="red",
                linestyle="--",
                label="5. a 95. percentil",
            )
            ax.axvline(x=api_counts.quantile(0.95), color="red", linestyle="--")
            ax.set_title(f"Distribúcia počtu API – {slovak_trans(cls).capitalize()}")
            ax.set_xlabel("Počet API")
            ax.set_ylabel("Hustota")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.04))
        save_plot(None, "api_count_class_dist_kde")

        print_log(
            f"\nAPI count limits for each class:\n"
            f"{'Class':<20}{'Lower Limit':<20}{'Upper Limit':<20}\n"
        )
        limits = {}
        for cls in classes:
            subset = df[df["class"] == cls]
            lower_limit = int(subset["api_count"].quantile(0.05))
            upper_limit = int(subset["api_count"].quantile(0.95))
            limits[cls] = (lower_limit, upper_limit)
            print_log(f"{cls:<20}{lower_limit:<20}{upper_limit:<20}\n")
        df = df.apply(lambda row: self._is_within_limits(row, limits), axis=1)
        df.dropna(inplace=True)
        print_log(f"Rows after removing class specific outliers: {len(df)}\n")

        unique_apis = set(
            api.strip().lower() for apis in df["api"] for api in apis.split(",")
        )
        print_log(
            f"\nRows after preprocessing: {len(df)}\n"
            f"Number of unique API calls after preprocessing: {len(unique_apis)}\n"
            f"Class distribution after preprocessing:\n{df['class'].value_counts()}\n"
        )

        (DATASETS_DIR_PATH / "preproc_log.txt").write_text(log_text)
        df.to_csv(self.FINAL_DATASET, index=False)
        print(f"Successfully preprocessed datasets. Saved to {self.FINAL_DATASET}")

    @timer
    def run(self) -> None:
        self.preprocess_datasets()


if __name__ == "__main__":
    Preprocessor().run()
