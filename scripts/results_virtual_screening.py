""" Script to output results for molecular optimization. """

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Hard coding, ideally remove and put into arguments
ZINC_SAMPLE_DIR = "./official_results/random_zinc_sample"
DATASET_PATH = "./data/dockstring-dataset.tsv"
RESULT_DIR = "./official_results/virtual-screening"
TARGETS_TESTED = "KIT PARP1 PGR".split()
METHODS_TESTED = "ridge attentivefp".split()
PERC_Q_LIST = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4]
PERC_Q_SINGLE = 1e-3
NAME_REMAP = dict(ridge="Ridge", attentivefp="Attentive FP")
OUTPUT_DIR = "./plots/virtual_screening"

if __name__ == "__main__":
    # Collect arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true", help="Flag to output latex.")
    args = parser.parse_args()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Read main dataset
    df_dataset = pd.read_csv(DATASET_PATH, sep="\t")

    # Put all results into a table
    df_rows = []
    for target in TARGETS_TESTED:
        # Read ZINC
        df_zinc = df_zinc_na = pd.read_csv(
            Path(ZINC_SAMPLE_DIR) / f"random_sample_{target}.tsv",
            sep="\t",
        )
        df_zinc = df_zinc.dropna()
        print(f"{target} ZINC Frac NaN: {1 - len(df_zinc) / len(df_zinc_na)}")

        for cutoff_idx, cutoff_quantile in enumerate([1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4]):
            cutoff_value = np.quantile(df_zinc.score.values, cutoff_quantile)
            num_pass_zinc = np.average(df_zinc.score.values <= cutoff_value)
            curr_row = dict(
                Target=target,
                Percentile=cutoff_quantile * 100,
            )
            curr_row["Cutoff Score"] = cutoff_value
            curr_row["Max EF"] = int(1 / cutoff_quantile)

            for method in "ridge attentivefp".split():
                # Read method results
                df_top_pred = df_top_pred_na = pd.read_csv(
                    Path(RESULT_DIR) / f"{method}/{target}/predictions-trial-0-top-scored.tsv",
                    sep="\t",
                )
                df_top_pred = df_top_pred.dropna()

                if cutoff_idx == 0:
                    print(f"\t{method} Frac NaN: {1 - len(df_top_pred) / len(df_top_pred_na)}")
                    print(
                        f"\t{method}: Number better than best: "
                        f"{np.sum(df_top_pred[target].values < df_dataset[target].min()):d}"
                    )

                num_pass_method = np.average(df_top_pred[target].values <= cutoff_value)
                ef = num_pass_method / num_pass_zinc
                curr_row["EF " + NAME_REMAP[method]] = round(ef, 1)
            df_rows.append(curr_row)

    # Output big table with lots of different cutoffs
    ef_df_all = pd.DataFrame(data=df_rows)
    ef_df_big = ef_df_all.set_index(["Target", "Percentile", "Max EF"])
    float_format = "{:.3f}".format
    if args.latex:
        table_str = ef_df_big.to_latex(escape=False, float_format=float_format)
    else:
        table_str = ef_df_big.to_string(
            float_format=float_format,
        )
    with open(output_dir / "big-ef-table.txt", "w") as f:
        f.write(table_str)

    # Small table with just 1 cutoff
    ef_df_small = ef_df_all.copy()
    ef_df_small = ef_df_small[np.abs(ef_df_small["Percentile"].values / 100 - PERC_Q_SINGLE) < 1e-4]
    for col_name in ["Max EF", "Percentile", "Cutoff Score"]:
        del ef_df_small[col_name]
    if args.latex:
        table_str = ef_df_small.to_latex(escape=False, float_format=float_format)
    else:
        table_str = ef_df_small.to_string(
            float_format=float_format,
        )
    with open(output_dir / "small-ef-table.txt", "w") as f:
        f.write(table_str)
