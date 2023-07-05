""" Script to print/analyze regression results """

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

NAME_REMAP = {
    "gp_exact": "GP (exact)",
    "gp_sparse": "GP (sparse)",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "xgb": "XGBoost",
    "mpnn": "MPNN",
    "gat": "GAT",
    "attentivefp": "Attentive FP",
}

PREFFERED_ORDER = "ridge lasso xgb gp_exact gp_sparse mpnn attentivefp".split()

TOY_TASKS = {"QED", "logP"}


if __name__ == "__main__":
    # Collect arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        default="./official_results/regression",
        help="Result dir to make table from",
    )
    parser.add_argument("--metric", type=str, default="R2", help="Which metric to plot.")
    parser.add_argument(
        "--std",
        action="store_true",
        help="Flag to include std in the table, in addition to the mean.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output latex table instead of regular table.",
    )
    args = parser.parse_args()

    # Read in data
    protein_res_dict = defaultdict(dict)
    results_path = Path(args.results_path)
    assert results_path.exists()
    for method_res_dir in sorted(results_path.iterdir()):
        for protein_res_dir in sorted(method_res_dir.iterdir()):
            res_jsons = []
            for res_file in protein_res_dir.glob("*.json"):
                with open(res_file) as f:
                    res_jsons.append(json.load(f))

            if len(res_jsons) == 0:
                continue

            # Get mean + std
            vals = [r["metrics_test"][args.metric] for r in res_jsons]
            mu = np.average(vals)
            std = np.std(vals)
            protein_res_dict[protein_res_dir.name][method_res_dir.name] = (
                mu,
                std,
                len(vals),
            )

    # Dataframe columns
    all_methods = list(set(method_name for r in protein_res_dict.values() for method_name in r.keys()))
    all_methods_renamed = [NAME_REMAP[name] if name in NAME_REMAP else name for name in all_methods]
    reverse_method_name_map = {rename: orig_name for rename, orig_name in zip(all_methods_renamed, all_methods)}
    all_methods_renamed_sorted = [NAME_REMAP[n] for n in PREFFERED_ORDER if n in all_methods]
    all_methods_renamed_sorted += sorted([n for n in all_methods_renamed if n not in all_methods_renamed_sorted])
    all_methods_renamed = all_methods_renamed_sorted
    del all_methods_renamed_sorted
    if args.std:
        columns = pd.MultiIndex.from_product([all_methods_renamed, "mean std".split()])
    else:
        columns = list(all_methods_renamed)

    # Dataframe rows for proteins
    df = []
    index = []

    def _add_row(protein_name, protein_res):
        index.append(protein_name)
        row = []
        for method_name in all_methods_renamed:
            if args.std:
                n = 2
            else:
                n = 1
            orig_method_name = reverse_method_name_map[method_name]
            if orig_method_name in protein_res:
                row.extend(protein_res[orig_method_name][:n])
            else:
                row.extend([math.nan] * n)
        df.append(row)

    for protein_name, protein_res in protein_res_dict.items():
        if protein_name in TOY_TASKS:
            continue
        _add_row(protein_name, protein_res)

    # Average row
    if not args.std:
        index.append(r"\textbf{Average Rank}")
        ranks = np.argsort(np.argsort(-np.asarray(df))) + 1
        avg_rank = np.average(ranks, axis=0)
        df.append(list(avg_rank))

    # Rows for non-proteins
    for protein_name, protein_res in protein_res_dict.items():
        if protein_name in TOY_TASKS:
            _add_row(protein_name, protein_res)
            index = index[-1:] + index[:-1]
            df = df[-1:] + df[:-1]

    # Make dataframe
    df = pd.DataFrame(df, columns=columns, index=index)
    df.index.set_names("Target", inplace=True)

    # Print final dataframe
    float_format = "{:.3f}".format
    if args.latex:
        # Apply bolding and formatting
        df_latex = df.copy()
        for col in df_latex.columns:
            df_latex[col] = df_latex[col].apply(float_format)
        for row_name, row in df_latex.iterrows():
            if "rank" in row_name.lower():
                mult = -1
            else:
                mult = 1
            max_idx = np.nanargmax(mult * df.loc[row_name].values)
            val_orig = df_latex.loc[row_name][df_latex.columns[max_idx]]
            df_latex.loc[row_name][df_latex.columns[max_idx]] = "\\textbf{{{:s}}}".format(val_orig)
        print(df_latex.to_latex(escape=False))
    else:
        print(df.to_string(float_format=float_format))
