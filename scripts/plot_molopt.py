""" Script to output results for molecular optimization. """

import argparse
import functools
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm.auto import tqdm

plt.rcParams.update({"font.size": 6})

colors = [
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]

METHOD_NAME_REMAP = {
    "graph_ga": "Graph GA",
    "bo_gp_exact": "GP-BO",
}

MAXIMIZATION_OBJECTIVES = {"logP", "QED"}

OBJECTIVE_DEPENDENCIES = {
    "F2_qed-pen-v3": ["F2"],
    "PPAR-all_qed-pen-v3": ["PPARA", "PPARD", "PPARG"],
    "JAK2-not-LCK-v2_qed-pen-v3": ["JAK2", "LCK"],
}


def top1_so_far(vals):
    out = vals[:1]
    for v in vals[1:]:
        out.append(min(out[-1], v))
    return out


def topn_so_far(vals, n):
    assert n > 0
    top_list = sorted(vals[:n])
    out = [math.nan] * (n - 1)
    out.append(top_list[-1])
    for v in vals[n:]:
        top_list.append(v)
        top_list.sort()
        top_list = top_list[:n]
        out.append(top_list[-1])
    return out


def _get_min_median_max(method_res_list, plot_metric=top1_so_far, is_min=True):
    if len(method_res_list) == 0:
        return

    # Get all score lists; make negative, make sure they are the same length
    objective_list = [[-x for x in r["scores"]] for r in method_res_list]
    if not all([len(o) == len(objective_list[0]) for o in objective_list]):
        max_len = max([len(o) for o in objective_list])
        objective_list = [l + [math.nan] * (max_len - len(l)) for l in objective_list]

    # Convert to metric over time
    plot_list = [plot_metric(l) for l in objective_list]
    plot_list = np.array(plot_list)
    if not is_min:
        plot_list = -plot_list
    return (
        np.min(plot_list, axis=0),
        np.median(plot_list, axis=0),
        np.max(plot_list, axis=0),
    )


def batch_tanimoto_numpy(fp_arr1, fp_arr2):
    fp_int = fp_arr1 @ fp_arr2.T
    fp_union = np.sum(fp_arr1, axis=1, keepdims=True) + np.sum(fp_arr2, axis=1, keepdims=True).T - fp_int
    return fp_int / fp_union


def _get_numpy_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(Chem.RDKFingerprint(mol, maxPath=6))


def parse_args():
    # Collect arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        default="./official_results/molopt",
        help="Result directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plots/molopt",
        help="Where to output plots.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/dockstring-dataset-extra-props.tsv",
        help="Path to (augmented) dataset.",
    )
    parser.add_argument("--n_top_mols", type=int, default=12, help="Number of top molecules to plot.")
    parser.add_argument(
        "--plot_top_n",
        type=int,
        nargs="+",
        default=[1, 25],
        help="Which N values to plot for top N.",
    )
    parser.add_argument("--scaffolds", action="store_true")
    parser.add_argument("--fingerprints", action="store_true")
    parser.add_argument("--sub_img_size", type=int, default=400, help="RDKit sub image size")
    parser.add_argument("--latex", action="store_true", help="Flag to output latex.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Read in dataset and calculate objective function values
    df = pd.read_csv(args.dataset_path, sep="\t", header=0)

    df["F2_qed-pen-v3"] = df["F2"] + 10 * (1 - df["QED"])

    df["PPAR-all"] = df["PPARA PPARD PPARG".split()].max(axis=1)
    df["PPAR-all_qed-pen-v3"] = df["PPAR-all"] + 10 * (1 - df["QED"])

    df["JAK2-not-LCK-v2"] = df["JAK2"] - np.minimum(df["LCK"] - (-8.1), 0)
    df["JAK2-not-LCK-v2_qed-pen-v3"] = df["JAK2-not-LCK-v2"] + 10 * (1 - df["QED"])

    # Scaffolds
    if args.scaffolds:
        df["gen-scaffold"] = [
            Chem.MolToSmiles(
                MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)))
            )
            for s in tqdm(list(df["smiles"]), desc="Calculating train set scaffolds")
        ]
        set(df["gen-scaffold"])
    if args.fingerprints:
        np.stack([_get_numpy_fp(s) for s in tqdm(df["smiles"].values, desc="Calculating fingerprints.")])

    # Read in BO results
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
            protein_res_dict[protein_res_dir.name][method_res_dir.name] = res_jsons

    # Results for all targets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    PLOT_OBJECTIVES = OrderedDict(
        [
            ("F2_qed-pen-v3", "F2"),
            ("PPAR-all_qed-pen-v3", "Promiscuous PPAR"),
            ("JAK2-not-LCK-v2_qed-pen-v3", "Selective JAK2"),
            ("logP", "logP"),
            ("QED", "QED"),
        ]
    )

    fig_width = 5.50107  # inches, NeurIPS template
    fig_height = 2.4

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(fig_width, fig_height), constrained_layout=True)
    axes = [ax for row in axes for ax in row]

    # Top n plots
    n = 25

    for ax, obj in zip(axes, PLOT_OBJECTIVES):
        if obj in MAXIMIZATION_OBJECTIVES:
            dataset_best = df[obj].max()
        else:
            dataset_best = df[obj].min()

        ax.axhline(dataset_best, color="k", linestyle="--", label="Dataset best")
        for method_name, method_res_list in protein_res_dict[obj].items():
            assert len(method_res_list) == 3  # should always be 3 replicates
            mn, md, mx = _get_min_median_max(
                method_res_list,
                functools.partial(topn_so_far, n=n),
                is_min=obj not in MAXIMIZATION_OBJECTIVES,
            )

            if obj == "QED":
                mn /= 10.0
                md /= 10.0
                mx /= 10.0

            ax.plot(md, label=METHOD_NAME_REMAP[method_name])
            ax.fill_between(range(len(md)), mn, mx, alpha=0.3)

        if obj in ["F2_qed-pen-v3", "PPAR-all_qed-pen-v3"]:
            ax.set_ylim(-12, -7.5)
        elif obj == "JAK2-not-LCK-v2_qed-pen-v3":
            ax.set_ylim(-10, -7.5)
        elif obj == "QED":
            ax.set_ylim(0.5, 1.0)

        ax.set_title(PLOT_OBJECTIVES[obj])

    axes[-2].legend()
    axes[-1].remove()

    axes[0].set_ylabel("Objective")
    axes[3].set_ylabel("Objective")

    # Figure saving
    fig.savefig(output_dir / f"molopt_top{n}.pdf")


if __name__ == "__main__":
    main()
