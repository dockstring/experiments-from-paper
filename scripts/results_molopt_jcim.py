""" Script to output results for molecular optimization. """

import argparse
import functools
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, AllChem, Crippen, Descriptors, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm.auto import tqdm

METHOD_NAME_REMAP = {
    "random_zinc": "Rand. ZINC",
    "selfies_ga": "SELFIES GA",
    "graph_ga": "Graph GA",
    "bo_gp_exact": "GP-BO (UCB)",
    "bo_gp_exact_ei": "GP-BO (EI)",
}

# Figure config
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
METHOD_TO_COLOR = {mname: c for mname, c in zip(METHOD_NAME_REMAP, colors)}
plt.rcParams.update(
    {
        "font.size": 5,
        "ytick.major.size": 3,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3,
        "xtick.major.width": 1.0,
    }
)

MAXIMIZATION_OBJECTIVES = {"logP", "QED"}
PLOT_SKIP = 10

PLOT_OBJECTIVES = {
    "F2-no-pen": {
        "F2": "F2",
    },
    "dockstring-obj-pen": {
        "F2_qed-pen-v3": "F2",
        "PPAR-all_qed-pen-v3": "Promiscious PPAR",
        "JAK2-not-LCK-v2_qed-pen-v3": "Selective JAK2",
    },
    "toy-tasks": {
        "logP": "logP",
        "QED": "QED",
    },
}

BEST_MOL_OBJECTIVES = {
    "dockstring-obj": {
        "F2": "F2 (no QED)",
        "F2_qed-pen-v3": "F2",
        "PPAR-all_qed-pen-v3": "Promiscious PPAR",
        "JAK2-not-LCK-v2_qed-pen-v3": "Selective JAK2",
    },
}
N_BEST_MOL = 4

SIM_MOL_OBJECTIVES = {
    "dockstring-obj": {
        "F2_qed-pen-v3": "F2",
        "PPAR-all_qed-pen-v3": "Promiscious PPAR",
        "JAK2-not-LCK-v2_qed-pen-v3": "Selective JAK2",
    },
}
N_BEST_SIM_MOL = 1
N_SIM_MOL = 3
OBJECTIVE_DEPENDENCIES = {
    "F2_qed-pen-v3": ["F2"],
    "PPAR-all_qed-pen-v3": ["PPARA", "PPARD", "PPARG"],
    "JAK2-not-LCK-v2_qed-pen-v3": ["JAK2", "LCK"],
}

NO_TRAINSIM = {"logP", "QED", "F2"}


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


if __name__ == "__main__":
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
        default="./plots/molopt-jcim",
        help="Where to output plots.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/dockstring-excape/dockstring-dataset-extra-props.tsv",
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
    parser.add_argument(
        "--no_latex",
        dest="latex",
        action="store_false",
        help="Flag to NOT output latex.",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Flag to output SVG images.",
    )

    args = parser.parse_args()

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
        train_set_scaffolds = set(df["gen-scaffold"])
    if args.fingerprints:
        train_set_fingerprints = np.stack(
            [_get_numpy_fp(s) for s in tqdm(df["smiles"].values, desc="Calculating fingerprints.")]
        )

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
    best_mols_dict = dict()
    best_mols_info = dict()
    for obj_dict_name, obj_dict in PLOT_OBJECTIVES.items():
        _fig_height = 3.6 if len(obj_dict) > 1 else 3.0
        fig, axes_grid = plt.subplots(
            nrows=len(args.plot_top_n),
            ncols=len(obj_dict),
            figsize=(7.0 / 3 * len(obj_dict), _fig_height),
            sharex=True,
            squeeze=False,
        )
        del _fig_height

        for n_idx, n in enumerate(args.plot_top_n):
            assert len(obj_dict) <= 3
            legend_dict = dict()
            for obj_idx, obj in enumerate(obj_dict):
                plt.sca(axes_grid[n_idx, obj_idx])

                # Best / nth best in dataset
                if obj in MAXIMIZATION_OBJECTIVES:
                    dataset_best = df[obj].max()
                else:
                    dataset_best = df[obj].min()
                dataset_nth_best = df[obj].sort_values(ascending=obj not in MAXIMIZATION_OBJECTIVES).values[n - 1]

                # Top n plots
                plt.axhline(dataset_best, color="k", linestyle="--", label="Dataset best")
                if n > 1:
                    plt.axhline(
                        dataset_nth_best,
                        color="k",
                        linestyle=":",
                        label=f"Dataset {n}th best",
                    )
                for method_name in METHOD_NAME_REMAP.keys():
                    method_res_list = protein_res_dict[obj].get(method_name)
                    if method_res_list is None:
                        continue

                    assert len(method_res_list) == 3  # should always be 3 replicates
                    mn, md, mx = _get_min_median_max(
                        method_res_list,
                        functools.partial(topn_so_far, n=n),
                        is_min=obj not in MAXIMIZATION_OBJECTIVES,
                    )
                    # Fix QED (undo hack I used for numerical stability)
                    if obj == "QED":
                        mn /= 10.0
                        md /= 10.0
                        mx /= 10.0

                    # Actual plotting
                    plt.plot(
                        list(range(len(md)))[::PLOT_SKIP],
                        md[::PLOT_SKIP],
                        label=METHOD_NAME_REMAP[method_name],
                        color=METHOD_TO_COLOR[method_name],
                    )
                    plt.fill_between(
                        range(len(md))[::PLOT_SKIP],
                        mn[::PLOT_SKIP],
                        mx[::PLOT_SKIP],
                        alpha=0.3,
                        color=METHOD_TO_COLOR[method_name],
                    )
                plt.title(f"{obj_dict[obj]} Top {n}")

                if n_idx == len(args.plot_top_n) - 1:
                    plt.xlabel("Number of function evaluations")

                if obj_idx == 0:
                    plt.ylabel("Objective")

                # y limits
                if dataset_best < -5:
                    plt.ylim(None, -5.0)
                elif obj == "QED":
                    plt.ylim(0.92, 0.96)

                # Legend update
                for _hdl, _lbl in zip(*plt.gca().get_legend_handles_labels()):
                    legend_dict[_lbl] = _hdl
                    del _hdl, _lbl

                # mol pics; just produce once per objective
                if n_idx == 0:
                    # Overall best molecules
                    all_smiles_scores = []
                    for method_name, method_res_list in protein_res_dict[obj].items():
                        for res in method_res_list:
                            all_smiles_scores += list(zip(res["scores"], res["new_smiles"], res["raw_scores"]))
                    best_smiles_scores = heapq.nlargest(12, all_smiles_scores)
                    best_mols = [Chem.MolFromSmiles(t[1]) for t in best_smiles_scores]

                    # Table with best molecules
                    obj_mult = 1 if obj in MAXIMIZATION_OBJECTIVES else -1
                    if obj == "QED":
                        obj_mult /= 10  # undo a random hack that I did to make BO more numerically stable
                    best_obj_df = pd.DataFrame(
                        data=np.array([obj_mult * t[0] for t in best_smiles_scores]),
                        columns=["objective"],
                    )
                    best_obj_df["Rank"] = list(range(1, 1 + len(best_smiles_scores)))
                    best_obj_df.set_index("Rank", inplace=True)
                    if obj in best_smiles_scores[0][-1]:
                        best_obj_df[obj] = [t[-1][obj] for t in best_smiles_scores]
                    else:
                        for col in OBJECTIVE_DEPENDENCIES[obj]:
                            best_obj_df[col] = [t[-1][col] for t in best_smiles_scores]
                    best_obj_df["QED"] = [QED.qed(m) for m in best_mols]

                    def _write_table(best_obj_df, file_path):
                        float_format = "{:.2f}".format
                        if args.latex:
                            table_str = best_obj_df.to_latex(escape=False, float_format=float_format)
                        else:
                            table_str = best_obj_df.to_string(
                                float_format=float_format,
                            )
                        with open(file_path, "w") as f:
                            f.write(table_str)

                    # Write intermediate table
                    best_mols_dict[obj] = best_mols
                    best_mols_info[obj] = [dict(row) for _, row in best_obj_df.iterrows()]
                    _write_table(best_obj_df, output_dir / f"{obj}-top-table.txt")

                    # Add extra descriptions
                    best_obj_df["Molecular Weight"] = [Descriptors.MolWt(m) for m in best_mols]
                    best_obj_df["logP"] = [Crippen.MolLogP(m) for m in best_mols]
                    best_obj_df["HBA"] = [AllChem.CalcNumHBA(m) for m in best_mols]
                    best_obj_df["HBD"] = [AllChem.CalcNumHBD(m) for m in best_mols]
                    _write_table(best_obj_df, output_dir / f"{obj}-top-table-FULL.txt")

                    # Draw
                    # Previous code for SVG
                    # with open(output_dir / f"{obj}-bestmols.svg", "w") as f:
                    #     f.write(img)

        # Figure saving
        _all_labels = list(legend_dict.keys())
        if axes_grid.shape[1] == 1:
            plt.subplots_adjust(bottom=0.2, wspace=0.4, left=0.2, right=0.95, hspace=0.2, top=0.95)
            bbox_legend = (0.5, 0.1)
        else:
            plt.subplots_adjust(bottom=0.13, left=0.07, right=0.98, hspace=0.15, top=0.95)
            bbox_legend = (0.5, 0.06)
        fig.legend(
            [legend_dict[l] for l in _all_labels],
            _all_labels,
            ncol={1: 2, 2: 5, 3: 7}[len(obj_dict)],
            loc="upper center",
            bbox_to_anchor=bbox_legend,
        )
        del _all_labels
        # plt.tight_layout()
        plt.savefig(output_dir / f"top{n}_{obj_dict_name}.pdf")
        plt.close()

    # Make images of best molecules
    for obj_dict_name, obj_dict in BEST_MOL_OBJECTIVES.items():
        all_best_mols = []
        all_best_mol_labels = []

        for obj_idx, (obj, obj_name) in enumerate(obj_dict.items()):
            all_best_mols.extend(best_mols_dict[obj][:N_BEST_MOL])
            for rank, d in enumerate(best_mols_info[obj][:N_BEST_MOL]):
                d = dict(d)
                label = f"Obj={d.pop('objective'):.1f}\n"
                label += ", ".join(f"{k}={v:.1f}" for k, v in d.items())
                all_best_mol_labels.append(label)
                del label

        img = Draw.MolsToGridImage(
            all_best_mols,
            legends=all_best_mol_labels,
            useSVG=args.svg,
            subImgSize=(300, 200),
            molsPerRow=N_BEST_MOL,
        )
        if args.svg:
            with open(output_dir / f"{obj_dict_name}-bestmols.svg", "w") as f_out:
                f_out.write(img)
        else:
            img.save(output_dir / f"{obj_dict_name}-bestmols.png")

        # Save the SMILES
        with open(output_dir / f"{obj_dict_name}-bestmols.csv", "w") as f_out:
            f_out.write("\n".join([Chem.MolToSmiles(m) for m in all_best_mols]))

    # Images of similar train set molecules
    if args.fingerprints:
        sim_mols_dict = dict()
        for obj_dict_name, obj_dict in SIM_MOL_OBJECTIVES.items():
            for obj_idx, (obj, obj_name) in enumerate(obj_dict.items()):
                # Fingerprints
                if args.fingerprints and obj not in NO_TRAINSIM:
                    N_SIM = N_SIM_MOL
                    best_mols = best_mols_dict[obj]
                    for i, mol in enumerate(best_mols):
                        mol_list = []
                        mol_legends = []
                        fp = _get_numpy_fp(Chem.MolToSmiles(mol)).reshape(1, -1)
                        fp_sims = batch_tanimoto_numpy(train_set_fingerprints, fp).flatten()
                        argsort = np.argsort(-fp_sims)
                        for sim_rank, j in enumerate(argsort[:N_SIM]):
                            mol_list.append(Chem.MolFromSmiles(df["smiles"].values[j]))
                            mol_legends.append(f"Obj={df[obj].values[j]:.1f}, Sim={fp_sims[j]:.2f}")
                        sim_mols_dict[(obj, i)] = mol_list, mol_legends

            # all mols for image:
            all_draw_mols = []
            all_draw_labels = []
            for obj_idx, (obj, obj_name) in enumerate(obj_dict.items()):
                for i in range(N_BEST_SIM_MOL):
                    all_draw_mols.append(best_mols_dict[obj][i])
                    all_draw_labels.append(f"Obj={best_mols_info[obj][i]['objective']:.1f}")
            for sim_rank in range(N_SIM_MOL):
                for obj_idx, (obj, obj_name) in enumerate(obj_dict.items()):
                    for mol_rank in range(N_BEST_SIM_MOL):
                        all_draw_mols.append(sim_mols_dict[(obj, mol_rank)][0][sim_rank])
                        all_draw_labels.append(sim_mols_dict[(obj, mol_rank)][1][sim_rank])

            img = Draw.MolsToGridImage(
                all_draw_mols,
                legends=all_draw_labels,
                useSVG=args.svg,
                subImgSize=(300, 200),
                molsPerRow=N_BEST_SIM_MOL * len(obj_dict),
            )
            if args.svg:
                with open(output_dir / f"{obj_dict_name}-train-sim-mols.svg", "w") as f_out:
                    f_out.write(img)
            else:
                img.save(output_dir / f"{obj_dict_name}-train-sim-mols.png")

            # Save the SMILES
            with open(output_dir / f"{obj_dict_name}-train-sim-mols.csv", "w") as f_out:
                f_out.write("\n".join([Chem.MolToSmiles(m) for m in all_draw_mols]))
