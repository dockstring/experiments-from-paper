import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from clustering import read_smiles
from rdkit import DataStructs
from rdkit.Chem import AllChem, RDKFingerprint
from rdkit.DataStructs import ExplicitBitVect

# Styling
fig_width = 2.1
fig_height = 2.1
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


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="path to CSV file", required=True)
    parser.add_argument("--max_size", type=int, help="maximum number of entries in dataset", required=False)
    parser.add_argument("--seed", type=int, default=0, help="Random seed (only used for data subsets)")
    return parser.parse_args(args=args)


def mol_from_smiles(smiles: str) -> AllChem.Mol:
    mol = AllChem.MolFromSmiles(smiles)
    if not mol:
        raise RuntimeError(f"Cannot read SMILES: {smiles}")

    return mol


def get_sorted_distances(fps: List[ExplicitBitVect]) -> List[List[float]]:
    sorted_distances = []
    for fp_i in fps:
        distances = []
        for fp_j in fps:
            similarity = DataStructs.FingerprintSimilarity(fp_i, fp_j, metric=DataStructs.TanimotoSimilarity)
            distances.append(1 - similarity)

        distances.sort()
        sorted_distances.append(distances)

    return sorted_distances


def standard_fingerprint(mol):
    return RDKFingerprint(mol, maxPath=6)


def main():
    args = parse_args()
    smiles_list = read_smiles(args.dataset)

    if args.max_size is not None and args.max_size < len(smiles_list):
        print(f"Choosing random subset of size {args.max_size}")
        generator = np.random.default_rng(args.seed)
        random_indices = generator.choice(len(smiles_list), size=args.max_size, replace=False)
        smiles_list = [smiles_list[index] for index in random_indices]

    fingerprints = [standard_fingerprint(mol_from_smiles(smiles)) for smiles in smiles_list]
    sorted_distances = get_sorted_distances(fingerprints)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), constrained_layout=True)
    for i, k in enumerate([1, 2, 3]):
        ax.hist(
            [distances[k] for distances in sorted_distances],
            bins=10,
            color=colors[i],
            label=f"k={k}",
            histtype="stepfilled",
            alpha=0.75,
            density=True,
            edgecolor="black",
        )

    ax.set_xlabel("Tanimoto Distance")
    ax.set_ylabel("Density")
    ax.legend()

    fig.savefig("distances.pdf")


if __name__ == "__main__":
    main()
