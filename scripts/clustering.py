import argparse
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, RDKFingerprint
from rdkit.DataManip.Metric import GetTanimotoDistMat
from rdkit.DataStructs import ExplicitBitVect
from scipy.spatial.distance import jaccard
from sklearn.cluster import DBSCAN


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="path to CSV file", required=True)
    parser.add_argument("--max_size", type=int, help="maximum number of entries in dataset", required=False)
    parser.add_argument("--epsilon", type=float, help="epsilon for clustering algorithm", required=False, default=0.3)
    parser.add_argument("--num_jobs", type=int, help="number of cores to use (-1 = all)", required=False, default=-1)
    parser.add_argument("--alg", type=str, help="clustering algorithm to use", required=False, default="brute")
    parser.add_argument(
        "--save_path", type=str, help="path to save cluster files", required=False, default="clusters.txt"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (only used for data subsets)")
    return parser.parse_args(args=args)


def read_smiles(path: os.PathLike) -> List[str]:
    return pd.read_csv(path, sep="\t", usecols=["smiles"])["smiles"].to_list()


def mol_from_smiles(smiles: str) -> AllChem.Mol:
    mol = AllChem.MolFromSmiles(smiles)
    if not mol:
        raise RuntimeError(f"Cannot read SMILES: {smiles}")

    return mol


def fingerprints_to_array(fps: List[ExplicitBitVect]) -> np.ndarray:
    arrays = [np.array(list(map(int, fp.ToBitString())), dtype=bool) for fp in fps]  # hack
    return np.stack(arrays, axis=0)  # [batch, fp_size]


def standard_fingerprint(mol):
    return RDKFingerprint(mol, maxPath=6)


def form_clusters(smiles_list: List[str], labels: List[int]) -> List[List[str]]:
    # Clustered molecules
    # yapf: disable
    clusters = [
        [node for label, node in zip(labels, smiles_list) if label == cluster_idx]
        for cluster_idx in range(0, max(labels) + 1)
    ]
    # yapf: enable

    # Isolated molecules
    for label, node in zip(labels, smiles_list):
        if label == -1:
            clusters.append([node])

    assert len(smiles_list) == sum(len(cluster) for cluster in clusters)
    return clusters


def write_clusters_to_file(clusters: List[List[str]], path: str) -> None:
    print(f"Writing clusters to '{path}'")
    with open(path, mode="w") as f:
        for cluster in clusters:
            f.write(",".join(smiles for smiles in cluster))
            f.write("\n")


def main():
    args = parse_args()
    smiles_list = read_smiles(args.dataset)

    if args.max_size is not None and args.max_size < len(smiles_list):
        print(f"Choosing random subset of size {args.max_size}")
        generator = np.random.default_rng(args.seed)
        random_indices = generator.choice(len(smiles_list), size=args.max_size, replace=False)
        smiles_list = [smiles_list[index] for index in random_indices]

    print("Calculating fingerprints...")
    fingerprints = [standard_fingerprint(mol_from_smiles(smiles)) for smiles in smiles_list]

    x = fingerprints_to_array(fingerprints)
    print(f"Number of molecules: {x.shape[0]}, fingerprint size: {x.shape[1]}")

    # Check
    assert np.allclose(
        GetTanimotoDistMat(fingerprints[:3]),
        np.array([jaccard(x[0], x[1]), jaccard(x[0], x[2]), jaccard(x[1], x[2])]),
    )

    print("DBSCAN epsilon:", args.epsilon)
    clustering = DBSCAN(
        min_samples=2,  # default: 5, but we want to group isomers so 2 is safer
        eps=args.epsilon,  # max distance between two samples for neighborhood
        metric="jaccard",  # Tanimoto distance
        n_jobs=args.num_jobs,
        algorithm=args.alg,
    )

    print("Clustering...")
    start_time = datetime.now()
    clustering.fit(x)
    fit_duration = datetime.now() - start_time

    print(f"Fit done in {fit_duration}")
    print("Number of clusters (size>1):", max(clustering.labels_) + 1)
    print("Number of isolated nodes:", len([label for label in clustering.labels_ if label == -1]))

    clusters = form_clusters(smiles_list, labels=clustering.labels_)

    write_clusters_to_file(clusters, path=args.save_path)


if __name__ == "__main__":
    main()
