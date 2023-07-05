""" Analyze the results of clustering (make plots) """
import argparse

import clustering
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import pdist, squareform

# Styling
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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=str, help="TSV file containing clusters", required=True)
    parser.add_argument("--scaffolds", type=str, help="TSV file containing scaffold information", required=True)
    parser.add_argument("--max_num_clusters", type=int, default=25, help="Number of clusters to plot.")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Get clusters
    df = pd.read_csv(args.clusters, sep="\t")
    counts = df["cluster"].value_counts()

    # Print basic info
    print(f"Number of clusters (size>1): {len(counts[counts > 1])}")
    print(f"Number of isolated points: {sum(counts == 1)}")

    # Print cluster sizes
    print(f"Largest {args.max_num_clusters} clusters:")
    print(counts[: args.max_num_clusters])

    # Plot
    fig_width = 5.50107  # inches, NeurIPS template
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(fig_width, 1.5), constrained_layout=True)

    # 1st plot
    ax = axes[0]
    log_bins = np.geomspace(counts.min(), counts.max(), 16)
    ax.hist(counts, bins=log_bins, color=colors[2], alpha=0.75, edgecolor="black")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks([1, 10, 100, 1_000, 10_000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())

    ax.yaxis.set_tick_params(which="minor", left=False, bottom=False)
    ax.set_yticks([1, 10, 100, 1_000, 10_000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())

    ax.set_xlabel("DBSCAN Cluster Size")
    ax.set_ylabel("Count")

    # 2nd plot
    ax = axes[1]

    # Get fingerprints + distances
    cluster_id1, cluster_id2 = counts.index[1], counts.index[2]
    smiles_1 = df.loc[df["cluster"] == cluster_id1, "smiles"].tolist()
    smiles_2 = df.loc[df["cluster"] == cluster_id2, "smiles"].tolist()

    fps = [clustering.standard_fingerprint(Chem.MolFromSmiles(s)) for s in (smiles_1 + smiles_2)]
    fp_array = clustering.fingerprints_to_array(fps)
    dist_mat = squareform(pdist(fp_array, metric="jaccard"))

    # Collect distances
    size_1 = len(smiles_1)
    intra_dists = np.concatenate([dist_mat[size_1:, size_1:].flatten(), dist_mat[:size_1, :size_1].flatten()])
    inter_dists = dist_mat[size_1:, :size_1].flatten()

    # Plot histograms
    hist_kwargs = dict(density=True, bins=np.linspace(start=0, stop=1.0, num=25), alpha=0.75, edgecolor="black")
    ax.hist(intra_dists, label="intra", color=colors[0], **hist_kwargs)
    ax.hist(inter_dists, label="inter", color=colors[1], **hist_kwargs)

    ax.legend()

    ax.set_ylabel("Normalized Count")
    ax.set_xlabel("Jaccard Distance")

    # 3rd plot
    ax = axes[2]
    scaffold_sizes = pd.read_csv(args.scaffolds, sep="\t", header=None, names=["sizes"])["sizes"]

    merged = scaffold_sizes + counts
    log_bins = np.geomspace(merged.min(), merged.max(), 16)

    ax.hist(scaffold_sizes, bins=log_bins, color=colors[3], alpha=0.75, edgecolor="black", label="scaffold")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks([1, 10, 100, 1_000, 10_000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())

    ax.yaxis.set_tick_params(which="minor", left=False, bottom=False)
    ax.set_yticks([1, 10, 100, 1_000, 10_000, 100_000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())

    ax.set_xlabel("Scaffold Cluster Size")
    ax.set_ylabel("Count")

    # Save figures
    fig.savefig("clusters.pdf")


if __name__ == "__main__":
    main()
