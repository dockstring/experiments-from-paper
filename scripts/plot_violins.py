import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
"""
targets = [
    # Kinases
    'JAK2', 'MAPK14', 'LCK', 'IGF1R', 'MAPKAPK2', 'MET', 'PTK2', 'SRC',
    # Proteases
    'F2', 'F10', 'ADAM17',
    # Nuclear receptors
    'PPARG', 'PPARD', 'PPARA', 'ESR1', 'ESR2', 'NR3C1',
    # Enzymes and others
    'HSD11B1', 'PDE5A', 'PTGS2', 'PTPN1', 'PARP1'
]
"""

targets = sorted(
    [
        "ADAM17",
        "ESR1",
        "ESR2",
        "F10",
        "F2",
        "HSD11B1",
        "IGF1R",
        "JAK2",
        "KIT",
        "LCK",
        "MAPK14",
        "MAPKAPK2",
        "MET",
        "NR3C1",
        "PARP1",
        "PDE5A",
        "PGR",
        "PPARA",
        "PPARD",
        "PPARG",
        "PTGS2",
        "PTK2",
        "PTPN1",
        "SRC",
    ]
)


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="path to TSV file", required=True)
    return parser.parse_args(args=args)


def prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.dropna(axis=0).copy()  # discard rows without score
    for target in targets:
        dataset.loc[dataset[target] > 0.0, target] = 0.0
    return dataset
    # return dataset[dataset['score'] < 0.0]


def main() -> None:
    args = parse_args()
    df = prepare_dataset(pd.read_csv(args.dataset, sep="\t"))

    fig_width = 5.50107  # inches, NeurIPS template
    fig_height = 1.7

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), constrained_layout=True)

    # parts = ax.violinplot(data, showextrema=False, showmedians=True, quantiles=[[0.25, 0.75]] * len(targets))
    parts = ax.violinplot(df[targets], showextrema=False, showmedians=True)

    for body in parts["bodies"]:
        body.set_facecolor(colors[0])
        body.set_edgecolor("black")

    parts["cmedians"].set_edgecolor(colors[0])

    ax.set_xticks(np.arange(1, len(targets) + 1))
    ax.set_xticklabels(targets, rotation=90)
    ax.set_ylabel("Score")

    fig.savefig("violin.pdf")


if __name__ == "__main__":
    main()
