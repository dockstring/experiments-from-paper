import argparse

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

# yapf: disable
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
# yapf: enable


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="path to dataset file", required=True)
    parser.add_argument("--overlap", help="path to overlap file", required=True)
    return parser.parse_args(args=args)


def main() -> None:
    args = parse_args()

    ref = "LCK"
    # others = targets.copy()
    # others.remove(ref)
    others = ["HSD11B1", "PDE5A", "PTGS2", "PTPN1", "PARP1"]

    dataset = pd.read_csv(args.dataset, sep="\t")
    overlap = pd.read_csv(args.overlap, sep="\t")

    dataset = dataset.set_index("inchikey")
    overlap = overlap.set_index("inchikey")

    # Slice
    overlap = overlap.loc[dataset.index]
    actives = overlap.loc[overlap[ref + "_label"] == "A"]

    selection = dataset.loc[actives.index]

    fig_width = 5.50107 / 3  # inches

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_width), constrained_layout=True)

    ax.hist(
        selection[ref],
        bins=25,
        color=colors[0],
        label=ref,
        histtype="stepfilled",
        alpha=0.75,
        density=True,
        edgecolor="black",
    )

    ax.hist(
        selection[others].values.flatten(),
        bins=25,
        color=colors[1],
        label="others",
        histtype="stepfilled",
        alpha=0.75,
        density=True,
        edgecolor="black",
    )

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()

    fig.savefig("hists.pdf")


if __name__ == "__main__":
    main()
