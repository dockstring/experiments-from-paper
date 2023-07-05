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


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", help="path to metrics TSV file", required=True)
    return parser.parse_args(args=args)


def main() -> None:
    args = parse_args()
    print(f"Reading file: {args.metrics}")
    dataset = pd.read_csv(args.metrics, sep="\t", low_memory=False)

    fig_width = 5.50107  # inches, NeurIPS template
    fig_height = 1.5
    num_bins = 25
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(fig_width, fig_height), constrained_layout=True, sharey="row")

    # Logp
    ax = axes[0]
    ax.hist(dataset["logp"], color=colors[0], bins=num_bins, edgecolor="black")
    ax.axvline(5, color=colors[0], linestyle="dashed", zorder=-1)

    ax.set_xticks([-5, 0, 5, 10])
    ax.set_xlabel("logP")

    # Weight
    ax = axes[1]
    ax.hist(dataset["mol_weight"], color=colors[2], bins=num_bins, edgecolor="black")
    ax.axvline(500, color=colors[2], linestyle="dashed", zorder=-1)

    ax.set_xlabel("Weight [g/mol]")

    # Counts
    ax = axes[2]
    ax.hist(dataset["num_hbd"], color=colors[3], bins=range(20), label="HBD", alpha=0.6, edgecolor="black")
    ax.axvline(5, color=colors[3], linestyle="dashed", zorder=-1)

    ax.hist(dataset["num_hba"], color=colors[4], bins=range(20), label="HBA", alpha=0.6, edgecolor="black")
    ax.axvline(10, color=colors[4], linestyle="dashed", zorder=-1)

    ax.hist(dataset["num_rot_bonds"], color=colors[6], bins=range(20), label="RB", alpha=0.6, edgecolor="black")
    ax.axvline(10, color=colors[6], linestyle="dotted", zorder=-1)

    ax.legend()

    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Count")

    # QED
    ax = axes[-1]
    ax.hist(dataset["qed"], color=colors[1], bins=num_bins, edgecolor="black")
    ax.set_xlabel("QED")
    ax.set_xlim(0.0, 1.0)

    ticks = [
        (0, "0"),
        (20_000, "20k"),
        (40_000, "40k"),
        (60_000, "60k"),
        (80_000, "80k"),
        (100_000, "100k"),
    ]

    axes[0].set_yticks([p for p, l in ticks])
    axes[0].set_yticklabels([l for p, l in ticks])

    fig.savefig("dataset.pdf")


if __name__ == "__main__":
    main()
