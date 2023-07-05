import argparse
import itertools
import sys

import matplotlib.colors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

plt.rcParams.update({"font.size": 6})

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
    parser.add_argument("--dataset", help="path to input TSV file", required=True)
    return parser.parse_args(args=args)


def convert_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(labels=["inchikey"], axis="columns")
    df = df.dropna(axis="index")  # drop rows with missing values

    columns = df.columns.to_list()
    columns.remove("smiles")

    positive_score_row = (df[columns] > 0).any(axis=1)
    print(f"Warning: dropping {positive_score_row.sum()} rows with positive scores", file=sys.stderr)
    df = df.loc[~positive_score_row]

    # print('Warning: setting positive scores to 0.0')
    # df[df > 0.0] = 0.0

    df = df.set_index("smiles")
    return df[targets].copy()


def main() -> None:
    args = parse_args()
    print(f"Reading file: {args.dataset}")
    df = pd.read_csv(args.dataset, sep="\t")

    print("Preparing dataset")
    df = convert_dataset(df)

    pearson_rs = {frozenset({a, b}): stats.pearsonr(df[a], df[b])[0] for a, b in itertools.combinations(df.columns, 2)}

    with pd.option_context("display.max_rows", None):
        print(pd.Series(pearson_rs, name="pearson_r").sort_values())

    pairs = [("SRC", "LCK"), ("ESR2", "LCK"), ("PTGS2", "LCK")]
    offset = 0.75

    fig_width = 5.50107  # inches, NeurIPS template
    fig, axes = plt.subplots(nrows=1, ncols=len(pairs), figsize=(fig_width, 1.7), constrained_layout=True)

    hb_ax_tuples = []
    for ax, (target_a, target_b) in zip(axes, pairs):
        selection = df[[target_a, target_b]].to_numpy()
        lower, upper = np.min(selection), np.max(selection)

        hb = ax.hexbin(
            df[target_a],
            df[target_b],
            gridsize=25,
            extent=(lower - 2 * offset, upper + 2 * offset, lower - 2 * offset, upper + 2 * offset),
            bins="log",
            cmap="Blues",
            linewidths=0.1,
        )
        hb_ax_tuples.append((hb, ax))

        ax.plot(
            (lower - 2 * offset, upper + 2 * offset),
            (lower - 2 * offset, upper + 2 * offset),
            linestyle="dashed",
            zorder=1,
            color="black",
            alpha=0.5,
        )

        ax.set_xlim(lower - offset, upper + offset)
        ax.set_ylim(lower - offset, upper + offset)
        ax.set_aspect("equal", "box")

        ax.set_xlabel(target_a)
        ax.set_ylabel(target_b)

        print(f"Correlation coefficient {target_a}, {target_b}: {pearson_rs[frozenset({target_a, target_b})]:.3f}")

    # Set set norm for all plots
    all_bins = np.concatenate([hb.get_array() for hb, ax in hb_ax_tuples], axis=0)
    norm = matplotlib.colors.LogNorm(vmin=np.min(all_bins), vmax=np.max(all_bins))
    for hb, ax in hb_ax_tuples:
        hb.set_norm(norm)

    last_hb, last_ax = hb_ax_tuples[-1]
    fig.colorbar(last_hb, ax=last_ax, pad=0.10)
    # cb.set_label('counts')

    fig.savefig("correlation.pdf")


if __name__ == "__main__":
    main()
