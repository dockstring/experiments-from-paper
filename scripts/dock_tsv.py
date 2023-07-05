""" Run docking on a tsv file """

import argparse

import dockstring
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

if __name__ == "__main__":
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file to dock.")
    parser.add_argument("--output_path", type=str, required=True, help="Path of file to output")
    parser.add_argument("--target", type=str, required=True, help="Name of target for docking.")
    args = parser.parse_args()

    # Start by opening the output file for continuous writing
    df = pd.read_csv(args.input_file, sep="\t", header=0)
    smiles = list(df["smiles"].values)
    docking_scores = []
    target = dockstring.load_target(args.target)
    for s in tqdm(smiles):
        try:
            docking_output = target.dock(s)
            score = docking_output[0]
        except dockstring.DockstringError:
            score = float("nan")
        docking_scores.append(score)
    df[args.target] = np.asarray(docking_scores)

    # Write output file
    df.to_csv(args.output_path, sep="\t", index=False, header=True)
