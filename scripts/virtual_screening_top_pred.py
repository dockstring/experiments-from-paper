""" Get top predictions from virtual screening """

import argparse
import heapq
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

if __name__ == "__main__":
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", required=True, help="Input files to sort.")
    parser.add_argument("--n_top", type=int, required=True, help="Number of top SMILES to keep.")
    parser.add_argument("--output_path", type=str, required=True, help="Path of file to output")
    args = parser.parse_args()

    # Read in files and maintain the smallest at all times
    best_list = list()
    header = None
    for input_file in tqdm(args.input_files):
        assert Path(input_file).exists(), input_file

        # Read in lines of file
        with open(input_file) as f:
            lines = list(f.readlines())
            new_header = lines[0]
            if header is None:
                header = new_header
            else:
                assert new_header == header, "File headers must match!"
            lines = lines[1:]  # Skip header

        # Read in pandas csv
        df = pd.read_csv(input_file, sep="\t", header=0)
        assert len(df) == len(lines)

        # Create tuples
        y_pred_list = list(map(float, df["y_pred"].values.flatten()))
        score_line_tuples = list(zip(y_pred_list, lines))

        # Do the sorting
        candidate_list = best_list + score_line_tuples
        best_list = heapq.nsmallest(args.n_top, candidate_list)

    # Write output file
    with open(args.output_path, "w") as f:
        f.write(header)
        for _, line in best_list:
            f.write(line)
