"""
Given a list of molecules,
find the nearest neighbours in the dockstring dataset
and output to a csv.

By default, it is configured to work with the ZINC dataset
"""

import argparse

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, DataStructs, RDKFingerprint


def standard_fingerprint(mol):
    return RDKFingerprint(mol, maxPath=6)


# Standard argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=str,
    default="./data/dockstring-dataset-extra-props.tsv",
    help="Path to dataset.",
)
parser.add_argument(
    "--query_csv_path",
    type=str,
    required=True,
    help="Path to csv file of query molecules.",
)
parser.add_argument(
    "--query_csv_key",
    type=str,
    default="zinc_id",
    help="How to denote molecules from the query set (what column of csv).",
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="(csv) file to output results to.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Read dockstring dataset and find fingerprints
    df_data = pd.read_csv(args.dataset_path, sep="\t")
    df_smiles = list(df_data["smiles"])
    df_inchi = list(df_data["inchikey"])
    df_mols = list(map(AllChem.MolFromSmiles, df_smiles))
    df_fps = list(map(standard_fingerprint, df_mols))
    del df_mols
    print("Dockstring dataset read and processed")

    # Read in the query dataset
    df_query = pd.read_csv(args.query_csv_path)

    # Go through rows one at a time and find nearest neighbours
    output = []
    for row_idx, (_, row) in enumerate(df_query.iterrows()):
        fp_query = standard_fingerprint(AllChem.MolFromSmiles(row["smiles"]))
        sims = DataStructs.BulkTanimotoSimilarity(fp_query, df_fps)
        i = np.argmax(sims)
        output.append(
            {
                args.query_csv_key: row[args.query_csv_key],
                "NN_inchi": df_inchi[i],
                "NN_dist": sims[i],
            }
        )

        if row_idx % 1_000 == 0:
            print(f"Done row {row_idx}", flush=True)

    # Output result to file
    df_out = pd.DataFrame(output)
    df_out.to_csv(args.output_path, index=False)
    print("End of script!")
