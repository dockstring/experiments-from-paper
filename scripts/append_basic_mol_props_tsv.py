""" Run docking on a tsv file """

import argparse

import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED as qed_module
from rdkit.Chem import Crippen
from tqdm.auto import tqdm

if __name__ == "__main__":
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input tsv file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path of tsv file to output.")
    args = parser.parse_args()

    # Read all molecules from df
    df = pd.read_csv(args.input_file, sep="\t", header=0)
    smiles = list(df["smiles"].values)
    df["logP"] = [Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in tqdm(smiles, desc="logP")]
    df["QED"] = [qed_module.qed(Chem.MolFromSmiles(s)) for s in tqdm(smiles, desc="QED")]

    # Write output file
    df.to_csv(args.output_path, sep="\t", index=False, header=True)
