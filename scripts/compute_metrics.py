import argparse
from typing import Any, Dict

import pandas as pd
from rdkit.Chem import AllChem, Descriptors


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="path to input TSV file", required=True)
    return parser.parse_args(args=args)


def parse_smiles(smiles: str) -> AllChem.Mol:
    mol = AllChem.MolFromSmiles(smiles)
    if not mol:
        raise RuntimeError(f"Cannot parse {smiles}")
    return mol


def compute_metrics(inchikey: str, smiles: str) -> Dict[str, Any]:
    mol = parse_smiles(smiles)
    return {
        "inchikey": inchikey,
        "qed": AllChem.QED.qed(mol),
        "logp": Descriptors.MolLogP(mol),
        "mol_weight": Descriptors.ExactMolWt(mol),
        "num_rot_bonds": AllChem.CalcNumRotatableBonds(mol),
        "num_hba": AllChem.CalcNumHBA(mol),
        "num_hbd": AllChem.CalcNumHBD(mol),
    }


def main():
    args = parse_args()

    print(f"Reading file: {args.dataset}")
    df = pd.read_csv(args.dataset, sep="\t")
    df = df.set_index("inchikey")

    output = pd.DataFrame(compute_metrics(key, smiles) for key, smiles in df["smiles"].items())

    # Write output
    output_file_name = "metrics.tsv"
    print(f"Writing file: {output_file_name}")
    output.to_csv(output_file_name, sep="\t", header=True, index=False)


if __name__ == "__main__":
    main()
