from rdkit import Chem
from rdkit.Chem import QED as qed_module
from rdkit.Chem import Crippen, Descriptors

# Guacamol
guacamol_funcs = dict()
try:
    from guacamol import benchmark_suites as guac_benchmarks

    for benchmark in guac_benchmarks.goal_directed_suite_v2():
        converted_name = benchmark.name.lower().replace(" ", "-")
        guacamol_funcs[converted_name] = benchmark.objective.score
except ImportError:
    pass


def QED(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return qed_module.qed(mol)


def logP(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Crippen.MolLogP(mol)


def molecular_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)
