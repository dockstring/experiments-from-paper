# Dockstring baselines

This repo contains:

1. Code to run the experiments in the [dockstring paper](https://doi.org/10.1021/acs.jcim.1c01334).
2. The numerical results used in the paper.
3. Code to generate most figures / tables in the paper.

This code is not super polished, but should be sufficient to reproduce the results.
In particular, we note that this code does not use the dockstring benchmark API
which we added after running our experiments.
We recommend you use this for new code.

Everything should run correctly if you follow the instructions below.
Please contact us or raise a github issue if anything doesn't work for you.

## Data preparation

There are 2 datasets used in the benchmarking:

1. Our subset of ExCAPE, downloaded from figshare (around 0.2 Gb)
2. The ZINC 20 dataset, downloaded from their website (around 60 Gb).

### Dockstring dataset

First, download the dockstring dataset (instructions [here](https://github.com/dockstring/dataset))
and place the contents in a folder called `data` (optionally could be a simlink).
In particular, the following files should be present:

- `./data/dockstring-dataset.tsv`
- `./data/cluster_split.tsv`

Some of the dockstring tasks require `logP` and `QED` values.
To avoid recomputing these, the following commands can be run to make an augmented version of the dockstring dataset with these extra columns:

```bash
bash scripts/dockstring-add-extra-props.sh
```

Some of the scripts below use the file generated by this command,
so if you want to reproduce the experiments you may need to run this command.

### ZINC dataset

Unfortunately, the terms of use of ZINC prohibit us from redistributing "large subsets of ZINC",
so at the moment we are unable to post a link to the dataset.
You should be able to download molecules directly from ZINC and assemble them into a csv file
like the following:

```
zinc_id,smiles
7,C=CCc1ccc(OCC(=O)N(CC)CC)c(OC)c1
10,C[C@@]1(c2ccccc2)OC(C(=O)O)=CC1=O
11,COc1cc(Cc2cnc(N)nc2N)cc(OC)c1N(C)C
12,O=C(C[S@@](=O)C(c1ccccc1)c1ccccc1)NO
17,CCC[S@](=O)c1ccc2[nH]/c(=N\C(=O)OC)[nH]c2c1
18,C=CCN1C(=O)[C@@H](CC(C)C)NC1=S
22,C=C(C)CNc1ccc([C@H](C)C(=O)O)cc1
23,C=CCc1ccccc1OC[C@H](O)CNC(C)C
24,O[C@@H]1C=C2CCN3Cc4cc5c(cc4[C@@H]([C@H]23)[C@H]1O)OCO5
[...]
```

Our file has the molecules in sorted order (by ZINC ID) and contains 997597004 lines.
Its `md5sum` is `e65b33ddf9e7178aba732e396e72fe54`.
Alternatively, feel free to contact us give you the dataset.
We know it's annoying to have data available upon request to the authors,
but our hands are somewhat tied on this issue...

This file should be accessible from `data/zinc/zinc20-sorted.csv` in order for the scripts to work correctly
(using a symlink if needed).

## Python environment

Different parts of the code have different requirements.
The basic requirements are:

- python 3.7+ (tested on 3.7)
- scipy
- numpy
- pandas
- rdkit
- scikit-learn
- joblib

For specific tasks:

- deepchem (for deepchem regression models)
- gpytorch and botorch (for Gaussian process regression and BO)
- matplotlib (for plotting)
- xgboost (for xgboost regression model)

## How to reproduce all experiments from the paper

### Regression on dockstring dataset

1. Ensure that the dockstring dataset is downloaded (see [Data Preparation](#data-preparation))
2. Ensure your [python environment](#python-environment) is set up for the models you want to run.
3. Run the scripts in `src/regression`.
  Bash scripts to supply the correct arguments to the python scripts are located in
  `experiments/regression`.
  For example, to run all experiments for the lasso method, run `bash experiments/regression/lasso.sh`.
  This will produce json output files in the `./results` directory
  with train and test set metrics.
  These scripts can be modified to dock different proteins,
  use a different train/test split, etc.
  There are similar scripts in `experiments/regression-toy` for QED and logP.
4. To reproduce the regression results table in the paper,
  run the script `python scripts/results_regression.py`
  (various arguments can be provided to change the table formatting).
  Since we don't explicitly control random seeds the results probably won't match
  exactly but they should be very similar.

### Virtual screening with ZINC

#### Training the virtual screening models

1. Ensure that the dockstring ExCAPE dataset is downloaded (see [Data Preparation](#data-preparation))
2. Ensure [Python Environment](#python-environment) is set up.
3. Run whichever tasks you want. For example, `bash experiments/virtual_screening/training/ridge.sh`.
  The trained model weights will be dumped into the `results` folder.

#### Predictions on ZINC (or another dataset)

1. Ensure that the ZINC dataset is downloaded (see [Data Preparation](#data-preparation))
2. Split up the ZINC dataset into chunks of size 1M by running: `bash experiments/virtual_screening/zinc-predictions/split_zinc.sh`
3. Get the model weights, either by [training new models](#training-the-virtual-screening-models) or by copying in the old weights from `official_results`.
  For example, you can run `mkdir -p results && cp -r official_results/virtual-screening results/`
4. Get model predictions, using the `pred_csv` argument to feed in the file to be predicted.
  For example, to run predictions for the first split of ZINC,
  run `pred_csv="data/zinc/split-size-1000000/zinc-split000000000.csv" bash experiments/virtual_screening/zinc-predictions/ridge.sh` (all on one line).
5. Find the top predictions and put them into 1 file. This can be done by running `bash experiments/virtual_screening/zinc-predictions/topn_predictions.sh`, which will find the 5000 lowest docking score and output them to a csv.
6. Find their actual docking scores using the `scripts/dock_tsv.py` script. To do this automatically, run `bash experiments/virtual_screening/zinc-predictions/score_top_predictions.sh`. Alternetively, you can split the files using `scripts/split_smiles_csv.sh` and score the predictions in parallel.
7. To reproduce the results in the paper, run: `python scripts/results_virtual_screening.py`

### Molecular optimization (*de novo* design)

1. Ensure that the dockstring dataset is downloaded (see [Data Preparation](#data-preparation))
2. Ensure your [python environment](#python-environment) is set up for the models you want to run.
3. Run the scripts in `src/mol_opt`.
  Bash scripts to supply the correct arguments to the python scripts are located in
  `experiments/molopt`.
  For example, to run all experiments for the graph GA method, run `bash experiments/molopt/graph_ga.sh`.
  This will produce json output files in the `./results` directory
  with the new SMILES and their scores.
  Note that these scripts can take up to 100 hours to run!
  These scripts can be modified to test different objectives,
  use a different budget, or different dataset, etc.
  Here are some useful notes about the scripts:
   - you may want to only run a specific experiment, which can be done by supplying an experiment index.
     For example, `expt_idx=12 bash experiments/molopt/graph_ga.sh`
   - you may also want to reduce the `num_cpus` argument if you don't have 8 CPUs
     (but there is little point to increasing it beyond 8 due to how VINA works).
   - Because the scripts take a long time to run, by default they leave extensive log files,
     and a much larger result file at the end.
     These have been omitted from the repo and can be deleted after the run has done.
4. To reproduce the results in the paper, run: `python scripts/results_molopt.py`

## Development and PRs

This code is not actively developed, but we set up pre-commit to handle code formatting and some other checks.
If you want to contribute to this repo (e.g. by opening a pull request),
please install pre-commit and run it on your local version before committing anything:

```bash
conda install pre-commit  # if not already installed
pre-commit install
```