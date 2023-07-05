import argparse
import functools
import json
import logging
import pickle
import random

import numpy as np
import pandas as pd
import torch
from bo import acquisition_funcs, gp_bo
from fingerprints import smiles_to_fp_array
from gp import (
    TanimotoGP,
    fit_gp_hyperparameters,
)
from mol_opt import get_base_molopt_parser, get_cached_objective_and_dataframe


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--n_train_gp_best",
        type=int,
        default=2000,
        help="Number of top-scoring training points to use for GP.",
    )
    parser.add_argument(
        "--n_train_gp_rand",
        type=int,
        default=3000,
        help="Number of random training points to use for GP.",
    )
    parser.add_argument("--ucb_beta", type=float, required=True, help="Beta value for UCB.")
    parser.add_argument(
        "--max_bo_iter",
        type=int,
        default=10000,
        help="Maximum number of iterations of BO.",
    )
    parser.add_argument("--bo_batch_size", type=int, default=1, help="Batch size for BO.")
    parser.add_argument(
        "--ga_max_generations",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--ga_offspring_size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--ga_mutation_rate",
        type=float,
        default=1e-2,
    )
    return parser


def get_trained_gp(
    X_train,
    y_train,
):
    # Fit model using type 2 maximum likelihood
    model = TanimotoGP(train_x=torch.as_tensor(X_train), train_y=torch.as_tensor(y_train))
    fit_gp_hyperparameters(model)
    return model


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(parents=[get_parser(), get_base_molopt_parser()])
    args = parser.parse_args()

    # Load dataset
    dataset = pd.read_csv(args.dataset, sep="\t", header=0)

    # Get function to be optimized
    opt_func, df_processed = get_cached_objective_and_dataframe(
        objective_name=args.objective,
        dataset=dataset,
        minimize=not args.maximize,
        keep_nan=False,
        max_docking_score=0.0,
        dock_kwargs=dict(
            num_cpus=args.num_cpu,
        ),
        # process_df_kwargs=dict(fp=True)
    )
    dataset_smiles = set(map(str, df_processed.smiles))

    # Fit an exact GP
    all_smiles = list(dataset_smiles)
    all_smiles_sorted = sorted(all_smiles, reverse=True, key=lambda s: opt_func(s))
    smiles_train = all_smiles_sorted[: args.n_train_gp_best] + list(
        random.sample(all_smiles_sorted[args.n_train_gp_best :], k=args.n_train_gp_rand)
    )
    x_train = np.stack(list(map(smiles_to_fp_array, smiles_train))).astype(np.float32)
    y_train = np.asarray(opt_func(smiles_train, batch=True)).astype(np.float32)
    gp_model = get_trained_gp(x_train, y_train)
    del x_train, y_train, all_smiles, all_smiles_sorted

    # Decide on acquisition function
    def acq_f_of_time(bo_iter, status_dict):
        return functools.partial(acquisition_funcs.upper_confidence_bound, beta=args.ucb_beta)

    # Run GP-BO
    gp_bo.logger.setLevel(logging.DEBUG)
    bo_res = gp_bo.gp_bo_loop(
        gp_model=gp_model,
        scoring_function=opt_func,
        smiles_to_np_fingerprint=smiles_to_fp_array,
        acq_func_of_time=acq_f_of_time,
        max_bo_iter=args.max_bo_iter,
        bo_batch_size=args.bo_batch_size,
        ga_num_cpu=args.num_cpu,
        gp_train_smiles=smiles_train,
        smiles_pool=dataset_smiles,
        max_func_calls=args.max_func_calls,
        log_ga_smiles=True,
        ga_max_generations=args.ga_max_generations,
        ga_offspring_size=args.ga_offspring_size,
        ga_mutation_rate=args.ga_mutation_rate,
    )

    # Format results by providing new SMILES + scores
    new_smiles = [r["smiles"] for r in bo_res[0] if r["smiles"] not in dataset_smiles]
    new_smiles_scores = [opt_func(s) for s in new_smiles]
    new_smiles_raw_info = [opt_func.cache[s] for s in new_smiles]
    json_res = dict(
        gp_params=gp_model.hparam_dict,
        new_smiles=new_smiles,
        scores=new_smiles_scores,
        raw_scores=new_smiles_raw_info,
    )

    # Save results
    with open(args.output_path, "w") as f:
        json.dump(json_res, f, indent=2)
    if args.extra_output_path is not None:
        with open(args.extra_output_path, "wb") as f:
            pickle.dump(bo_res, f)
