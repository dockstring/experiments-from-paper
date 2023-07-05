import argparse
import functools
import json
import logging

import deepchem as dc
import dockstring_data
import numpy as np
import pandas as pd
from deepchem.models.torch_models import MPNNModel
from regression.regression_utils import (
    eval_regression,
    get_regression_parser,
    split_dataframe_train_test,
)

logging.basicConfig(level=logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_epochs", type=int, default=10)
    return parser


def get_dataset(df: pd.DataFrame, target=None):
    # Get SMILES and properties
    smiles = list(map(str, df["smiles"].values))
    if target is None:
        y = np.zeros((len(smiles), 1))
    else:
        y = df[target].values.reshape(-1, 1)

    # Produce official deepchem dataset
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    X = featurizer.featurize(smiles, log_every_n=25000)
    dataset = dc.data.NumpyDataset(X=X, y=y)
    return dataset


def _get_model():
    model = MPNNModel(mode="regression", n_tasks=1, batch_size=32, learning_rate=0.001)
    return model


def get_trained_model(train_dataset, num_epochs):
    model = _get_model()
    model.fit(dataset=train_dataset, nb_epoch=num_epochs)
    return model


def get_predictions(model, dataset):
    return model.predict(dataset)


def save_model(model, save_dir):
    model.save_checkpoint(max_checkpoints_to_keep=1, model_dir=save_dir)


def load_model(save_dir):
    model = _get_model()
    model.restore(save_dir + "/checkpoint1.pt")
    return model


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(parents=[get_parser(), get_regression_parser()])
    args = parser.parse_args()

    # Load and process dataframes
    process_df = functools.partial(
        dockstring_data.process_dataframe,
        targets=[args.target],
        fp=True,
        max_docking_score=args.max_docking_score,
    )
    if args.data_split is None:
        df_train = pd.read_csv(args.dataset, sep="\t", header=0)
        df_test = None
    else:
        df_train, df_test = split_dataframe_train_test(args.dataset, args.data_split, n_train=args.n_train)
        df_test = process_df(df_test)
    df_train = process_df(df_train)

    # Train model with train dataset
    dataset_train = get_dataset(df_train, target=args.target)
    model = get_trained_model(dataset_train, num_epochs=args.num_epochs)

    # Save weights
    if args.model_save_dir is not None:
        save_model(model, args.model_save_dir)

    # Test on test dataset
    if df_test is not None:
        dataset_test = get_dataset(df_test, target=args.target)

        # Get predictions from best estimator
        y_train_pred = get_predictions(model, dataset_train)
        y_test_pred = get_predictions(model, dataset_test)

        # Save results
        result_dict = dict(
            metrics_train=eval_regression(y_train_pred, dataset_train.y),
            metrics_test=eval_regression(y_test_pred, dataset_test.y),
        )
        if args.full_preds:
            result_dict["full_preds"] = dict(
                smiles=list(map(str, df_test.smiles)),
                y_true=list(map(float, dataset_test.y.flatten())),
                y_pred=list(map(float, y_test_pred.flatten())),
            )
        with open(args.output_path, "w") as f:
            json.dump(result_dict, f, indent=4)
