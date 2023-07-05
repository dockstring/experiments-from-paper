import argparse
import functools
import json
from pathlib import Path

import dockstring_data
import numpy as np
import pandas as pd
from regression.regression_utils import (
    eval_regression,
    get_regression_parser,
    split_dataframe_train_test,
)
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV

SAVE_FILE_NAME = "weights.npz"

cv_params = {"alpha": np.geomspace(1e-4, 1e4, 1000)}


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--num_models",
        type=int,
        default=20,
        help="Number of models to try in CV search.",
    )
    return parser


def get_dataset(df: pd.DataFrame, target=None):
    X = np.stack(df["fp"].values)
    if target is None:
        y = np.zeros((len(X), 1))
    else:
        y = df[target].values.reshape(-1, 1)
    return X, y


def get_trained_model(train_dataset, num_models):
    X_train, y_train = train_dataset

    # Fit model using CV search
    model = Lasso()
    random_search = RandomizedSearchCV(
        model,
        param_distributions=cv_params,
        n_iter=num_models,
        verbose=100,
        n_jobs=-1,
    )
    cv_results = random_search.fit(X_train, y_train)

    # Get predictions from best estimator
    return cv_results.best_estimator_


def get_predictions(model, dataset):
    X, _ = dataset
    return model.predict(X)


def save_model(model: Lasso, save_dir):
    np.savez_compressed(
        Path(save_dir) / SAVE_FILE_NAME,
        coef_=model.coef_,
        intercept_=model.intercept_,
        alpha=model.alpha,
    )


def load_model(save_dir):
    model = Lasso()
    with np.load(Path(save_dir) / SAVE_FILE_NAME) as npz:
        model.coef_ = npz["coef_"]
        model.intercept_ = npz["intercept_"]
        model.alpha = npz["alpha"]
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
    best_model = get_trained_model(dataset_train, num_models=args.num_models)

    # Save weights
    if args.model_save_dir is not None:
        Path(args.model_save_dir).mkdir(exist_ok=True)
        save_model(best_model, args.model_save_dir)

    # Test on test dataset
    if df_test is not None:
        dataset_test = get_dataset(df_test, target=args.target)

        # Get predictions from best estimator
        y_train_pred = get_predictions(best_model, dataset_train)
        y_test_pred = get_predictions(best_model, dataset_test)

        # Save results
        result_dict = dict(
            metrics_train=eval_regression(y_train_pred, dataset_train[1]),
            metrics_test=eval_regression(y_test_pred, dataset_test[1]),
            model_params=dict(alpha=best_model.alpha),
        )
        if args.full_preds:
            result_dict["full_preds"] = dict(
                smiles=list(map(str, df_test.smiles)),
                y_true=list(map(float, dataset_test[1].flatten())),
                y_pred=list(map(float, y_test_pred.flatten())),
            )
        with open(args.output_path, "w") as f:
            json.dump(result_dict, f, indent=4)
