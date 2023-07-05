import dockstring_data
import numpy as np
import pandas as pd
from regression.gp_exact import get_dataset, get_predictions, load_model
from virtual_screening import get_base_parser

if __name__ == "__main__":
    parser = get_base_parser()
    args = parser.parse_args()

    # Load data
    if args.dataset.endswith(".csv"):
        df_pred = pd.read_csv(args.dataset, sep=",", header=0)
    else:
        df_pred = pd.read_csv(args.dataset, sep="\t", header=0)
    df_pred = dockstring_data.process_dataframe(df_pred, targets=None, fp=True)
    dataset_pred = get_dataset(df_pred)
    del df_pred["fp"]

    # Load model
    model = load_model(args.load_model_dir)

    # Make predictions
    y_pred, y_pred_var = get_predictions(model, dataset_pred, include_var=True)
    y_pred = np.asarray(y_pred).flatten()
    y_pred_var = np.asarray(y_pred_var).flatten()
    y_pred_std = np.sqrt(y_pred_var)

    # Save everything
    df_pred["y_pred"] = y_pred
    df_pred["y_pred_std"] = y_pred_std
    df_pred.to_csv(args.pred_save_path, sep="\t", index=False, header=True)
