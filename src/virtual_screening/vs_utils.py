import argparse


def get_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_model_dir",
        type=str,
        required=True,
        help="Directory to load saved model parameters from.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to tsv file with dataset to predict.",
    )
    parser.add_argument(
        "--pred_save_path",
        type=str,
        required=True,
        help="Path to new tsv file for saving predictions.",
    )

    return parser
