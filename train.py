import argparse

from tosem.core import easy_train


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser("Training config")

    parser.add_argument(
        "--config", default="config/config.yaml", type=str, required=False, help="path to the YAML configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="local directory where the best model checkpoint is saved at the end of training.",
    )

    parser.add_argument(
        "--data-dir",
        metavar="N",
        help="Input data dir path.",
    )

    # TODO: add resume from
    parser.add_argument("--resume-from", default=None, help="Path to ckpt file to resume training from.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    easy_train(args)
