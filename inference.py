import argparse

from tosem.core import easy_inference


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

    parser.add_argument(
        "--ckpt",
        metavar="N",
        help="Path to ckpt/pth model weights file.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="N",
        help="Binary segmentation mask threshold",
    )

    parser.add_argument(
        "--apply-mask",
        action="store_true",
        help="If True, apply masks to original images and save them.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    easy_inference(args)
