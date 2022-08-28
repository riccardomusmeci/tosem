import os
import argparse
from src.core.train import train

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
warnings.filterwarnings("ignore", category=LightningDeprecationWarning) 


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser("Training config")
    
    parser.add_argument(
        "--config",
        default="config/config.yml",
        type=str,
        required=False,
        help="path to the YAML configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/Users/riccardomusmeci/Developer/github/output/smart-arrotino/pothole",
        type=str,
        help="local directory where the best model checkpoint is saved at the end of training. Default set to SM_OUTPUT_DATA_DIR env var."
    )
    
    parser.add_argument(
        "--data-dir",
        metavar="N",
        default="/Users/riccardomusmeci/Developer/data/smart-arrotino/pothole/dataset/split",
        help="Input data dir path."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args=args)
    
    
    
    
    