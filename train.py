import os
import argparse

from numpy import require
from src.core import train

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
        type=str,
        required=True,
        help="local directory where the best model checkpoint is saved at the end of training."
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
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
    
    
    
    
    