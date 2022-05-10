import os
import yaml
from typing import Dict


def load_yml(path: str) -> Dict:
    """loads a single yml file

    Args:
        path (str): path to yml file

    Returns:
        Dict: yml dict
    """
    with open(path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
    return params

def load_config(path: str) -> Dict:
    """loads a single yml file

    Args:
        path (str): path to yml file

    Returns:
        Dict: yml dict
    """
    with open(path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
    return params

def load_configs(config_dir: str):
    
    # loading configurations for transform, model and train
    transform_config = load_yml(path=os.path.join(config_dir, "transform.yml"))
    model_config = load_yml(path=os.path.join(config_dir, "model.yml"))
    train_config = load_yml(path=os.path.join(config_dir, "train.yml"))
    dataset_config = load_yml(path=os.path.join(config_dir, "dataset.yml"))
    
    # modifying config by adding missing fields
    transform_config["train"]["height"] = model_config["input_size"]["height"]
    transform_config["train"]["width"] = model_config["input_size"]["width"]
    transform_config["val"]["height"] = model_config["input_size"]["height"]
    transform_config["val"]["width"] = model_config["input_size"]["width"]
    
    return transform_config, model_config, train_config, dataset_config
    
    
    