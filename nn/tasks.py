import yaml
from pathlib import Path
from easydict import EasyDict as edict


def yaml_model_load(path):
    """Load a model config from a YAML file."""
    path = Path(path)
    d = yaml_load(path)  # model dict
    d.yaml_file = str(path)
    return d


def yaml_load(file="base.yaml"):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'base.yaml'.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        return edict(data)


if __name__=="__main__":
    path = "../cfg/base.yaml"
    d = yaml_model_load(path)
    print(d)