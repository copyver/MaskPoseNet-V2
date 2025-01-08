from utils import yaml_load, IterableSimpleNamespace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union
from easydict import EasyDict as edict


def cfg2dict(cfg):
    """
    Converts a configuration object to a dictionary.

    Args:
        cfg (str | Path | Dict | EasyDict |SimpleNamespace): Configuration object to be converted. Can be a file path,
            a string, a dictionary, or a SimpleNamespace object.

    Returns:
        (Dict): Configuration object in dictionary format.

    Examples:
        Convert a YAML file path to a dictionary:
        >>> config_dict = cfg2dict("config.yaml")

        Convert a SimpleNamespace to a dictionary:
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        Pass through an already existing dictionary:
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    Notes:
        - If cfg is a path or string, it's loaded as YAML and converted to a dictionary.
        - If cfg is a SimpleNamespace object, it's converted to a dictionary using vars().
        - If cfg is already a dictionary, it's returned unchanged.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    elif isinstance(cfg, edict):
        cfg = dict(cfg)
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, edict, SimpleNamespace] = None,
            overrides: Union[str, Path, Dict, edict, SimpleNamespace] = None):
    """
    Load and merge configuration data from a file or dictionary, with optional overrides.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data source. Can be a file path, dictionary, or
            SimpleNamespace object.
        overrides (str | Path | Dict | SimpleNamespace | None): Dictionary containing key-value pairs to override the base configuration.

    Returns:
        (SimpleNamespace): Namespace containing the merged configuration arguments.

    Examples:
        >>> config = get_cfg()  # Load default configuration
        >>> config = get_cfg("path/to/config.yaml", overrides={"EPOCHS": 50, "BATCH_SIZE": 16})

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric
          `project` and `name` to strings and validating configuration keys and values.
        - The function performs type and value checks on the configuration data.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)

        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Return instance
    return IterableSimpleNamespace(**cfg)