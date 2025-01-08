import torch
import torch.nn as nn
from loguru import logger

from utils.checks import check_suffix


def torch_safe_load(weight):
    """
    Attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Example:
    ```python
    ckpt, file = torch_safe_load("path/to/best.pt")
    ```
    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename
    """
    check_suffix(file=weight, suffix=".pt")
    try:
        ckpt = torch.load(weight, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        logger.warning(f"{e.name} is  missing module")

    return ckpt, weight


def attempt_load_one_weight(weight, is_train, device=None):
    """Loads a single model weights."""
    ckpt, weight_path = torch_safe_load(weight)  # load ckpt

    if "model" not in ckpt:
        raise ValueError(f"'model' key missing in checkpoint {weight_path}")

    # 根据 'model' 是 state_dict 还是完整 nn.Module，做区分
    model_data = ckpt["model"]

    if isinstance(model_data, dict):
        # 只保存了 model.state_dict()
        from models import PoseModel
        from easydict import EasyDict
        if PoseModel is None:
            raise ValueError(
                "Checkpoint contains only a state_dict, but no model_builder provided!\n"
                "Please provide a callable model_builder() that returns an uninitialized model."
            )
        # 创建模型实例 load_state_dict
        model_cfg = EasyDict(ckpt["train_args"]['POSE_MODEL'])
        if not is_train:
            model_cfg.FEATURE_EXTRACTION.PRETRAINED = False # inference not need to load pretrained mae weights

        model = PoseModel(model_cfg)
        model.load_state_dict(model_data)
        model.to(device).float()
    else:
        # nn.Module 对象序列化
        model = model_data.to(device).float()

    model.pt_path = weight_path  # attach *.pt file path to model
    model.class_names = ckpt.get("class_names", None)

    model = model.eval()

    # Return model and ckpt
    return model, ckpt
