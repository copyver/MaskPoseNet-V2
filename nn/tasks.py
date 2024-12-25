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

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        logger.warning(
            f"WARNING The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, weight


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt

    model = ckpt["model"].to(device).float()  # FP32 model

    model.pt_path = weight  # attach *.pt file path to model

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt
