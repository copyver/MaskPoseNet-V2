import math
import os
import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from utils import NUM_THREADS, __version__
from utils.checks import check_version

TORCH_2_4 = check_version(torch.__version__, "2.4.0")
TORCH_1_9 = check_version(torch.__version__, "1.9.0")


def convert_optimizer_state_dict_to_fp16(state_dict):
    """
    Converts the state_dict of a given optimizer to FP16, focusing on the 'state' key for tensor conversions.

    This method aims to reduce storage size without altering 'param_groups' as they contain non-tensor data.
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict


def autocast(enabled: bool):
    """
    Get the appropriate autocast context manager based on PyTorch version and AMP setting.

    This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both
    older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.

    Returns:
        (torch.amp.autocast): The appropriate autocast context manager.

    """
    return torch.amp.autocast('cuda', enabled=enabled)


def get_gpu_info(index):
    """Return a string with system GPU information, i.e. 'Tesla T4, 15102MiB'."""
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    import cpuinfo  # pip install py-cpuinfo
    k = "brand_raw", "hardware_raw", "arch_string_raw"  # keys sorted by preference
    info = cpuinfo.get_cpu_info()  # info dict
    string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], "unknown")
    string = string.replace("(R)", "").replace("CPU ", "").replace("@ ", "")
    return string


def select_device(device="", batch=0):
    """
    Selects the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object.
            Options are 'None', 'cpu', or 'cuda', or '0' or '0,1,2,3'. Defaults to an empty string, which auto-selects
            the first available GPU, or CPU if no GPU is available.
        batch (int, optional): Batch size being used in your model. Defaults to 0.

    Returns:
        (torch.device): Selected device.

    Raises:
        ValueError: If the specified device is not available or if the batch size is not a multiple of the number of
            devices when using multiple GPUs.

    Examples:
        >>> select_device("cuda:0")
        device(type='cuda', index=0)

        >>> select_device("cpu")
        device(type='cpu')

    Note:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """
    if isinstance(device, torch.device) or str(device).startswith("tpu"):
        return device

    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])  # remove sequential commas, i.e. "0,,1" -> "0,1"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # i.e. "0,1" -> ["0", "1"]
        n = len(devices)  # device count
        if n > 1:  # multi-GPU
            if batch < 1:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    "please specify a valid batch size, i.e. batch=16."
                )
            if batch >= 0 and batch % n != 0:  # check batch_size is divisible by device_count
                raise ValueError(
                    f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                    f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}."
                )
        for i, d in enumerate(devices):
            logger.info(f"CUDA:{d}: ({get_gpu_info(i)})\n")
        arg = "cuda:0"
    elif mps and torch.backends.mps.is_available():
        # Prefer MPS if available
        logger.info(f"MPS ({get_cpu_info()})\n")
        arg = "mps"
    else:  # revert to CPU
        logger.info(f"CPU ({get_cpu_info()})\n")
        arg = "cpu"

    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)  # reset OMP_NUM_THREADS for cpu training

    return torch.device(arg)


def init_seeds(seed=0):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Ensures all processes in distributed training wait for the local master (rank 0) to complete a task first."""
    initialized = dist.is_available() and dist.is_initialized()

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank])


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


class EarlyStopping:
    """Early stopping class that stops training when a specified number of epochs have passed without improvement."""

    def __init__(self, patience=50):
        """
        Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False

        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            prefix = "EarlyStopping: "
            logger.info(
                f"{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop


def print_model_summary(model, verbose=False):
    # 打印模型的结构
    logger.info(f"\n===== Model Structure =====\n{model}")

    total_params = 0
    for name, param in model.named_parameters():
        if verbose:
            logger.info(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
        total_params += param.numel()

    logger.info(f"Total number of parameters: {total_params}")


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


def smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(fn):
        """Applies appropriate torch decorator for inference mode based on torch version."""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # already in inference_mode, act as a pass-through
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)

    return decorate


def strip_optimizer(f: Union[str, Path] = "best.pt", s: str = "", to_half: bool = False, updates: dict = None) -> dict:
    """
    Strip optimizer from 'f' to finalize training, optionally save as 's'.

    Args:
        f (str): file path to model to strip the optimizer from. Default is 'best.pt'.
        s (str): file path to save the model with stripped optimizer to. If not provided, 'f' will be overwritten.
        to_half (bool): if to convert to half
        updates (dict): a dictionary of updates to overlay onto the checkpoint before saving.

    Returns:
        (dict): The combined checkpoint dictionary.

    Example:
        ```python
        from pathlib import Path
        from ultralytics.utils.torch_utils import strip_optimizer

        for f in Path("path/to/model/checkpoints").rglob("*.pt"):
            strip_optimizer(f)
        ```

    Note:
        Use `ultralytics.nn.torch_safe_load` for missing modules with `x = torch_safe_load(f)[0]`
    """
    try:
        ckpt = torch.load(f, map_location=torch.device("cpu"))
        assert isinstance(ckpt, dict), "Checkpoint is not a Python dictionary."
        assert "model" in ckpt, "Missing 'model' key in checkpoint."
    except Exception as e:
        logger.warning(f"WARNING: Skipping {f}, not a valid checkpoint: {e}")
        return {}

    metadata = {
        "date": datetime.now().isoformat(),
        "version": __version__,
    }

    # Update model
    # 只保留基本推理所需的 "model" (state_dict)，故以下键可以被置空
    ckpt["optimizer"] = None
    ckpt["best_fitness"] = None
    ckpt["train_results"] = None

    # 重置 epoch
    ckpt["epoch"] = -1

    # 将模型参数转为 FP16
    # model 现已是 state_dict()，我们要遍历其中所有张量，并转换为 half()
    if to_half:
        # model 现已是 state_dict()，我们要遍历其中所有张量，并转换为 half()
        for k, v in ckpt["model"].items():
            if isinstance(v, torch.Tensor):
                ckpt["model"][k] = v.half()

    # Update other keys
    args = {**ckpt.get("train_args", {})}  # combine cfg
    ckpt["train_args"] = {k: v for k, v in args.items()}  # strip non-default keys

    # Save
    combined = {**metadata, **ckpt, **(updates or {})}
    torch.save(combined, s or f)  # combine dicts (prefer to the right)
    mb = os.path.getsize(s or f) / 1e6  # file size
    logger.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
    return combined


def get_latest_opset():
    """Return the second-most recent ONNX opset version supported by this version of PyTorch, adjusted for maturity."""
    return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1

