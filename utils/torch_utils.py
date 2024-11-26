import os
import random

import numpy as np
import torch
from loguru import logger
from contextlib import contextmanager
import torch.distributed as dist

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))


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
