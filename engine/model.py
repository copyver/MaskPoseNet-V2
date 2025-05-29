import inspect
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from loguru import logger

from cfg import get_cfg
from engine import PoseTrainer, PosePredictor, PoseValidator, SegTrainer, SegPredictor, SegValidator
from models import PoseModel, SegModel
from nn.tasks import attempt_load_one_weight
from utils import yaml_model_load, RANK, yaml_print
from utils.callback import DefaultCallbacks
from utils.results import Results
from utils.torch_utils import print_model_summary


class Model(nn.Module):
    def __init__(
            self,
            model: Union[str, Path],
            task: str = None,
            verbose: bool = False,
            is_train: bool = True,
            device: str = "cuda",
    ) -> None:
        super().__init__()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.metrics = None  # validation/training metrics
        self.task = task  # task type
        self.is_train = is_train
        self.device = device
        model = str(model).strip()

        # Load or create new model
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task, is_train=is_train)

    def _new(self, cfg: str, task: str = None, verbose: bool = False):
        """
        Initializes a new model and infers the task type from the model definitions.
        Args:
            cfg (str): Path to the model configuration file in YAML format.
            task (str | None): The specific task for the model. If None, it will be inferred from the config.

        Raises:
            ValueError: If the configuration file is invalid or the task cannot be inferred.
            ImportError: If the required dependencies for the specified task are not installed.
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg_dict
        self.task = task
        self.model = self._smart_load("model")(cfg_dict.POSE_MODEL)  # build model
        print_model_summary(self.model, verbose)

    def _load(self, weights: str, task: str = None, is_train: bool = True) -> None:
        """
        Loads a model from a checkpoint file or initializes it from a weights file.

        Args:
            weights (str): Path to the model weights file to be loaded.
            task (str | None): The task associated with the model. If None, it will be inferred from the model.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.
        """
        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights, is_train, self.device)
            self.task = task or self.model.task
            self.ckpt_path = self.model.pt_path
            self.cfg = edict(self.ckpt['train_args'])
            logger.info(self.ckpt['train_args'].keys())
        else:
            raise ValueError(f"Unsupported weights file '{weights}'")

    def train(self, trainer=None, **kwargs):
        """
        Trains the model using the specified dataset and training configuration.

        Args:
            trainer (BaseTrainer | None): Custom trainer instance for model training. If None, uses default.
            **kwargs (Any): Arbitrary keyword arguments for training configuration. Common options include:
                EPOCHS (int): Number of training epochs.
                DEVICE (str): Device to run training on (e.g., 'cuda', 'cpu').
        """

        self._check_is_pytorch_model()
        self.cfg.IS_TRAIN = True
        self.cfg.SOLVERS.update(kwargs)
        if self.cfg.SOLVERS.get("RESUME") and self.cfg.SOLVERS.RESUME:
            # Todo: resume training
            pass

        self.trainer = (trainer or self._smart_load("trainer"))(cfg=self.cfg, model=self.model,
                                                                callbacks=DefaultCallbacks)
        self.trainer.train()
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt, is_train=False)
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

    def predict(
            self,
            source: Dict = None,
            stream: bool = False,
            predictor=None,
            override=None,
    ) -> List[Results]:
        self._check_is_pytorch_model()
        self.cfg.IS_TRAIN = False
        if source is None:
            logger.warning(f"WARNING 'source' is missing. Using 'source={source}'.")
        if override:
            self.cfg = get_cfg(self.cfg, override)
            yaml_print(self.cfg.POSE_MODEL)

        if not self.predictor:
            self.predictor = (predictor or self._smart_load("predictor"))(cfg=self.cfg,
                                                                          _callbacks=DefaultCallbacks,
                                                                          verbose=True)
            self.predictor.setup_model(model=self.model)

        return self.predictor(source=source, stream=stream)

    def export(
            self,
            override=None,
    ) -> str:
        """
        Exports the model to a different format suitable for deployment.

        Returns:
            (str): The path to the exported model file.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            ValueError: If an unsupported export format is specified.
            RuntimeError: If the export process fails due to errors.
        """
        self._check_is_pytorch_model()
        from engine.exporter import Exporter

        if override:
            self.cfg = get_cfg(self.cfg, override)
            print(self.cfg)
        return Exporter(cfg=self.cfg, _callbacks=DefaultCallbacks)(model=self.model)

    def _check_is_pytorch_model(self) -> None:
        """
        Checks if the model is a PyTorch model and raises a TypeError if it's not.

        Raises:
            TypeError: If the model is not a PyTorch module or a .pt file. The error message provides detailed
                information about supported model formats and operations.
        """
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def _smart_load(self, key: str):
        """
        Loads the appropriate module based on the model task.

        Args:
            key (str): The type of module to load. Must be one of 'model', 'trainer', 'validator', or 'predictor'.

        Returns:
            (object): The loaded module corresponding to the specified key and current task.
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                f"WARNING '{name}' model does not support '{mode}' mode for '{self.task}' task yet."
            ) from e

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "segment": {
                "model": SegModel,
                "trainer": SegTrainer,
                "validator": SegValidator,
                "predictor": SegPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": PoseTrainer,
                "validator": PoseValidator,
                "predictor": PosePredictor,
            },
        }
