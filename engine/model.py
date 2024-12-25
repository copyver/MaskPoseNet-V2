import inspect
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from loguru import logger

from engine import PoseTrainer, PosePredictor, PoseValidator, SegTrainer, SegPredictor, SegValidator
from models import PoseModel, SegModel
from nn.tasks import attempt_load_one_weight
from utils import yaml_model_load, RANK
from utils.callback import DefaultCallbacks
from utils.results import Results
from utils.torch_utils import print_model_summary
from cfg import get_cfg


class Model(nn.Module):
    def __init__(
            self,
            model: Union[str, Path],
            task: str = None,
            verbose: bool = False,
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
        model = str(model).strip()

        # Load or create new model
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

    def _new(self, cfg: str, task: str = None, verbose: bool = False):
        """
        Initializes a new model and infers the task type from the model definitions.

        This method creates a new model instance based on the provided configuration file. It loads the model
                configuration, infers the task type if not specified, and initializes the model using the appropriate
                class from the task map.

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
        print_model_summary(self.model)

    def _load(self, weights: str, task=None) -> None:
        """
        Loads a model from a checkpoint file or initializes it from a weights file.

        This method handles loading models from either .pt checkpoint files or other weight file formats. It sets
        up the model, task, and related attributes based on the loaded weights.

        Args:
            weights (str): Path to the model weights file to be loaded.
            task (str | None): The task associated with the model. If None, it will be inferred from the model.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.

        Examples:
            >>> model = Model()
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = task or self.model.task
            self.ckpt_path = self.model.pt_path
            self.cfg = edict(self.ckpt['train_args'])
            logger.info(self.ckpt['train_args'].keys())
        else:
            raise ValueError(f"Unsupported weights file '{weights}'")
        self.model_name = weights

    def train(self, trainer=None, **kwargs):
        self._check_is_pytorch_model()

        self.cfg.IS_TRAIN = True
        if self.cfg.get("RESUME"):
            self.cfg.RESUME = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(cfg=self.cfg, model=self.model,
                                                                callbacks=DefaultCallbacks)
        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

    def predict(
            self,
            source: Dict = None,
            stream: bool = False,
            predictor=None,
            **kwargs,
    ) -> List[Results]:
        """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): The source
                of the image(s) to make predictions on. Accepts various types including file paths, URLs, PIL
                images, numpy arrays, and torch tensors.
            stream (bool): If True, treats the input source as a continuous stream for predictions.
            predictor (BasePredictor | None): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor.
            **kwargs (Any): Additional keyword arguments for configuring the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, each encapsulated in a
                Results object.

        Examples:

        Notes:
            - If 'source' is not provided, it defaults to the ASSETS constant with a warning.
            - The method sets up a new predictor if not already present and updates its arguments with each call.
            - For SAM-type models, 'prompts' can be passed as a keyword argument.
        """
        if source is None:
            logger.warning(f"WARNING 'source' is missing. Using 'source={source}'.")

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

        This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
        purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
        defaults, and any additional arguments provided.

        Args:
            **kwargs (Dict): Arbitrary keyword arguments to customize the export process. These are combined with
                the model's overrides and method defaults. Common arguments include:
                format (str): Export format (e.g., 'onnx', 'engine', 'coreml').
                half (bool): Export model in half-precision.
                int8 (bool): Export model in int8 precision.
                device (str): Device to run the export on.
                workspace (int): Maximum memory workspace size for TensorRT engines.
                nms (bool): Add Non-Maximum Suppression (NMS) module to model.
                simplify (bool): Simplify ONNX model.

        Returns:
            (str): The path to the exported model file.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            ValueError: If an unsupported export format is specified.
            RuntimeError: If the export process fails due to errors.
        """
        self._check_is_pytorch_model()
        from engine.exporter import Exporter

        # custom = {
        #     "imgsz": self.model.args["imgsz"],
        #     "batch": 1,
        #     "data": None,
        #     "device": None,  # reset to avoid multi-GPU errors
        #     "verbose": False,
        # }  # method defaults
        # args = {**custom, **kwargs, "mode": "export"}  # highest priority args on the right
        if override:
            self.cfg = get_cfg(self.cfg, override)
            print(self.cfg)
        return Exporter(cfg=self.cfg, _callbacks=DefaultCallbacks)(model=self.model)

    def _check_is_pytorch_model(self) -> None:
        """
        Checks if the model is a PyTorch model and raises a TypeError if it's not.

        This method verifies that the model is either a PyTorch module or a .pt file. It's used to ensure that
        certain operations that require a PyTorch model are only performed on compatible model types.

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
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def _smart_load(self, key: str):
        """
        Loads the appropriate module based on the model task.

        This method dynamically selects and returns the correct module (model, trainer, validator, or predictor)
        based on the current task of the model and the provided key. It uses the task_map attribute to determine
        the correct module to load.

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
