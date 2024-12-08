import inspect
from pathlib import Path
from typing import Union

import torch.nn as nn

from engine import PoseTrainer, PosePredictor, PoseValidator, SegTrainer, SegPredictor, SegValidator
from models import PoseModel, SegModel
from nn.tasks import attempt_load_one_weight
from utils import yaml_model_load
from utils.callback import DefaultCallbacks
from utils.torch_utils import (
    RANK,
    print_model_summary
)


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
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def train(self, trainer=None, **kwargs):
        self._check_is_pytorch_model()

        self.cfg.IS_TRAIN = True
        if self.cfg.get("RESUME"):
            self.cfg.RESUME = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(cfg=self.cfg, model=self.model, callbacks=DefaultCallbacks)
        # if not self.cfg.get("RESUME"):  # manually set model only if not resuming
        #     self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
        #     self.model = self.trainer.model

        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

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
                print(f"WARNING '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
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
