import torch
import torch.nn as nn
import inspect

class Model(nn.Module):
    def __init__(
            self,
            model: Union[str, Path] = "yolo11n.pt",
            task: str = None,
            verbose: bool = False,
    ) -> None:
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
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
        self.cfg = cfg
        self.task = task
        self.model = self._smart_load(task)(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
        self.model_name = cfg

    def _smart_load(self, task: str):
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
                emojis(f"WARNING '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

