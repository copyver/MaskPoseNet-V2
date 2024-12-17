import time
from pathlib import Path


class BaseValidator(object):
    """
    Base class for all validators.
    """

    def __init__(self, dataset=None, dataloader=None, save_dir=None, pbar=None, cfg=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.cfg = cfg
        self.dataset = dataset
        self.dataloader = dataloader
        self.pbar = pbar
        self.device = None
        self.training = True
        self.names = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plots = {}
        self.callbacks = _callbacks

    def __call__(self, trainer=None, model=None):
        raise NotImplementedError("__call__ function not implemented for this validator")

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        raise NotImplementedError("preprocess function not implemented in validator")

    def process(self, model, batch):
        raise NotImplementedError("process function not implemented in validator")

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        raise NotImplementedError("postprocess function not implemented in validator")

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        raise NotImplementedError("init_metrics function not implemented in validator")

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        raise NotImplementedError("update_metrics function not implemented in validator")

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
