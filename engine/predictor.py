import threading
from pathlib import Path

import torch
from data.dataset.data_utils import load_pose_inference_source
from utils import get_override_cfg


class BasePredictor:
    """
    BasePredictor.

    A base class for creating predictors.

    Attributes:
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
    """

    def __init__(self, cfg=None, save_dir=None, verbose=False, _callbacks=None, **kwargs):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
        """
        self.cfg = get_override_cfg(cfg, **kwargs)

        self.save_dir = save_dir or self.get_save_dir()
        self.done_warmup = False
        self.verbose = verbose

        # Usable if setup is done
        self.model = None
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.results = None
        self.callbacks = _callbacks
        self._lock = threading.Lock()

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def get_save_dir(self):
        output_dir = Path(self.cfg.SOLVERS.OUTPUT_DIR)
        return output_dir / self.cfg.SOLVERS.LOGS_NAME / self.cfg.SOLVERS.RESULTS_DIR

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        raise NotImplementedError("preprocess function not implemented in validator")

    def process(self, model, batch):
        raise NotImplementedError("process function not implemented in validator")

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        raise NotImplementedError("postprocess function not implemented in validator")

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        raise NotImplementedError("inference function not implemented in validator")

    def setup_source(self, source):
        """Sets up source and inference mode."""
        image, depth_image, seg_mask, obj, camera_k = source
        self.dataset = load_pose_inference_source(
            image=image,
            depth_image=depth_image,
            mask=seg_mask,
            obj=obj,
            camera_k=camera_k,
            cfg=self.cfg,
        )

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        raise NotImplementedError("stream_inference function not implemented in trainer")

    def setup_model(self, model, verbose=True):
        raise NotImplementedError("setup_model function not implemented in trainer")

    def write_results(self, i, p, im, s):
        """Write inference results to a file or directory."""
        raise NotImplementedError("write_results function not implemented in trainer")

    def save_predicted_images(self, save_path="", frame=0):
        """Save predicted images."""
        raise NotImplementedError("save_predicted_images function not implemented in trainer")

    def show(self, p=""):
        """Display an image in a window using the OpenCV imshow function."""
        raise NotImplementedError("show function not implemented in trainer")

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """Add callback."""
        self.callbacks[event].append(func)
