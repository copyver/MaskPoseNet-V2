from utils.torch_utils import smart_inference_mode
import torch
from loguru import logger
from utils.torch_utils import (
    select_device,
    de_parallel,
)
from nn.autobackend import AutoBackend
from utils.ops import Profile
from tqdm import tqdm
import numpy as np
import json
from utils import colorstr
from pathlib import Path
import time


class BaseValidator(object):
    """
    Base class for all validators.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, cfg=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.cfg = cfg
        self.dataloader = dataloader
        self.pbar = pbar
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.plots = {}
        self.callbacks = _callbacks

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # force FP16 val during training
            self.cfg.HALF = self.device.type != "cpu" and trainer.amp
            model = trainer.model
            model = model.half() if self.cfg.HALF else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.cfg.PLOTS &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.model).endswith(".yaml"):
                logger.warning("WARNING validating an untrained model YAML will result in 0 mAP.")
            model = AutoBackend(
                weights=model or self.model,
                device=select_device(device=self.cfg.SOLVERS.DEVICE, batch=self.cfg.TRAIN_DATA.BATCH_SIZE),
                dnn=self.cfg.DNN,
                fp16=self.cfg.HALF,
            )
            # self.model = model
            self.device = model.device  # update device
            self.cfg.HALF = model.fp16  # update half
            pt = model.pt

            if self.device.type in {"cpu", "mps"}:
                self.cfg.TEST_DATA.workers = 0  # faster CPU val as time dominated by inference, not dataloading

            model.eval()
            b = 1 if pt else self.cfg.TEST_DATA.BATCH_SIZE
            input_shapes = {
                'pts': (b, self.cfg.TEST_DATA.N_SAMPLE_OBSERVED_POINT, 3),
                'rgb': (b, 3, self.cfg.TEST_DATA.IMG_SIZE, self.cfg.TEST_DATA.IMG_SIZE),
                'rgb_choose': (b, self.cfg.TEST_DATA.N_SAMPLE_OBSERVED_POINT),
                'translation_label': (1, 3),
                'rotation_label': (1, 3, 3),
                'tem1_rgb': (b, 3, self.cfg.TEST_DATA.IMG_SIZE, self.cfg.TEST_DATA.IMG_SIZE),
                'tem1_choose': (b, self.cfg.TEST_DATA.N_SAMPLE_TEMPLATE_POINT),
                'tem1_pts': (b, self.cfg.TEST_DATA.N_SAMPLE_TEMPLATE_POINT, 3),
                'tem2_rgb': (b, 3, self.cfg.TEST_DATA.IMG_SIZE, self.cfg.TEST_DATA.IMG_SIZE),
                'tem2_choose': (b, self.cfg.TEST_DATA.N_SAMPLE_TEMPLATE_POINT),
                'tem2_pts': (b, self.cfg.TEST_DATA.N_SAMPLE_TEMPLATE_POINT, 3),
            }

            model.warmup(input_shapes=input_shapes)  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = tqdm(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            logger.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.cfg.SAVE_JSON and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    logger.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.cfg.PLOTS or self.cfg.SAVE_JSON:
                logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

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