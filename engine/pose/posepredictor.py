from pathlib import Path

import cv2
import torch
from loguru import logger

from engine import BasePredictor
from utils import colorstr
from utils.ops import Profile
from utils.torch_utils import smart_inference_mode
from nn.autobackend import AutoBackend
from utils.torch_utils import (
    select_device,
)


class PosePredictor(BasePredictor):
    def __init__(self, cfg=None, save_dir=None, _callbacks=None):
        super(PosePredictor, self).__init__(cfg, save_dir, _callbacks)

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.verbose:
            logger.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source)

            # Check if save_dir/ label file exists
            if self.cfg.SAVE:
                self.save_dir.mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                # Todo: shape
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            profilers = (
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im, s)

                # Print batch results
                if self.args.verbose:
                    logger.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            logger.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            logger.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def preprocess(self, batch):
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, list):
                    batch[key] = [item.to(self.device, non_blocking=True) if torch.is_tensor(item) else item for item in
                                  value]
                elif torch.is_tensor(value):
                    batch[key] = value.to(self.device, non_blocking=True)
            return batch
        elif torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        else:
            raise TypeError(f"Unsupported type for batch: {type(batch)}")

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        automodel = AutoBackend(
            weights=model,
            device=select_device(device=self.cfg.SOLVERS.DEVICE, batch=self.cfg.TRAIN_DATA.BATCH_SIZE),
            fp16=self.cfg.HALF,
        )

        self.device = automodel.device  # update device
        self.cfg.HALF = automodel.fp16  # update half
        self.model = automodel.model
        self.model.eval()