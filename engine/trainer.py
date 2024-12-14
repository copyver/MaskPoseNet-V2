import datetime
import math
import os
import time
from copy import deepcopy
from pathlib import Path
import gc
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import distributed as dist
from torch import optim
from tqdm import tqdm

from nn.tasks import attempt_load_one_weight
from utils import colorstr, RANK, LOCAL_RANK
from utils import yaml_print, yaml_save, logger_format_function
from utils.log_utils import TensorBoardWriter
from utils.torch_utils import (
    select_device,
    init_seeds,
    torch_distributed_zero_first,
    one_cycle,
    autocast,
    EarlyStopping,
    TORCH_2_4,
    convert_optimizer_state_dict_to_fp16,
)


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, cfg, model, callbacks=None):
        self.cfg = cfg
        self.callbacks = callbacks
        self.is_train = self.cfg.IS_TRAIN
        self.device = select_device(device=cfg.SOLVERS.DEVICE, batch=cfg.TRAIN_DATA.BATCH_SIZE)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.cfg.SOLVERS.SEED + 1 + RANK)

        self.output_dir = Path(self.cfg.SOLVERS.OUTPUT_DIR)
        self.logs_name = self.cfg.SOLVERS.LOGS_NAME
        self.tensorboard_logs_dir = self.output_dir / self.cfg.LOGS_NAME / self.cfg.SOLVERS.TENSORBOARD_LOGS_DIR
        self.checkpoint_dir = self.output_dir / self.cfg.LOGS_NAME / self.cfg.SOLVERS.CHECKPOINTS_DIR
        self.save_dir = self.output_dir / self.cfg.LOGS_NAME / self.cfg.SOLVERS.SAVE_DIR
        logger.add(self.tensorboard_logs_dir / "log.txt", format=logger_format_function)
        if RANK in {-1, 0}:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)  # make dir
            yaml_save(self.tensorboard_logs_dir / "cfg.yaml", dict(self.cfg))

        self.save_epoch_freq = self.cfg.SOLVERS.SAVE_EPOCH_FREQ
        self.print_freq = self.cfg.SOLVERS.PRINT_FREQ

        self.nbs = self.cfg.SOLVERS.NOMINAL_BATCH_SIZE
        self.batch_size = self.cfg.TRAIN_DATA.BATCH_SIZE
        self.epochs = self.cfg.SOLVERS.EPOCHS or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = self.cfg.SOLVERS.START_EPOCH or 0
        if RANK == -1:
            yaml_print(dict(self.cfg))

        # Model and Dataset
        self.model = model
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.tensorboard_logs_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.cfg.SOLVERS.DEVICE, str) and len(self.cfg.SOLVERS.DEVICE):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.cfg.SOLVERS.DEVICE.split(","))
        elif isinstance(self.cfg.SOLVERS.DEVICE, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.cfg.SOLVERS.DEVICE)
        elif self.cfg.SOLVERS.DEVICE in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            if self.cfg.TRAIN_DATA.BATCH_SIZE < 1.0:
                logger.warning(
                    "WARNING  'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.cfg.TRAIN_DATA.BATCH_SIZE = 16

        # Todo:run ddp training

        else:
            self._do_train(world_size)

    def _do_train(self, world_size):
        self._set_up_train(world_size)

        nb = len(self.train_loader)
        nw = max(round(self.cfg.SOLVERS.WARMUP_EPOCHS * nb),
                 100) if self.cfg.SOLVERS.WARMUP_EPOCHS > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")

        epoch = self.start_epoch

        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.scheduler.step()

            self.model.train()
            pbar = enumerate(self.train_loader)

            if RANK in {-1, 0}:
                logger.info(self.progress_string())   # Todo: add function
                pbar = tqdm(enumerate(self.train_loader), total=nb)

            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.cfg.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.cfg.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.cfg.warmup_momentum, self.cfg.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items, dict_info_step = self._step(batch)
                    # self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                # self.loss.backward()
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.cfg.SOLVERS.TIME:
                        self.stop = (time.time() - self.train_time_start) > (self.cfg.SOLVERS.TIME * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["pts"].shape[0],  # batch size, i.e. 8
                            batch["rgb"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    # if self.cfg.SOLVERS.PLOTS and ni in self.plot_idx:   # Todo: add plots
                    #     self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                # self.ema.update_attr(self.model, include=["yaml", "nc", "cfg", "names", "stride", "class_weights"])

                # Validation
                if self.cfg.SOLVERS.VAL or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.cfg.SOLVERS.TIME:
                    self.stop |= (time.time() - self.train_time_start) > (self.cfg.SOLVERS.TIME * 3600)

                # Save model
                if self.cfg.SOLVERS.SAVE or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.cfg.SOLVERS.TIME:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.cfg.epochs = math.ceil(self.cfg.SOLVERS.TIME * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            logger.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.cfg.SOLVERS.PLOTS:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                # "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.cfg),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                # "version": __version__,
                # "license": "AGPL-3.0 (https://ultralytics.com/license)",
                # "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint

    def read_results_csv(self):
        """Read results.csv into a dict using pandas."""
        import pandas as pd  # scope for faster 'import ultralytics'

        return pd.read_csv(self.csv).to_dict(orient="list")

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # header
        t = time.time() - self.train_time_start
        with open(self.csv, "a") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        raise NotImplementedError("progress_string function is not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def _get_memory(self):
        """Get accelerator memory utilization in GB."""
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cpu":
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory / 1e9

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def _set_up_train(self, world_size=1):
        """Builds dataloaders and optimizer on correct rank process."""
        # Model
        # self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        # self.set_model_attributes

        self.loss_function = self._set_up_loss()

        # Freeze Layers   Todo: add freeze layer
        freeze_list = (
            self.cfg.POSE_MODEL.FREEZE
            if isinstance(self.cfg.POSE_MODEL.FREEZE, list)
            else range(self.cfg.POSE_MODEL.FREEZE)
            if isinstance(self.cfg.POSE_MODEL.FREEZE, int)
            else []
        )
        always_freeze_names = []  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                logger.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                logger.warning(
                    f"WARNING setting 'requires_grad=True' for frozen layer '{k}'. "
                )
                v.requires_grad = True

        # AMP  Todo: add amp
        # self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        # if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
        #     callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
        #     self.amp = torch.tensor(check_amp(self.model), device=self.device)
        #     callbacks.default_callbacks = callbacks_backup  # restore callbacks
        # if RANK > -1 and world_size > 1:  # DDP
        #     dist.broadcast(self.amp,
        #                    src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        # self.amp = bool(self.amp)  # as boolean
        # self.scaler = (
        #     torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(
        #         enabled=self.amp)
        # )
        self.amp = False
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )


        # Check imgsz  Todo: add img size check

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_dataset = self.get_dataset(rank=LOCAL_RANK, is_train=True)
        self.train_loader = self.get_dataloader(batch_size=batch_size, rank=LOCAL_RANK, is_train=True)
        if RANK in {-1, 0}:
            self.test_dataset = self.get_dataset(self.cfg, is_train=False)
            self.test_loader = self.get_dataloader(batch_size=self.cfg.TEST_DATA.BATCH_SIZE, rank=-1, is_train=False)
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            # self.ema = ModelEMA(self.model)

        if RANK in {-1, 0} and self.is_train:
            time_log = "{0:%Y-%m-%d-%H-%M-%S-tensorboard/}".format(datetime.datetime.now())
            tensorboard_log_dir = self.tensorboard_logs_dir / time_log
            self.tensorboard_writer = TensorBoardWriter(tensorboard_log_dir)

        # Optimizer
        self.accumulate = max(round(self.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.cfg.OPTIMIZER.WEIGHT_DECAY * self.batch_size * self.accumulate / self.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.cfg.OPTIMIZER.TYPE,
            lr=self.cfg.OPTIMIZER.LR,
            momentum=self.cfg.OPTIMIZER.MOMENTUM,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.cfg.SOLVERS.PATIENCE), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        logger.info(f"Start training from epoch {self.start_epoch}")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.cfg.SOLVERS.PRETRAINED, (str, Path)):
            weights, _ = attempt_load_one_weight(self.cfg.PRETRAINED)
        # self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            logger.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={lr}' and 'momentum={momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.cfg.OPTIMIZER.WARMUP_BIAS_LR = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        logger.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer

    def resume_training(self, ckpt):
        """Resume  training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.cfg.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'train model={self.cfg.model}'"
        )
        logger.info(f"Resuming training {self.cfg.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            logger.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.cfg.close_mosaic):
            self._close_dataloader_mosaic()

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.cfg.LR_SCHEDULER.TYPE == 'cosine':
            self.lf = one_cycle(1, self.cfg.LR_SCHEDULER.LRF, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.cfg.LR_SCHEDULER.LRF) + self.cfg.LR_SCHEDULER.LRF  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def get_dataset(self, rank=0, is_train=True):
        raise NotImplementedError("get_dataset function not implemented in trainer")

    def get_dataloader(self, batch_size=16, rank=0, is_train=True):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def _step(self, data):
        raise NotImplementedError("_step function is not implemented in trainer")

    def _set_up_loss(self):
        raise NotImplementedError("_set_up_loss function is not implemented in trainer")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def _clear_memory(self):
        """Clear accelerator memory on different platforms."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

