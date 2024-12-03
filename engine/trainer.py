import datetime
import math
import time
from pathlib import Path
import subprocess
import numpy as np
import torch
from loguru import logger
from torch import distributed as dist
from torch import optim
from tqdm import tqdm
import os
from data.dataloader.build_dataloader import BuildDataloader
from utils import yaml_print, yaml_save, colorstr
from utils.log_utils import TensorBoardWriter
from utils.torch_utils import (
    select_device,
    init_seeds,
    torch_distributed_zero_first,
    RANK,
    LOCAL_RANK,
    one_cycle,
    autocast,
    EarlyStopping,
)


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.is_train = self.cfg.IS_TRAIN
        self.device = select_device(device=cfg.SOLVERS.DEVICE, batch=cfg.TRAIN_DATALOADER.BATCH_SIZE)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.SOLVERS.SEED + 1 + RANK)

        self.output_dir = Path(self.cfg.SOLVERS.OUTPUT_DIR)
        self.logs_name = self.cfg.SOLVERS.LOGS_NAME
        self.tensorboard_logs_dir = self.output_dir / self.LOGS_NAME / self.cfg.SOLVERS.TENSORBOARD_LOGS_DIR
        self.checkpoint_dir = self.output_dir / self.cfg.LOGS.NAME / self.cfg.SOLVERS.CHECKPOINTS_DIR
        logger.add(self.tensorboard_logs_dir / "log.txt", format="{time} {level} {message}")
        if RANK in {-1, 0}:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)  # make dir
            yaml_save(self.save_dir / "cfg.yaml", vars(self.cfg))

        self.save_epoch_freq = self.cfg.SOLVERS.SAVE_EPOCH_FREQ
        self.print_freq = self.cfg.SOLVERS.PRINT_FREQ

        self.batch_size = self.TRAIN_DATALOADER.BATCH_SIZE
        self.epochs = self.SOLVERS.EPOCHS or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = self.SOLVERS.START_EPOCH or 0
        if RANK == -1:
            yaml_print(vars(self.cfg))

        # Model and Dataset
        self.model = model
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.trainset, self.testset = self.get_dataset()
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
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.cfg.DEVICE, str) and len(self.cfg.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.cfg.DEVICE.split(","))
        elif isinstance(self.cfg.DEVICE, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.cfg.DEVICE)
        elif self.cfg.DEVICE in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            if self.cfg.TRAIN_DATALOADER.BATCH_SIZE < 1.0:
                logger.warning(
                    "WARNING  'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.cfg.TRAIN_DATALOADER.BATCH_SIZE = 16

        # Todo:run ddp training

        else:
            self._do_train(world_size)

    def _do_train(self, world_size):
        self._set_up_train()

        nb = len(self.train_dataloader)  # number of batches
        nw = max(round(self.SOLVERS.WARMUP_EPOCHS * nb),
                 100) if self.SOLVERS.WARMUP_EPOCHS > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()

        epoch = self.start_epoch

        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.scheduler.step()

            self.model.train()
            pbar = enumerate(self.train_dataloader)

            if RANK in {-1, 0}:
                logger.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_dataloader), total=nb)
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
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.cfg.time:
                        self.stop = (time.time() - self.train_time_start) > (self.cfg.time * 3600)
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
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.cfg.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "cfg", "names", "stride", "class_weights"])

                # Validation
                if self.cfg.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.cfg.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.cfg.time * 3600)

                # Save model
                if self.cfg.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.cfg.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.cfg.epochs = math.ceil(self.cfg.time * 3600 / mean_epoch_time)
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
            if self.cfg.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

    def _set_up_train(self, world_size=1):
        """Builds dataloaders and optimizer on correct rank process."""
        # Model
        # self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        # self.set_model_attributes

        # Freeze Layers   Todo: add freeze layer
        freeze_list = (
            self.cfg.freeze
            if isinstance(self.cfg.freeze, list)
            else range(self.cfg.freeze)
            if isinstance(self.cfg.freeze, int)
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

        # Check imgsz  Todo: add img size check

        # Dataloaders
        build_dataloader_train = BuildDataloader(self.cfg, self.trainset, is_train=True)
        self.train_dataloader = build_dataloader_train.get_dataloader()
        self.train_data_size = build_dataloader_train.get_dataset_size()
        self.per_gpu_iter_num_per_epoch = self.train_data_size // self.batch_size
        logger.info(f"Training with {self.train_data_size} train images.")
        logger.info(f"Training per_gpu_iter_num_per_epoch {self.per_gpu_iter_num_per_epoch}")
        if RANK in {-1, 0}:
            build_dataloader_test = BuildDataloader(self.cfg, self.testset, is_train=False)
            self.test_dataloader = build_dataloader_test.get_dataloader()
            self.test_data_size = build_dataloader_test.get_dataset_size()
            self.test_per_gpu_iter_num_per_epoch = self.test_data_size // self.batch_size
            logger.info(f"Training with {self.test_data_size} test images.")
            logger.info(f"Training test_per_gpu_iter_num_per_epoch {self.test_per_gpu_iter_num_per_epoch}")

        self.total_steps = (self.start_epoch - 1) * self.per_gpu_iter_num_per_epoch
        self.train_data_iter = iter(self.train_dataloader)

        if RANK in {-1, 0} and self.is_train:
            time_log = "{0:%Y-%m-%d-%H-%M-%S-tensorboard/}".format(datetime.datetime.now())
            tensorboard_log_dir = self.tensorboard_logs_dir / time_log
            self.tensorboard_writer = TensorBoardWriter(tensorboard_log_dir)

        # Optimizer
        self.accumulate = max(round(self.cfg.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.cfg.weight_decay * self.batch_size * self.accumulate / self.cfg.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.cfg.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.cfg.optimizer,
            lr=self.cfg.lr0,
            momentum=self.cfg.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.cfg.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        logger.info(f"Start training from epoch {self.start_epoch}")

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
        if self.cfg.cos_lr:
            self.lf = one_cycle(1, self.cfg.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.cfg.lrf) + self.cfg.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        raise NotImplementedError("get_dataset function is not implemented in trainer")
