from loguru import logger
from pathlib import Path

from data.dataloader.build_dataloader import BuildDataloader
from data.dataset.posenet_dataset import PoseNetDataset
from utils import yaml_print, yaml_save
from utils.log_utils import TensorBoardWriter
from utils.torch_utils import (
    select_device,
    init_seeds,
    torch_distributed_zero_first,
    RANK,
    LOCAL_RANK,
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
        _set_up_train()

        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()

        epoch = self.start_epoch

        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

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
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
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
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
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
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

    def _set_up_train(self, world_size=1):
        """Builds dataloaders and optimizer on correct rank process."""
        # Model
        # self.run_callbacks("on_pretrain_routine_start")
        # ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        # self.set_model_attributes

        # Freeze Layers   Todo: add freeze layer
        freeze_list = (
            self.args.freeze
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
        build_dataloader_train = BuildDataloader(self.cfg, dataset, is_train=True)
        self.train_dataloader = build_dataloader_train.get_dataloader()
        self.train_data_size = build_dataloader_train.get_dataset_size()
        self.per_gpu_iter_num_per_epoch = self.train_data_size // self.batch_size
        logger.info(f"Training with {self.train_data_size} train images.")
        logger.info(f"Training per_gpu_iter_num_per_epoch {self.per_gpu_iter_num_per_epoch}")
        if RANK in {-1, 0}:
            build_dataloader_test = BuildDataloader(self.cfg, dataset, is_train=False)
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

        logger.info(f"Start training from epoch {self.start_epoch}")


    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        raise NotImplementedError("get_dataset function is not implemented in trainer")
