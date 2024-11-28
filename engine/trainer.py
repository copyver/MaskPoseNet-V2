from loguru import logger
from pathlib import Path

from data.dataloader.build_dataloader import BuildDataloader
from data.dataset.posenet_dataset import PoseNetDataset
from utils import yaml_print, yaml_save
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
        pass


    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        data = self._smart_load_dataset(self.cfg.task)
        self.data = data
        return data["train"], data.get("val") or data.get("test")

    def _smart_load_dataset(self, task=''):
        if task == 'segment':
            pass
        elif task == 'pose':
            dataset = PoseNetDataset(self.cfg, is_train=self.cfg.IS_TRAIN)
            build_dataloader_train = BuildDataloader(self.cfg, dataset, is_train=True)
            train_dataloader = build_dataloader_train.get_dataloader()
            self.train_data_size = build_dataloader_train.get_dataset_size()
            self.per_gpu_iter_num_per_epoch = self.data_size // self.batch_size
            logger.info(f"train_data_size:{self.train_data_size}")
            logger.info(f"per_gpu_iter_num_per_epoch:{self.per_gpu_iter_num_per_epoch}")

            build_dataloader_test = BuildDataloader(self.cfg, dataset, is_train=False)
            test_dataloader = build_dataloader_test.get_dataloader()
            self.test_data_size = build_dataloader_test.get_dataset_size()
            logger.info(f"test_data_size:{self.test_data_size}")
            return {
                "train": train_dataloader,
                "val": test_dataloader,
            }









