from pathlib import Path
from utils.torch_utils import (
    select_device,
    init_seeds,
    torch_distributed_zero_first,
    RANK,
    LOCAL_RANK,
)
from utils import yaml_print, yaml_save


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, cfg, model):
        self.cfgs = cfg
        self.device = select_device(device=cfg.SOLVERS.DEVICE, batch=cfg.TRAIN_DATALOADER.BATCH_SIZE)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.SOLVERS.SEED + 1 + RANK)

        self.output_dir = Path(self.cfgs.SOLVERS.OUTPUT_DIR)
        self.logs_name = self.cfgs.SOLVERS.LOGS_NAME
        self.tensorboard_logs_dir = self.output_dir / self.logs_name / self.cfgs.SOLVERS.TENSORBOARD_LOGS_DIR
        self.checkpoint_dir = self.output_dir / self.cfgs.LOGS.NAME / self.cfgs.SOLVERS.CHECKPOINTS_DIR
        if RANK in {-1, 0}:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)  # make dir
            yaml_save(self.save_dir / "cfg.yaml", vars(self.cfgs))

        self.save_epoch_freq = self.cfgs.SOLVERS.SAVE_EPOCH_FREQ
        self.print_freq = self.cfgs.SOLVERS.PRINT_FREQ

        self.batch_size = self.TRAIN_DATALOADER.BATCH_SIZE
        self.epochs = self.SOLVERS.EPOCHS or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = self.SOLVERS.START_EPOCH or 0
        if RANK == -1:
            yaml_print(vars(self.cfgs))

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


    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        data = self._smart_load_dataset(self.cfgs.task)  ## Todo:add
        self.data = data
        return data["train"], data.get("val") or data.get("test")

    def _smart_load_dataset(self):
        pass



