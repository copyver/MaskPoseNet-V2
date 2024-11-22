from torch.utils.data import BatchSampler
from torch.utils.data.dataloader import DataLoader

from data.dataloader import comm
from data.dataloader.distributed_sampler import TrainingSampler, InferenceSampler


class BuildDataloader:
    def __init__(self, cfg, dataset, is_train=True, collate_fn=None):
        if is_train:
            # 训练 train
            self.dataset_size = len(dataset)
            # 多卡
            if len(cfg.SOLVERS.GPU_IDS) > 1 and cfg.SOLVERS.USE_DDP:
                batch_size_per_gpu = cfg.TRAIN_DATALOADER.BATCH_SIZE // len(cfg.SOLVERS.GPU_IDS)
                sampler = TrainingSampler(len(dataset), shuffle=cfg.TRAIN_DATALOADER.SHUFFLE)
                batch_sampler = BatchSampler(sampler, batch_size_per_gpu, drop_last=cfg.TRAIN_DATALOADER.DROP_LAST)
                self.dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=cfg.TRAIN_DATALOADER.NUM_WORKERS,
                    worker_init_fn=comm.worker_init_reset_seed,
                    collate_fn=collate_fn
                )
            else:
                batch_size_per_gpu = cfg.TRAIN_DATALOADER.BATCH_SIZE
                sampler = TrainingSampler(len(dataset), shuffle=cfg.TRAIN_DATALOADER.SHUFFLE)
                batch_sampler = BatchSampler(sampler, batch_size_per_gpu, drop_last=cfg.TRAIN_DATALOADER.DROP_LAST)
                self.dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=cfg.TRAIN_DATALOADER.NUM_WORKERS,
                    worker_init_fn=comm.worker_init_reset_seed,
                    collate_fn=collate_fn
                )
        else:
            self.dataset_size = len(dataset)
            # 多卡
            if len(cfg.SOLVERS.GPU_IDS) > 1 and cfg.SOLVERS.USE_DDP:
                batch_size_per_gpu = cfg.TEST_DATALOADER.BATCH_SIZE
                sampler = InferenceSampler(len(dataset))
                batch_sampler = BatchSampler(sampler, batch_size_per_gpu, drop_last=cfg.TEST_DATALOADER.DROP_LAST)
                self.dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=cfg.TEST_DATALOADER.NUM_WORKERS,  # *len(cfg.SOLVERS.GPU_IDS),
                    worker_init_fn=comm.worker_init_reset_seed,
                    collate_fn=collate_fn
                )
            else:
                # 单卡
                self.dataloader = DataLoader(
                    dataset,
                    batch_size=cfg.TEST_DATALOADER.BATCH_SIZE,
                    shuffle=cfg.TEST_DATALOADER.SHUFFLE,
                    num_workers=cfg.TEST_DATALOADER.NUM_NUM_WORKERS,
                    worker_init_fn=comm.worker_init_reset_seed,
                    collate_fn=collate_fn,
                    drop_last=cfg.TEST_DATALOADER.DROP_LAST
                )

    def get_dataloader(self):
        return self.dataloader

    def get_dataset_size(self):
        return self.dataset_size
