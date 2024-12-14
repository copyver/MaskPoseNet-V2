import torch

from data.dataset.posenet_dataset import PoseNetDataset
from engine import BaseTrainer
from models.pose.loss_utils import Loss
from utils.torch_utils import torch_distributed_zero_first
from data.dataloader.build import build_dataloader
from copy import copy


class PoseTrainer(BaseTrainer):
    def __init__(self, cfg, model, callbacks=None):
        super(PoseTrainer, self).__init__(cfg, model, callbacks=callbacks)

    def build_dataset(self, cfg, is_train=True):
        return PoseNetDataset(cfg, is_train)

    def _set_up_loss(self):
        return Loss().cuda()

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "BATCH",
            "Size",
        )

    def _step(self, data):
        torch.cuda.synchronize()
        for key in data:
            data[key] = data[key].cuda()
        end_points = self.model(data)
        dict_info = self.loss_function(end_points)
        loss_all = dict_info['loss']
        loss_items = torch.tensor(
            [value.item() for key, value in dict_info.items() if 'coarse_loss' in key or 'fine_loss' in key],
            device=dict_info['loss'].device  # 保持与原始张量的设备一致
        )
        for key in dict_info:
            dict_info[key] = float(dict_info[key].item())

        return loss_all, loss_items, dict_info

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def get_dataset(self, rank=0, is_train=True):
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            if is_train:
                return self.build_dataset(self.cfg.TRAIN_DATA, is_train)
            else:
                return self.build_dataset(self.cfg.TEST_DATA, is_train)

    def get_dataloader(self, batch_size=16, rank=0, is_train=True):
        """Construct and return dataloader."""
        shuffle = is_train
        dataset = self.train_dataset if is_train else self.test_dataset
        workers = self.cfg.TRAIN_DATA.WORKERS if is_train else self.cfg.TEST_DATA.WORKERS
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        from engine.pose.posevalidator import PoseValidator
        self.loss_names = "coarse0", "coarse1", "coarse2", "fine0", "fine1", "fine2"
        return PoseValidator(
            self.test_dataset, self.test_loader, save_dir=self.save_dir, cfg=copy(self.cfg), _callbacks=self.callbacks
        )
