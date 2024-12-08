import torch

from data.dataset.posenet_dataset import PoseNetDataset
from engine import BaseTrainer
from models.pose.loss_utils import Loss


class PoseTrainer(BaseTrainer):
    def __init__(self, cfg, model, callbacks=None):
        super(PoseTrainer, self).__init__(cfg, model, callbacks=callbacks)

    def get_dataset(self):
        train_dataset = PoseNetDataset(self.cfg.TRAIN_DATASET, is_train=True)
        val_dataset = PoseNetDataset(self.cfg.TEST_DATASET, is_train=False)
        return train_dataset, val_dataset

    def _set_up_loss(self):
        self.loss_names = "coarse0", "coarse1", "coarse2", "fine0", "fine1", "fine2"
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
