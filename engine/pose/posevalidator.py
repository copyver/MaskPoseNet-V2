from engine import BaseValidator
from utils.metrics import PoseMetrics
import torch

class PoseValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, cfg=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, cfg, _callbacks)
        self.metrics = PoseMetrics(save_dir)

    def preprocess(self, batch):
        if isinstance(batch, dict):
            return {key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value for key, value in
                    batch.items()}
        elif torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        else:
            raise TypeError(f"Unsupported type for batch: {type(batch)}")

    def process(self, model, batch):
        n_instance = batch['pts'].size(1)
        dense_po, dense_fo = model.feature_extraction.get_obj_feats(batch['all_tem'], batch['all_tem_pts'],
                                                                    batch['all_tem_choose'])
        print(dense_po.shape)






