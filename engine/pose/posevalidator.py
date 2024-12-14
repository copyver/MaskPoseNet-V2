from engine import BaseValidator
from utils.metrics import PoseMetrics
import torch


class PoseValidator(BaseValidator):
    def __init__(self, dataset=None, dataloader=None, save_dir=None, pbar=None, cfg=None, _callbacks=None):
        super().__init__(dataset, dataloader, save_dir, pbar, cfg, _callbacks)
        self.metrics = PoseMetrics(save_dir)

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

    def process(self, model, batch):
        obj = batch['obj'][0].cpu().numpy()
        all_tem, all_tem_pts, all_tem_choose = self.dataset.get_all_templates(obj, self.device)
        dense_po, dense_fo = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)
        batch['dense_po'] = dense_po
        batch['dense_fo'] = dense_fo
        end_points = model(batch)

        return end_points

    def postprocess(self, preds):
        pred_Rs = []
        pred_Ts = []
        pred_scores = []
        pred_Rs.append(preds['pred_R'])
        pred_Ts.append(preds['pred_t'])
        pred_scores.append(preds['pred_pose_score'])
        pred_Rs = torch.cat(pred_Rs, dim=0).reshape(-1, 9).detach().cpu().numpy()
        pred_Ts = torch.cat(pred_Ts, dim=0).detach().cpu().numpy()
        pred_scores = torch.cat(pred_scores, dim=0).detach().cpu().numpy()
        preds = {
            'pred_Rs': pred_Rs,
            'pred_Ts': pred_Ts,
            'pred_scores': pred_scores,
        }
        return preds

    def init_metrics(self, model):
        self.names = model.names
        self.metrics.names = self.names
        self.metrics.plot = self.cfg.SOLVERS.PLOTS
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def update_metrics(self, preds, batch):
        pass







