import json

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from engine import BaseValidator
from nn.autobackend import AutoBackend
from utils import colorstr
from utils.metrics import PoseMetrics
from utils.ops import Profile
from utils.torch_utils import (
    select_device,
    de_parallel,
)
from utils.torch_utils import smart_inference_mode


class PoseValidator(BaseValidator):
    def __init__(self, dataset=None, dataloader=None, save_dir=None, pbar=None, cfg=None, _callbacks=None):
        super().__init__(dataset, dataloader, save_dir, pbar, cfg, _callbacks)
        self.metrics = PoseMetrics(save_dir)

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.cfg.HALF = self.device.type != "cpu" and trainer.amp  # force FP16 val during training
            model = trainer.model
            model = model.half() if self.cfg.HALF else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.cfg.SOLVERS.PLOTS &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            automodel = AutoBackend(
                weights=model,
                device=select_device(device=self.cfg.SOLVERS.DEVICE, batch=self.cfg.TRAIN_DATA.BATCH_SIZE),
                fp16=self.cfg.HALF,
            )
            # self.model = model
            self.device = automodel.device  # update device
            self.cfg.HALF = automodel.fp16  # update half
            pt = automodel.pt

            if self.device.type in {"cpu", "mps"}:
                self.cfg.TEST_DATA.workers = 0  # faster CPU val as time dominated by inference, not dataloading

            model = automodel.model
            model.eval()
            b = 1 if pt else self.cfg.TEST_DATA.BATCH_SIZE
            input_shapes = {
                'pts': (b, self.cfg.TEST_DATA.N_SAMPLE_OBSERVED_POINT, 3),
                'rgb': (b, 3, self.cfg.TEST_DATA.IMG_SIZE, self.cfg.TEST_DATA.IMG_SIZE),
                'rgb_choose': (b, self.cfg.TEST_DATA.N_SAMPLE_OBSERVED_POINT),
                'model': (b, self.cfg.TEST_DATA.N_SAMPLE_MODEL_POINT, 3),
                'dense_po': (b, self.cfg.POSE_MODEL.FINE_NPOINT, 3),
                'dense_fo': (b, self.cfg.POSE_MODEL.FINE_NPOINT, self.cfg.POSE_MODEL.FEATURE_EXTRACTION.OUT_DIM),
            }

            automodel.warmup(input_shapes=input_shapes)  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = tqdm(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = self.process(model, batch)

            # Postprocess
            with dt[2]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            logger.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.cfg.SOLVERS.SAVE_JSON and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    logger.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.cfg.SOLVERS.PLOTS or self.cfg.SOLVERS.SAVE_JSON:
                logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

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
        objs = batch['obj'].cpu().numpy()
        cls_names = [self.dataset.class_names[obj] for obj in objs]
        all_dense_po = []
        all_dense_fo = []
        for cls_name in cls_names:
            tem, tem_pts, tem_choose = self.dataset.get_all_templates(cls_name, self.device)
            dense_po, dense_fo = model.feature_extraction.get_obj_feats(tem, tem_pts, tem_choose)
            all_dense_po.append(dense_po.squeeze(0))
            all_dense_fo.append(dense_fo.squeeze(0))

        batch_dense_po = torch.stack(all_dense_po, dim=0)  # [batch, ...]
        batch_dense_fo = torch.stack(all_dense_fo, dim=0)  # [batch, ...]
        batch['dense_po'] = batch_dense_po
        batch['dense_fo'] = batch_dense_fo
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
        self.stats = dict(te=[], re=[], add=[], score=[], fitness=[])

    def update_metrics(self, preds, batch):
        """
        使用模型预测结果和对应的ground-truth更新指标统计。
        Args:
            preds (dict): 模型预测结果, 示例:
                preds = {
                    'pred_Rs': np.array([...])  # shape (n, 9) 或 (n,3,3)
                    'pred_Ts': np.array([...])  # shape (n,3)
                    'pred_scores': np.array([...]) # shape (n,)
                }
            batch (dict): batch中包含的GT标注值, 示例:
                batch = {
                    'translation_label': Tensor, shape (n,3)
                    'rotation_label': Tensor, shape (n,3,3)
                }
        """
        pred_Rs = np.array(preds['pred_Rs'])
        pred_Ts = np.array(preds['pred_Ts'])
        pose_scores = np.array(preds['pred_scores'])

        gt_Ts = batch['translation_label'].cpu().numpy()
        gt_Rs = batch['rotation_label'].cpu().numpy()
        # 如果gt_Rs是3x3，需要展平成9维的向量，或者在compute_xxx中支持3x3输入
        if gt_Rs.ndim == 3:
            gt_Rs = gt_Rs.reshape(gt_Rs.shape[0], -1)

        if pred_Rs.ndim == 3:
            pred_Rs = pred_Rs.reshape(pred_Rs.shape[0], -1)

        points = batch['pts'].cpu().numpy()

        # 使用pose进行更新
        self.metrics.process(pred_Rs, pred_Ts, gt_Rs, gt_Ts, points, pose_scores)

    def get_stats(self):
        """
        从当前的pose评估数据中获取统计结果并更新self.stats字典，然后返回该字典。
        """
        means = self.metrics.mean_results  # [mean_te, mean_re, mean_add, mean_ps]
        fitness = self.metrics.fitness
        self.stats['te'] = means[0]
        self.stats['re'] = means[1]
        self.stats['add'] = means[2]
        self.stats['score'] = means[3]
        self.stats['fitness'] = fitness
        return self.stats

    def print_results(self):
        """
        self.metrics.result_dict包括:
        {
          'fitness': np.float32(...),
          'metrics/ADD_err': np.float32(...),
          'metrics/pred_pose_score': np.float32(...),
          'metrics/rotation_err': np.float32(...),
          'metrics/translation_err': np.float32(...)
        }
        """
        results = self.metrics.results_dict
        logger.info(
            "\n" +
            ("%20s" * len(results.keys()) + "%11s") % (
                *results.keys(),
                "BATCH",
            ) +
            "\n" +
            colorstr("blue", ("%20.4g" * len(results.keys()) + "%11.4g") % (
                *results.values(),
                self.cfg.TEST_DATA.BATCH_SIZE,
            ))
        )
