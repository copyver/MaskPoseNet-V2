import torch
from loguru import logger

from data.dataset.data_utils import load_pose_inference_source, get_all_templates
from engine import BasePredictor
from nn.autobackend import AutoBackend
from utils import colorstr
from utils.ops import Profile
from utils.torch_utils import (
    select_device,
)
from utils.torch_utils import smart_inference_mode
from utils.visualize import visualize_pose_bbox


class PosePredictor(BasePredictor):
    def __init__(self, cfg=None, save_dir=None, verbose=False, _callbacks=None):
        super(PosePredictor, self).__init__(cfg, save_dir, verbose, _callbacks)

    def setup_source(self, source):
        """Sets up source and inference mode."""
        image, depth_image, seg_mask, obj, camera_k, class_names = (
            source["image"],
            source["depth_image"],
            source["seg_mask"],
            source["obj"],
            source["camera_k"],
            source["class_names"],
        )
        self.dataset, image, depth_image, mask, whole_model_points = load_pose_inference_source(
            image=image,
            depth_image=depth_image,
            mask=seg_mask,
            obj=obj,
            camera_k=camera_k,
            cfg=self.cfg.TEST_DATA,
            class_names=class_names,
        )
        self.class_names = class_names
        source['image'] = image
        source['depth_image'] = depth_image
        source['seg_mask'] = mask
        source['whole_model_points'] = whole_model_points

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source)

            # Check if save_dir/ label file exists
            if self.save_dir is not None:
                self.save_dir.mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                b = 1 if self.model.pt else self.cfg.TEST_DATA.BATCH_SIZE
                input_shapes = {
                    'pts': (b, self.cfg.TEST_DATA.N_SAMPLE_OBSERVED_POINT, 3),
                    'rgb': (b, 3, self.cfg.TEST_DATA.IMG_SIZE, self.cfg.TEST_DATA.IMG_SIZE),
                    'rgb_choose': (b, self.cfg.TEST_DATA.N_SAMPLE_OBSERVED_POINT),
                    'model': (b, self.cfg.TEST_DATA.N_SAMPLE_MODEL_POINT, 3),
                    'dense_po': (b, self.cfg.POSE_MODEL.FINE_NPOINT, 3),
                    'dense_fo': (b, self.cfg.POSE_MODEL.FINE_NPOINT, self.cfg.POSE_MODEL.FEATURE_EXTRACTION.OUT_DIM),
                }
                self.model.warmup(input_shapes=input_shapes)
                self.done_warmup = True

            profilers = (
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            batch = self.dataset
            self.run_callbacks("on_predict_batch_start")

            # Preprocess
            with profilers[0]:
                batch = self.preprocess(batch)

            # Inference
            with profilers[1]:
                preds = self.inference(batch)

            # Postprocess
            with profilers[2]:
                self.results = self.postprocess(preds)
                self.results.update(source)

            self.run_callbacks("on_predict_postprocess_end")
            # Visualize, save, write results

            self.seen = batch['pts'].size(0)
            self.results.update({"speed": {
                "preprocess": profilers[0].dt * 1e3 / self.seen,
                "inference": profilers[1].dt * 1e3 / self.seen,
                "postprocess": profilers[2].dt * 1e3 / self.seen,
            }})

            # Print batch results
            if self.verbose:
                logger.info("\n")
            yield from self.results.items()
            self.run_callbacks("on_predict_batch_end")

        # Print final results
        if self.verbose and self.seen:
            all_t = sum(x.t * 1e3 for x in profilers)
            logger.info(f"Total {self.seen} inst, Total resume {all_t:.3f}ms")
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            logger.info(
                f"Speed: %.3fms preprocess, %.3fms inference, %.3fms postprocess per inst at shape " % t
            )
        if self.save_dir is not None:
            visualize_pose_bbox(self.results, self.save_dir)
            logger.info(f"Results saved to {colorstr('red', self.save_dir)}")
        self.run_callbacks("on_predict_end")

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

    def inference(self, batch):
        objs = batch['obj'].cpu().numpy()
        cls_names = [self.class_names[obj] for obj in objs]
        all_dense_po = []
        all_dense_fo = []

        for cls_name in cls_names:
            all_tem, all_tem_pts, all_tem_choose = get_all_templates(self.cfg.TEST_DATA, cls_name, self.device)

            # 调用特征提取函数
            dense_po, dense_fo = self.model.model.feature_extraction.get_obj_feats(
                all_tem, all_tem_pts, all_tem_choose
            )
            all_dense_po.append(dense_po.squeeze(0))
            all_dense_fo.append(dense_fo.squeeze(0))

        batch_dense_po = torch.stack(all_dense_po, dim=0)  # [batch, ...]
        batch_dense_fo = torch.stack(all_dense_fo, dim=0)  # [batch, ...]

        batch['dense_po'] = batch_dense_po
        batch['dense_fo'] = batch_dense_fo

        end_points = self.model.model(batch)

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

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        automodel = AutoBackend(
            weights=model,
            device=select_device(device=self.cfg.SOLVERS.DEVICE, batch=self.cfg.TRAIN_DATA.BATCH_SIZE),
            fp16=False,
        )

        self.device = automodel.device  # update device
        self.cfg.HALF = automodel.fp16  # update half
        self.model = automodel
        self.model.model.eval()
