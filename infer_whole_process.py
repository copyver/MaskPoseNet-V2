import contextlib
import os
import random
import re
import time
from pathlib import Path
from typing import Dict
from typing import List, Union

import cv2
import numpy as np
import onnxruntime as ort
import torch
import trimesh
import yaml
from scipy.spatial import cKDTree
from torchvision import transforms

from utils.visualize import visualize_point_cloud

DefaultArgs = {
    "MODEL_DIR": "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/models",
    "TEMPLATE_DIR": "/home/yhlever/DeepLearning/6D_object_pose_estimation/datasets/indus/templates",
    "IMG_SIZE": 224,
    "N_SAMPLE_OBSERVED_POINT": 2048,
    "N_SAMPLE_TEMPLATE_POINT": 5000,
    "N_SAMPLE_MODEL_POINT": 1024,
    "MINIMUM_N_POINT": 8,
    "RGB_MASK_FLAG": True,
    "SEG_FILTER_SCORE": 0.25,
    "N_TEMPLATE_VIEW": 42,
    "DEPTH_SCALE": 1.0,  # if real 1.0, syn 5.0
}
DefaultCameraK = np.array([
    1062.67, 0, 646.17,
    0, 1062.67, 474.24,
    0, 0, 1.00
], dtype=np.float64)


class Profile(contextlib.ContextDecorator):
    def __init__(self, t=0.0, device: torch.device = None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
            device (torch.device): Devices used for model inference. Defaults to None (cpu).
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """Get current time."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


def init_seeds(seed=0):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe


def compute_add_error(pc1, pc2):
    """
    计算两个点云的ADD误差（平均最近邻距离）。
    """
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    # 计算pc1中每个点到pc2的最近距离
    dist1, _ = tree2.query(pc1)
    # 计算pc2中每个点到pc1的最近距离
    dist2, _ = tree1.query(pc2)
    return (np.mean(dist1) + np.mean(dist2)) / 2 * 1000  # mm


class Colors:
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68", "FF6FDD", "FF444F",
            "CCED00", "00F344", "BD00FF", "00B4FF", "DD00BA", "00FFFF", "26C000",
            "01FFB3", "7D24FF", "7B0068", "FF1B6C", "FC6D2F", "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
                [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
                [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
                [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
                [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def print_preds_table(preds):
    """
    打印 preds 字典中的数据
    """
    # 定义表格列宽
    col_width = 10
    header_fmt = "{:<{w}} {:<{w}} {:<{w}} {:<{w}}".format
    row_fmt = "{:<{w}} {:<{w}.2f} {:<{w}.2f} {:<{w}.2f}".format

    # 打印表头
    header = header_fmt("cls_name", "seg_score", "pose_score", "add", w=col_width)
    print(header)

    # 遍历并打印每一行数据
    for cls_name, seg_score, pose_score, add in zip(preds['cls_names'], preds['seg_scores'], preds['pose_scores'],
                                                    preds['adds']):
        print(row_fmt(cls_name, seg_score, pose_score, add, w=col_width))

    # 计算平均值
    avg_seg_score = np.mean(preds['seg_scores'])
    avg_pose_score = np.mean(preds['pose_scores'])
    avg_add = np.mean(preds['adds'])
    print(row_fmt("all", avg_seg_score, avg_pose_score, avg_add, w=col_width))

    print("-" * len(header))
    print(f"Segmentation inference time: {preds['seg_inference']:.4f} ms, "
          f"Pose inference time: {preds['pose_inference']:.4f} ms")


class ModelSeg:
    """Segmentation model."""

    def __init__(self, onnxsession):
        """
        Initialization.
        """
        # Build Ort session
        self.session = onnxsession

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        # Get model width and height(YOLOv8-seg only has one input)
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

        # Load COCO class names
        self.classes = \
            yaml_load("/home/yhlever/DeepLearning/ultralytics/ultralytics/cfg/datasets/indus-seg-real.yaml")["names"]

        # Create color palette
        self.color_palette = Colors()

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )
        return boxes, segments, masks

    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """
        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose dim 1: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # Decode and return
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []

    @staticmethod
    def masks2segments(masks):
        """
        Takes a list of masks(n,h,w) and returns a list of segments(n,xy), from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        """
        Takes a mask and a bounding box, and returns a mask that is cropped to the bounding box, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
        quality but is slower, from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def generate_masks(self, im_shape, segments):
        """
        Generate individual mask images based on the segments.

        Args:
            im_shape (tuple): Shape of the original image (height, width, channels).
            segments (list): List of segment masks (each segment is a list of points).

        Returns:
            list: A list of binary masks, each corresponding to one segment.
        """
        masks = []

        for segment in segments:
            # Create an empty mask with the same height and width as the original image
            mask = np.zeros((im_shape[0], im_shape[1]), dtype=np.uint8)
            # Convert segment points to a numpy array of integers
            segment_points = np.int32([segment])
            # Fill the polygon region in the mask
            cv2.fillPoly(mask, segment_points, 255)
            # Append the individual mask to the list
            masks.append(mask)

        return masks

    def draw_and_visualize(self, im, bboxes, segments, vis=False, save=True):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """
        # Draw rectangles and polygons
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # draw contour and fill mask
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
            cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))

            # draw bbox rectangle
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.color_palette(int(cls_), bgr=True),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                im,
                f"{self.classes[cls_]}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_palette(int(cls_), bgr=True),
                2,
                cv2.LINE_AA,
            )

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # Show image
        if vis:
            cv2.imshow("demo", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image
        if save:
            cv2.imwrite("demo.jpg", im)


class ModelPose:
    """PoseEstimation Model"""

    def __init__(self, pt_model: str, args: Dict = None, device: Union[str, int, List[int,],] = "cuda"):
        """
        Initialization.
        Args:
            pt_model (str): Path to the pt model.
        """
        self.weights = pt_model
        self.device = device
        self.args = args
        self.model = None
        self.class_names = None
        self._load()

    def __call__(self, source: Dict):
        with torch.no_grad():
            batch, whole_target_pts, whole_model_points = self.load_pose_inference_source(
                image=source["image"],
                depth_image=source["depth_image"],
                mask=source["seg_mask"],
                obj=source["cls_ids"],
                camera_k=source["camera_k"],
                seg_scores=source["seg_scores"],
            )

            batch = self.preprocess(batch)
            end_points = self.inference(batch)
            preds = self.postprocess(end_points)
            preds['cls_names'] = end_points["cls_names"]
            preds['cls_ids'] = end_points['cls_ids']
            preds["whole_src_points"] = whole_target_pts
            preds["whole_model_points"] = whole_model_points
        return preds

    def _load(self):
        if Path(self.weights).suffix == ".pt":
            self.model, _ = self._attempt_load_one_weight()
        else:
            raise ValueError(f"Unsupported weights file '{self.weights}'")

    def _attempt_load_one_weight(self):
        """Loads a single model weights."""
        ckpt = torch.load(self.weights, map_location="cpu")

        if "model" not in ckpt:
            raise ValueError(f"'model' key missing in checkpoint {self.weights}")
        if "class_names" not in ckpt:
            raise ValueError(f"'class_names' key missing in checkpoint {self.weights}")
        self.class_names = ckpt["class_names"]

        # 根据 'model' 是 state_dict 还是完整 nn.Module，做区分
        model_data = ckpt["model"]

        if isinstance(model_data, dict):
            # 只保存了 model.state_dict()
            from models import PoseModel
            from easydict import EasyDict
            if PoseModel is None:
                raise ValueError(
                    "Checkpoint contains only a state_dict, but no model_builder provided!\n"
                    "Please provide a callable model_builder() that returns an uninitialized model."
                )
            # 创建模型实例 load_state_dict
            model_cfg = EasyDict(ckpt["train_args"]['POSE_MODEL'])
            model_cfg.FEATURE_EXTRACTION.PRETRAINED = False  # inference not need to load pretrained mae weights
            model = PoseModel(model_cfg)
            model.load_state_dict(model_data)
            model.to(self.device).float()
        else:
            # nn.Module 对象序列化
            model = model_data.to(self.device).float()

        model.pt_path = self.weights  # attach *.pt file path to model

        model = model.eval()

        # Return model and ckpt
        return model, ckpt

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
            all_tem, all_tem_pts, all_tem_choose = self.get_all_templates(cls_name)

            # 调用特征提取函数
            dense_po, dense_fo = self.model.feature_extraction.get_obj_feats(
                all_tem, all_tem_pts, all_tem_choose
            )
            all_dense_po.append(dense_po.squeeze(0))
            all_dense_fo.append(dense_fo.squeeze(0))

        batch_dense_po = torch.stack(all_dense_po, dim=0)  # [batch, ...]
        batch_dense_fo = torch.stack(all_dense_fo, dim=0)  # [batch, ...]

        batch['dense_po'] = batch_dense_po
        batch['dense_fo'] = batch_dense_fo

        end_points = self.model(batch)
        end_points['cls_names'] = cls_names
        end_points['cls_ids'] = objs

        return end_points

    def postprocess(self, preds):
        pred_Rs = []
        pred_Ts = []
        pose_scores = []
        pred_Rs.append(preds['pred_R'])
        pred_Ts.append(preds['pred_t'])
        pose_scores.append(preds['pred_pose_score'])
        pred_Rs = torch.cat(pred_Rs, dim=0).reshape(-1, 9).detach().cpu().numpy()
        pred_Ts = torch.cat(pred_Ts, dim=0).detach().cpu().numpy()
        pose_scores = torch.cat(pose_scores, dim=0).detach().cpu().numpy()
        preds = {
            'pred_Rs': pred_Rs,
            'pred_Ts': pred_Ts,
            'pose_scores': pose_scores,
        }
        return preds

    @staticmethod
    def get_models(file_base, cls_name, n_sample):
        mesh_path = os.path.join(file_base, cls_name + '.obj')
        mesh = trimesh.load(mesh_path, force='mesh')
        model_pts, _ = trimesh.sample.sample_surface(mesh, n_sample)
        return model_pts.astype(np.float32)

    @staticmethod
    def get_bbox(label):
        img_width, img_length = label.shape
        rows = np.any(label, axis=1)
        cols = np.any(label, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1
        r_b = rmax - rmin
        c_b = cmax - cmin
        b = min(max(r_b, c_b), min(img_width, img_length))
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]

        rmin = center[0] - int(b / 2)
        rmax = center[0] + int(b / 2)
        cmin = center[1] - int(b / 2)
        cmax = center[1] + int(b / 2)

        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > img_width:
            delt = rmax - img_width
            rmax = img_width
            rmin -= delt
        if cmax > img_length:
            delt = cmax - img_length
            cmax = img_length
            cmin -= delt
        return [rmin, rmax, cmin, cmax]

    def get_point_cloud_from_depth(self, depth, k, bbox=None):
        cam_fx, cam_fy, cam_cx, cam_cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

        im_h, im_w = depth.shape
        xmap = np.array([[i for i in range(im_w)] for _ in range(im_h)])
        ymap = np.array([[j for _ in range(im_w)] for j in range(im_h)])

        if bbox is not None:
            rmin, rmax, cmin, cmax = bbox
            depth = depth[rmin:rmax, cmin:cmax].astype(np.float32)
            xmap = xmap[rmin:rmax, cmin:cmax].astype(np.float32)
            ymap = ymap[rmin:rmax, cmin:cmax].astype(np.float32)

        pt2 = depth.astype(np.float32)
        pt0 = (xmap.astype(np.float32) - cam_cx) * pt2 / cam_fx
        pt1 = (ymap.astype(np.float32) - cam_cy) * pt2 / cam_fy

        cloud = np.stack([pt0, pt1, pt2]).transpose((1, 2, 0))
        return cloud

    def get_resize_rgb_choose(self, choose, bbox, img_size):
        rmin, rmax, cmin, cmax = bbox
        crop_h = rmax - rmin
        ratio_h = img_size / crop_h
        crop_w = cmax - cmin
        ratio_w = img_size / crop_w

        row_idx = choose // crop_w
        col_idx = choose % crop_w
        choose = (np.floor(row_idx * ratio_h) * img_size + np.floor(col_idx * ratio_w)).astype(np.int64)
        return choose

    def load_pose_inference_source(self, image, depth_image, mask, obj, camera_k, seg_scores):
        """
        Loads data sources for pose estimation, normalizes the input image, depth map, and mask,
        and generates a dictionary containing point clouds, RGB data, model points, and other relevant information.
        Supports batch processing for multiple objects.
        """
        # 确保obj和mask是列表，以便统一处理多对象场景
        if not isinstance(obj, (list, tuple)):
            obj = [obj]
        if not isinstance(mask, (list, tuple)):
            mask = [mask]

        if len(mask) != len(obj):
            raise ValueError("mask和obj数量不匹配")

        # 加载/检查image
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Image is not a valid numpy array")

        # 加载/检查depth_image
        if depth_image is None or not isinstance(depth_image, np.ndarray):
            raise ValueError("Depth image is not a valid numpy array")
        if depth_image.ndim == 3 and depth_image.shape[2] == 3:
            # Convert the 3-channel image to a single channel
            depth_image = depth_image[:, :, 0]

        # 相机内参处理
        camera_k = np.array(camera_k, dtype=np.float32).reshape(3, 3)

        # 模型点加载
        model_dir = self.args["MODEL_DIR"]

        # 将RGB从BGR转为RGB备用
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 深度归一化为米
        depth = depth_image.astype(np.float32) * self.args["DEPTH_SCALE"] / 1000.0 + 0.085

        whole_pts = self.get_point_cloud_from_depth(depth, camera_k)

        all_pts = []
        all_rgb = []
        all_rgb_choose = []
        all_model_points = []
        all_obj = []
        whole_model_points = []

        for m, o, s in zip(mask, obj, seg_scores):
            # if s < 0.1:
            #     continue
            # 加载/检查mask
            if m is None or not isinstance(m, np.ndarray):
                raise ValueError("Mask is not a valid numpy array")
            m = m.astype(np.uint8)

            cls_name = self.class_names[o]
            model_points = self.get_models(model_dir, cls_name, self.args["N_SAMPLE_MODEL_POINT"])
            if model_points is None:
                raise ValueError("Model points not found for class {}".format(cls_name))
            radius = np.max(np.linalg.norm(model_points, axis=1))

            # 获取mask的bbox
            m = np.logical_and(m > 0, depth > 0)
            bbox = self.get_bbox(m > 0)
            y1, y2, x1, x2 = bbox
            sub_mask = m[y1:y2, x1:x2]
            choose = sub_mask.astype(np.float32).flatten().nonzero()[0]

            # 获取点云
            pts = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
            center = np.mean(pts, axis=0)
            tmp_cloud = pts - center[None, :]
            flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
            if np.sum(flag) < 4:
                continue
            choose = choose[flag]
            pts = pts[flag]

            # 采样点
            if len(choose) < self.args["N_SAMPLE_OBSERVED_POINT"]:
                choose_idx = np.random.choice(np.arange(len(choose)), self.args["N_SAMPLE_OBSERVED_POINT"],
                                              replace=True)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.args["N_SAMPLE_OBSERVED_POINT"],
                                              replace=False)

            choose = choose[choose_idx]
            pts = pts[choose_idx]

            # 裁剪并处理RGB
            sub_rgb = rgb_img[y1:y2, x1:x2, :]
            if self.args["RGB_MASK_FLAG"]:
                sub_rgb = sub_rgb * (sub_mask[:, :, None] > 0).astype(np.uint8)
            sub_rgb = cv2.resize(sub_rgb, (self.args["IMG_SIZE"], self.args["IMG_SIZE"]),
                                 interpolation=cv2.INTER_LINEAR)
            sub_rgb = self.transform(sub_rgb)  # [C,H,W]

            # 获取resize后对应的rgb_choose
            rgb_choose = self.get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.args["IMG_SIZE"])

            all_pts.append(torch.FloatTensor(pts))
            all_rgb.append(torch.FloatTensor(sub_rgb))
            all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
            all_model_points.append(torch.FloatTensor(model_points))
            all_obj.append(torch.IntTensor([o]).long())
            whole_model_points.append(np.float64(model_points))

        if len(all_pts) == 0:
            return None

        ret_dict = {
            'pts': torch.stack(all_pts, dim=0),
            'rgb': torch.stack(all_rgb, dim=0),
            'rgb_choose': torch.stack(all_rgb_choose, dim=0),
            'model': torch.stack(all_model_points, dim=0),
            'obj': torch.cat(all_obj, dim=0).long()  # obj本身是[1]的tensor，这里cat后是[batch]
        }
        whole_target_pts = np.stack(all_pts, axis=0)
        whole_model_points = np.stack(whole_model_points, axis=0)
        return ret_dict, whole_target_pts, whole_model_points

    @staticmethod
    def transform(image):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])(image)

    @staticmethod
    def convert_blender_to_pyrender(blender_nocs):
        # 交换 Y 和 Z 轴
        pyrender_nocs = blender_nocs[:, :, [0, 2, 1]]
        # 反转 Y 轴
        pyrender_nocs[:, :, 2] *= -1
        return pyrender_nocs

    def get_template(self, cls_name, tem_index=0):
        file_base = self.args["TEMPLATE_DIR"]
        rgb_path = os.path.join(file_base, cls_name, f'rgb_{tem_index}.png')
        xyz_path = os.path.join(file_base, cls_name, f'xyz_{tem_index}.npy')
        mask_path = os.path.join(file_base, cls_name, f'mask_{tem_index}.png')

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) == 255
        bbox = self.get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[y1:y2, x1:x2, :]

        if self.args["RGB_MASK_FLAG"]:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.args["IMG_SIZE"], self.args["IMG_SIZE"]), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(rgb)

        choose = mask.astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= self.args["N_SAMPLE_TEMPLATE_POINT"]:
            choose_idx = np.random.choice(np.arange(len(choose)), self.args["N_SAMPLE_TEMPLATE_POINT"])
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.args["N_SAMPLE_TEMPLATE_POINT"], replace=False)
        choose = choose[choose_idx]

        xyz = np.load(xyz_path).astype(np.float32)
        xyz = self.convert_blender_to_pyrender(xyz)[y1:y2, x1:x2, :]
        xyz = xyz.reshape((-1, 3))[choose, :]
        choose = self.get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.args["IMG_SIZE"])

        return rgb, choose, xyz

    def get_all_templates(self, cls_name):
        n_template_view = self.args["N_TEMPLATE_VIEW"]
        all_tem_rgb = [[] for i in range(n_template_view)]
        all_tem_choose = [[] for i in range(n_template_view)]
        all_tem_pts = [[] for i in range(n_template_view)]

        for i in range(n_template_view):
            tem_rgb, tem_choose, tem_pts = self.get_template(cls_name, i)
            all_tem_rgb[i].append(torch.FloatTensor(tem_rgb))
            all_tem_choose[i].append(torch.IntTensor(tem_choose).long())
            all_tem_pts[i].append(torch.FloatTensor(tem_pts))

        for i in range(n_template_view):
            all_tem_rgb[i] = torch.stack(all_tem_rgb[i]).to(self.device, non_blocking=True)
            all_tem_choose[i] = torch.stack(all_tem_choose[i]).to(self.device, non_blocking=True)
            all_tem_pts[i] = torch.stack(all_tem_pts[i]).to(self.device, non_blocking=True)

        return all_tem_rgb, all_tem_pts, all_tem_choose


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input:
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def get_3d_bbox(scale, shift=0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                            [scale / 2, +scale / 2, -scale / 2],
                            [-scale / 2, +scale / 2, scale / 2],
                            [-scale / 2, +scale / 2, -scale / 2],
                            [+scale / 2, -scale / 2, scale / 2],
                            [+scale / 2, -scale / 2, -scale / 2],
                            [-scale / 2, -scale / 2, scale / 2],
                            [-scale / 2, -scale / 2, -scale / 2]]) + shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def draw_3d_bbox(img, imgpts, color, size=3):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars in blue color
    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img


def draw_3d_pts(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for point in imgpts:
        img = cv2.circle(img, (point[0], point[1]), size, color, -1)
    return img


def draw_detections(image, pred_rots, pred_trans, model_points, intrinsics, ids):
    num_pred_instances = len(pred_rots)
    draw_image_bbox = image.copy()
    color_palette = Colors()

    for ind in range(num_pred_instances):
        color = color_palette(ids[ind], bgr=True)
        # 3d bbox
        scale = (np.max(model_points[ind], axis=0) - np.min(model_points[ind], axis=0))
        shift = np.mean(model_points[ind], axis=0)
        bbox_3d = get_3d_bbox(scale, shift)

        # 3d point
        choose = np.random.choice(np.arange(len(model_points[ind])), 512)
        pts_3d = model_points[ind][choose].T
        # draw 3d bounding box
        transformed_bbox_3d = pred_rots[ind] @ bbox_3d + pred_trans[ind][:, np.newaxis]
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        draw_image_bbox = draw_3d_bbox(draw_image_bbox, projected_bbox, color, size=2)
        # draw point cloud
        transformed_pts_3d = pred_rots[ind] @ pts_3d + pred_trans[ind][:, np.newaxis]
        projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics)
        draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)

    return draw_image_bbox


def visualize_pose_bbox(results):
    rgb = results['image']
    pred_rot = results['pred_Rs'].reshape((-1, 3, 3))
    pred_trans = results['pred_Ts'].reshape((-1, 3))
    model_points = results['whole_model_points']
    K = results['camera_k'].reshape((3, 3))
    cls_ids = results['cls_ids']

    # draw_detections返回numpy数组格式的图像
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, cls_ids)
    img = np.uint8(img)  # 确保类型为uint8

    save_path = 'vis_pose_bbox.png'
    cv2.imwrite(save_path, img)


class Model:
    def __init__(self, pose_model_path, seg_model_path, args, device):
        """
        初始化模型，加载分割模型和姿态估计模型。

        Args:
            pose_model_path (str): 姿态模型文件路径。
            seg_model_path (str): 分割模型文件路径（ONNX格式）。
            device (str):  "cuda" 或 "cpu"。
        """
        init_seeds(42)

        onnxsession = ort.InferenceSession(
            seg_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU" and device == "cuda"
            else ["CPUExecutionProvider"],
        )

        self.seg_model = ModelSeg(onnxsession)
        self.pose_model = ModelPose(pose_model_path, args=args, device=device)

    def __call__(self, img: np.ndarray, depth_img: np.ndarray, camera_k: np.ndarray,
                 save_seg: bool = False, save_pose: bool = True, vis_pts: bool = False,
                 pose_all: bool = False):
        """
        对输入图像进行分割和姿态估计推理，并计算 ADD 误差。

        Args:
            img (np.ndarray): 输入图像
            depth_img (np.ndarray): 深度图像
            camera_k (np.ndarray): 相机内参矩阵
            save_seg (bool, optional): 是否保存分割结果
            save_pose (bool, optional): 是否保存姿态结果
            vis_pts (bool, optional): 是否可视化点云

        Returns:
            dict: 包含分割、姿态和 ADD 误差的预测结果字典。
        """
        source = {
            "image": img,
            "depth_image": depth_img,
            "camera_k": camera_k,
        }

        # Inference segmentation
        with Profile() as ps:
            boxes, segments, masks = self.seg_model(img, conf_threshold=0.4, iou_threshold=0.7)

        if len(boxes) > 0:
            seg_scores = [box[4] for box in boxes]
            cls_ids = [int(box[5]) + 1 for box in boxes]  # add background class
            mask_list = [masks[i] for i in range(masks.shape[0])]
            if not pose_all:
                seg_scores = seg_scores[:1]
                cls_ids = cls_ids[:1]
                mask_list = mask_list[:1]
            source.update({
                "seg_scores": seg_scores,
                "cls_ids": cls_ids,
                "seg_mask": mask_list,
            })
            if save_seg:
                self.seg_model.draw_and_visualize(image, boxes, segments, vis=False, save=save_seg)

        # Inference pose
        with Profile() as pp:
            preds = self.pose_model(source)

        source.pop('cls_ids')
        preds.update(source)
        preds.update({"seg_inference": ps.dt * 1000,
                      "pose_inference": pp.dt * 1000})
        if save_pose:
            visualize_pose_bbox(preds)

        adds = []

        for i in range(preds["whole_src_points"].shape[0]):
            sub_src_pts = preds["whole_src_points"][i]
            pred_R = preds["pred_Rs"][i].reshape(3, 3).astype(np.float32)
            pred_T = preds["pred_Ts"][i].reshape(3).astype(np.float32)
            sub_target_pts = (sub_src_pts - pred_T[None, :]) @ pred_R
            sub_model_pts = preds["whole_model_points"][i]
            adds.append(compute_add_error(sub_target_pts, sub_model_pts))
            if vis_pts:
                visualize_point_cloud(sub_target_pts, sub_model_pts)
        preds.update({"adds": adds})
        return preds


if __name__ == "__main__":
    pose_model_pt_path = \
        "/home/yhlever/DeepLearning/MaskPoseNet-V2/middle_log/0302-indus-train-b/checkpoints/best.pt"
    seg_model_onnx_path = '/home/yhlever/DeepLearning/ultralytics/indus/0219-real-train/weights/best.onnx'
    image = cv2.imread("/home/yhlever/CLionProjects/ROBOT_GRASP_TORCH/results/image_000002.png",
                       flags=cv2.IMREAD_COLOR)
    depth_image = cv2.imread("/home/yhlever/CLionProjects/ROBOT_GRASP_TORCH/results/image_000003.png",
                             flags=cv2.IMREAD_UNCHANGED)
    camera_k = DefaultCameraK
    my_model = Model(pose_model_pt_path, seg_model_onnx_path, DefaultArgs, device="cuda")
    preds = my_model(image, depth_image, camera_k, save_seg=False, save_pose=True, vis_pts=False)
    print_preds_table(preds)
