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
import torch
import trimesh
import yaml
from scipy.spatial import cKDTree
from torchvision import transforms

from utils.visualize import visualize_point_cloud
from ultralytics import YOLO

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
    """Initialize random number generator seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe


def compute_add_error(pc1, pc2):
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    dist1, _ = tree2.query(pc1)
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
    col_width = 10
    header_fmt = "{:<{w}} {:<{w}} {:<{w}} {:<{w}}".format
    row_fmt = "{:<{w}} {:<{w}.2f} {:<{w}.2f} {:<{w}.2f}".format

    header = header_fmt("cls_name", "seg_score", "pose_score", "add", w=col_width)
    print(header)

    for cls_name, seg_score, pose_score, add in zip(preds['cls_names'], preds['seg_scores'], preds['pose_scores'],
                                                    preds['adds']):
        print(row_fmt(cls_name, seg_score, pose_score, add, w=col_width))

    avg_seg_score = np.mean(preds['seg_scores'])
    avg_pose_score = np.mean(preds['pose_scores'])
    avg_add = np.mean(preds['adds'])
    print(row_fmt("all", avg_seg_score, avg_pose_score, avg_add, w=col_width))

    print("-" * len(header))
    print(f"Segmentation inference time: {preds['seg_inference']:.4f} ms, "
          f"Pose inference time: {preds['pose_inference']:.4f} ms")


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

    def __call__(self, batch):
        with torch.no_grad():
            batch = self.preprocess(batch)
            end_points = self.inference(batch)
            preds = self.postprocess(end_points)
            preds['cls_names'] = end_points["cls_names"]
            preds['cls_ids'] = end_points['cls_ids']

        return preds

    def _load(self):
        if Path(self.weights).suffix == ".pt":
            self.model, _ = self._attempt_load_one_weight()
        else:
            raise ValueError(f"Unsupported weights file '{self.weights}'")

    def _attempt_load_one_weight(self):
        """Loads a single model weights."""
        ckpt = torch.load(self.weights)

        if "model" not in ckpt:
            raise ValueError(f"'model' key missing in checkpoint {self.weights}")
        if "class_names" not in ckpt:
            raise ValueError(f"'class_names' key missing in checkpoint {self.weights}")
        self.class_names = ckpt["class_names"]

        # 根据 'model' 是 state_dict 还是完整 nn.Module，做区分
        model_data = ckpt["model"]

        if isinstance(model_data, dict):
            from models import PoseModel
            from easydict import EasyDict
            if PoseModel is None:
                raise ValueError(
                    "Checkpoint contains only a state_dict, but no model_builder provided!\n"
                    "Please provide a callable model_builder() that returns an uninitialized model."
                )
            model_cfg = EasyDict(ckpt["train_args"]['POSE_MODEL'])
            model_cfg.FEATURE_EXTRACTION.PRETRAINED = False  # inference not need to load pretrained mae weights
            model = PoseModel(model_cfg)
            model.load_state_dict(model_data)
            model.to(self.device).float()
        else:
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

    def load_pose_inference_source(self, image, depth_image, mask, obj, camera_k):
        """
        Loads data sources for pose estimation, normalizes the input image, depth map, and mask,
        and generates a dictionary containing point clouds, RGB data, model points, and other relevant information.
        Supports batch processing for multiple objects.
        """
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

        for m, o in zip(mask, obj):
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
            return None, None, None

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

    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, cls_ids)
    img = np.uint8(img)

    save_path = 'vis_pose_bbox.png'
    cv2.imwrite(save_path, img)


class Model:
    def __init__(self, pose_model_path, seg_model_path, args, device):
        """
        Initial Model.

        Args:
            pose_model_path (str): 姿态模型文件路径。
            seg_model_path (str): 分割模型文件路径（ONNX格式）。
            device (str):  "cuda" 或 "cpu"。
        """
        init_seeds(42)
        self.device = device
        self.seg_model = YOLO(seg_model_path, task='segment')
        self.pose_model = ModelPose(pose_model_path, args=args, device=device)

    def __call__(self, img: np.ndarray, depth_img: np.ndarray, camera_k: np.ndarray,
                 save_seg: bool = False, save_pose: bool = True, vis_pts: bool = False,
                 pose_all: bool = False):
        """
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
            seg_result = self.seg_model(img, conf=0.6, iou=0.6, imgsz=(960, 1280), half=True,
                                        device=self.device)[0]
            boxes = seg_result.boxes.cpu().numpy()
            seg_scores = boxes.conf.tolist()
            cls_ids = (boxes.cls.astype(int) + 1).tolist()
            masks = seg_result.masks.data.cpu().numpy()
            mask_list = [masks[i] for i in range(masks.shape[0])]
            segments = seg_result.masks.xy

        avg_depths = []
        for m in mask_list:
            valid = (m > 0)
            if np.any(valid):
                depth_vals = depth_img[valid]
                depth_vals = depth_vals[depth_vals > 0]
                if depth_vals.size > 0:
                    avg_depths.append(depth_vals.mean())
                else:
                    avg_depths.append(np.inf)
            else:
                avg_depths.append(np.inf)
        order = np.argsort(avg_depths)
        seg_scores = [seg_scores[i] for i in order]
        segments = [segments[i] for i in order]
        cls_ids = [cls_ids[i] for i in order]
        mask_list = [mask_list[i] for i in order]

        with Profile() as pp:
            if not pose_all and len(segments) > 1:
                for i in range(len(segments)):
                    # 按照 idx=i 仅保留单个结果
                    tmp_scores = seg_scores[i:i + 1]
                    tmp_segs = segments[i:i + 1]
                    tmp_cls = cls_ids[i:i + 1]
                    tmp_masks = mask_list[i:i + 1]
                    source.update({
                        "segments": tmp_segs,
                        "seg_scores": tmp_scores,
                        "cls_ids": tmp_cls,
                        "seg_mask": tmp_masks,
                    })

                    # 尝试加载并解包
                    result = self.pose_model.load_pose_inference_source(
                        image=source["image"],
                        depth_image=source["depth_image"],
                        mask=source["seg_mask"],
                        obj=source["cls_ids"],
                        camera_k=source["camera_k"],
                    )
                    if result[0] is not None:
                        batch, whole_target_pts, whole_model_points = result
                        preds = self.pose_model(batch)
                        preds["whole_src_points"] = whole_target_pts
                        preds["whole_model_points"] = whole_model_points
                        break
                else:
                    raise RuntimeError("所有分割结果都无法生成有效的点云，跳过该帧。")
            else:
                # pose_all=True 或者只有一条结果，正常加载
                result = self.pose_model.load_pose_inference_source(
                    image=source["image"],
                    depth_image=source["depth_image"],
                    mask=source["seg_mask"],
                    obj=source["cls_ids"],
                    camera_k=source["camera_k"],
                )
                if result[0] is None:
                    raise RuntimeError("load_pose_inference_source 返回 None，请检查分割结果。")
                batch, whole_target_pts, whole_model_points = result
                preds = self.pose_model(batch)
                preds["whole_src_points"] = whole_target_pts
                preds["whole_model_points"] = whole_model_points

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
    image = cv2.imread("/home/yhlever/CLionProjects/ROBOT_GRASP_TORCH/results/image_000050.png",
                       flags=cv2.IMREAD_COLOR)
    depth_image = cv2.imread("/home/yhlever/CLionProjects/ROBOT_GRASP_TORCH/results/image_000051.png",
                             flags=cv2.IMREAD_UNCHANGED)
    camera_k = DefaultCameraK
    my_model = Model(pose_model_pt_path, seg_model_onnx_path, DefaultArgs, device="cuda")
    preds = my_model(image, depth_image, camera_k, save_seg=False, save_pose=True, vis_pts=False, pose_all=False)
    print_preds_table(preds)
    print(preds['pred_Rs'])
    print(preds['pred_Ts'])
