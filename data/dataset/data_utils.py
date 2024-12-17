import json
import os
import time
import trimesh
import cv2
import numpy as np
from loguru import logger
from pycocotools import mask as mask_utils
import torchvision.transforms as transforms
import torch
from pathlib import Path

"""
Helper functions for loading data.
"""
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
DEAFULT_MODEL_DIR = "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/HANDLE/models"


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def convert_blender_to_pyrender(blender_nocs):
    # 交换 Y 和 Z 轴
    pyrender_nocs = blender_nocs[:, :, [0, 2, 1]]
    # 反转 Y 轴
    pyrender_nocs[:, :, 2] *= -1
    return pyrender_nocs


def load_color_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image.ndim != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def load_depth_image(depth_path):
    """Load a depth image from a file path."""
    image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if image.ndim == 3:
        image = image[:, :, 0]
    return image


def load_anns(dataset_dir, subset, annotation_key):
    assert annotation_key in ["scene_camera.json", "scene_instances_gt.json", "scene_pose_gt.json"], \
        f'Annotation file format {annotation_key} not supported.'
    annotations_file = os.path.join(dataset_dir, subset, annotation_key)
    logger.info(f'Loading {annotation_key} into memory...')
    tic = time.time()
    with open(annotations_file, 'r') as f:
        dataset = json.load(f)
    assert isinstance(dataset, dict), f'Annotation file format {type(dataset)} not supported.'
    logger.info(f'Done {annotation_key} (t={time.time() - tic:0.2f}s)')
    return dataset


def annToRLE(ann, height, width):
    """
    Convert annotation to RLE format.
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = mask_utils.frPyObjects(segm, height, width)
    else:
        rle = ann['segmentation']
    return rle


def annToMask(ann, height, width):
    """
    Convert RLE to a binary mask.
    """
    rle = annToRLE(ann, height, width)
    mask = mask_utils.decode(rle)
    return mask


def load_mask(ann, height, width):
    """Convert an annotation to a binary mask."""
    mask = annToMask(ann, height, width)
    if mask.max() < 1:
        return None
    return mask


def _gt_as_numpy(gt):
    if 'cam_R_m2c' in gt.keys():
        gt['cam_R_m2c'] = \
            np.array(gt['cam_R_m2c'], np.float64).reshape((3, 3))
    if 'cam_t_m2c' in gt.keys():
        gt['cam_t_m2c'] = \
            np.array(gt['cam_t_m2c'], np.float64).reshape((3, 1))
    return gt


def rle_to_binary_mask(rle):
    """Converts a COCOs run-length encoding (RLE) to binary mask.

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')

    start = 0
    for i in range(len(counts) - 1):
        start += counts[i]
        end = start + counts[i + 1]
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get('size'), order='F')

    return binary_mask


def get_point_cloud_from_depth(depth, k, bbox=None):
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


def get_resize_rgb_choose(choose, bbox, img_size):
    rmin, rmax, cmin, cmax = bbox
    crop_h = rmax - rmin
    ratio_h = img_size / crop_h
    crop_w = cmax - cmin
    ratio_w = img_size / crop_w

    row_idx = choose // crop_w
    col_idx = choose % crop_w
    choose = (np.floor(row_idx * ratio_h) * img_size + np.floor(col_idx * ratio_w)).astype(np.int64)
    return choose


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


def get_random_rotation():
    angles = np.random.rand(3) * 2 * np.pi
    rand_rotation = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ]) @ np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ]) @ np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    return rand_rotation


def get_models(file_base, category_id, n_sample):
    category_id_str = f"obj_{int(category_id):02d}.obj"
    mesh_path = os.path.join(file_base, category_id_str)
    mesh = trimesh.load(mesh_path, force='mesh')
    model_pts, _ = trimesh.sample.sample_surface(mesh, n_sample)
    return model_pts.astype(np.float32)


def load_image_if_path(input_data, flags=cv2.IMREAD_COLOR):
    if isinstance(input_data, (str, Path)):
        input_data = str(input_data)
        img = cv2.imread(input_data, flags)
        if img is None:
            raise FileNotFoundError(f"Image not found at {input_data}")
        return img
    return input_data


def load_pose_inference_source(image, depth_image, mask, obj, camera_k,
                               cfg, model_dir=None):
    """
    加载用于位姿推理的数据源，将图像、深度图和mask进行规范化处理，
    并生成点云、RGB、模型点以及其他相关信息的字典。

    Args:
        image: 可以是图像路径 (str/Path) 或者是通过cv2读取的numpy图片(BGR格式)。
        depth_image: 可以是深度图路径 (str/Path) 或者是深度的numpy数组。
        mask: 可以是掩码路径 (str/Path) 或者是掩码的numpy数组(0-1或0-255)。
        obj: 对象ID
        camera_k: 相机内参(3x3)
        cfg: 配置对象
        model_dir: 模型目录路径，可选，如果不提供则使用cfg中的默认路径

    Returns:：
        包含pts, rgb, rgb_choose, model, obj的字典。
    """

    # 加载/检查image
    image = load_image_if_path(image, flags=cv2.IMREAD_COLOR)
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Image is not a valid numpy array")

    # 加载/检查depth_image
    depth_image = load_image_if_path(depth_image, flags=cv2.IMREAD_UNCHANGED)
    if depth_image is None or not isinstance(depth_image, np.ndarray):
        raise ValueError("Depth image is not a valid numpy array")

    # 加载/检查mask
    mask = load_image_if_path(mask, flags=cv2.IMREAD_GRAYSCALE)
    if mask is None or not isinstance(mask, np.ndarray):
        raise ValueError("Mask is not a valid numpy array")

    # 检查mask值范围，若在0-1之间则转换到0-255
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    camera_k = np.array(camera_k, dtype=np.float32).reshape(3, 3)

    model_dir = model_dir or Path(cfg.DATA_DIR) / cfg.MODEL_DIR
    model_points = get_models(model_dir, obj, cfg.N_SAMPLE_MODEL_POINT)
    if model_points is None:
        return None

    bbox = get_bbox(mask > 0)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]
    choose = mask.astype(np.float32).flatten().nonzero()[0]

    depth = depth_image * cfg.DEPTH_SCALE / 1000.0
    pts = get_point_cloud_from_depth(depth, camera_k, [y1, y2, x1, x2])
    pts = pts.reshape(-1, 3)[choose, :]

    if len(choose) <= cfg.n_sample_observed_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
    choose = choose[choose_idx]
    pts = pts[choose_idx]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = rgb[y1:y2, x1:x2, :]
    if cfg.RGB_MASK_FLAG:
        rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
    rgb = cv2.resize(rgb, (cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    rgb = transform(rgb)
    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.IMG_SIZE)

    ret_dict = {
        'pts': torch.FloatTensor(pts),
        'rgb': torch.FloatTensor(rgb),
        'rgb_choose': torch.IntTensor(rgb_choose).long(),
        'model': torch.FloatTensor(model_points),
        'obj': torch.IntTensor([obj]).long(),
    }
    return ret_dict
