import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import trimesh
from loguru import logger
from pycocotools import mask as mask_utils

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
    assert annotation_key in ["scene_camera.json", "scene_instances_gt.json", "scene_pose_gt.json", "scene_gt.json",
                              "scene_gt_coco.json", "scene_gt_info.json"], \
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


def get_models(file_base, cls_name, n_sample):
    mesh_path = os.path.join(file_base, cls_name + '.obj')
    mesh = trimesh.load(mesh_path, force='mesh')
    model_pts, _ = trimesh.sample.sample_surface(mesh, n_sample)
    return model_pts.astype(np.float32)


def transform(image):
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])(image)


def get_template(cfg, cls_name, tem_index=0):
    file_base = cfg.TEMPLATE_DIR
    rgb_path = os.path.join(file_base, cls_name, f'rgb_{tem_index}.png')
    xyz_path = os.path.join(file_base, cls_name, f'xyz_{tem_index}.npy')
    mask_path = os.path.join(file_base, cls_name, f'mask_{tem_index}.png')

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) == 255
    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = rgb[y1:y2, x1:x2, :]

    if cfg.RGB_MASK_FLAG:
        rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
    rgb = cv2.resize(rgb, (cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = transform(rgb)

    choose = mask.astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.N_SAMPLE_TEMPLATE_POINT:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.N_SAMPLE_TEMPLATE_POINT)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.N_SAMPLE_TEMPLATE_POINT, replace=False)
    choose = choose[choose_idx]

    xyz = np.load(xyz_path).astype(np.float32)
    xyz = convert_blender_to_pyrender(xyz)[y1:y2, x1:x2, :]
    xyz = xyz.reshape((-1, 3))[choose, :]
    choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.IMG_SIZE)

    return rgb, choose, xyz


def get_all_templates(cfg, cls_name, device):
    n_template_view = cfg.N_TEMPLATE_VIEW
    all_tem_rgb = [[] for i in range(n_template_view)]
    all_tem_choose = [[] for i in range(n_template_view)]
    all_tem_pts = [[] for i in range(n_template_view)]

    for i in range(n_template_view):
        tem_rgb, tem_choose, tem_pts = get_template(cfg, cls_name, i)
        all_tem_rgb[i].append(torch.FloatTensor(tem_rgb))
        all_tem_choose[i].append(torch.IntTensor(tem_choose).long())
        all_tem_pts[i].append(torch.FloatTensor(tem_pts))

    for i in range(n_template_view):
        all_tem_rgb[i] = torch.stack(all_tem_rgb[i]).to(device, non_blocking=True)
        all_tem_choose[i] = torch.stack(all_tem_choose[i]).to(device, non_blocking=True)
        all_tem_pts[i] = torch.stack(all_tem_pts[i]).to(device, non_blocking=True)

    return all_tem_rgb, all_tem_pts, all_tem_choose


def load_image_if_path(input_data, flags=cv2.IMREAD_COLOR):
    if isinstance(input_data, (str, Path)):
        input_data = str(input_data)
        img = cv2.imread(input_data, flags)
        if img is None:
            raise FileNotFoundError(f"Image not found at {input_data}")
        return img
    return input_data


def load_pose_inference_source(image, depth_image, mask, obj, camera_k, cfg, class_names, model_dir=None):
    """
    Loads data sources for pose estimation, normalizes the input image, depth map, and mask,
    and generates a dictionary containing point clouds, RGB data, model points, and other relevant information.
    Supports batch processing for multiple objects.

    Args:
        image (str or Path or numpy.ndarray):
            Path to the image or a numpy array in BGR format loaded with cv2.
        depth_image (str or Path or numpy.ndarray):
            Path to the depth map or a numpy array representing depth values.
        mask (str or Path or numpy.ndarray or list):
            Single or multiple mask paths or numpy arrays with values in the range [0, 1] or [0, 255].
            For multiple objects, provide a list of masks corresponding to each object in `obj`.
        obj (int or list):
            Single or multiple object IDs. For multiple objects, provide a list of IDs.
        camera_k (numpy.ndarray):
            Camera intrinsic matrix of shape (3, 3).
        cfg:
            Configuration object containing system and data parameters.
        model_dir (str or Path, optional):
            Path to the model directory. If not specified, the default path from `cfg` will be used.

    Returns:
        dict: A dictionary containing the following keys:
            - 'pts': A numpy array of shape [batch, n_sample_observed_point, 3] representing point clouds.
            - 'rgb': A numpy array of shape [batch, 3, IMG_SIZE, IMG_SIZE] containing normalized RGB data.
            - 'rgb_choose': A numpy array of shape [batch, n_sample_observed_point] with chosen RGB indices.
            - 'model': A numpy array of shape [batch, N_SAMPLE_MODEL_POINT, 3] representing model points.
            - 'obj': A numpy array of shape [batch] containing object IDs.
        numpy.ndarray: The original RGB image.
        numpy.ndarray: The complete set of model points.
    """
    # 确保obj和mask是列表，以便统一处理多对象场景
    if not isinstance(obj, (list, tuple)):
        obj = [obj]
    if not isinstance(mask, (list, tuple)):
        mask = [mask]

    if len(mask) != len(obj):
        raise ValueError("mask和obj数量不匹配")

    # 加载/检查image
    image = load_image_if_path(image, flags=cv2.IMREAD_COLOR)
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Image is not a valid numpy array")

    # 加载/检查depth_image
    depth_image = load_image_if_path(depth_image, flags=cv2.IMREAD_UNCHANGED)
    if depth_image is None or not isinstance(depth_image, np.ndarray):
        raise ValueError("Depth image is not a valid numpy array")
    if depth_image.ndim == 3 and depth_image.shape[2] == 3:
        # Convert the 3-channel image to a single channel
        depth_image = depth_image[:, :, 0]

    # 相机内参处理
    camera_k = np.array(camera_k, dtype=np.float32).reshape(3, 3)

    model_dir = model_dir or Path(cfg.DATA_DIR) / cfg.MODEL_DIR

    # 将RGB从BGR转为RGB备用
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 深度归一化为米
    depth = depth_image.astype(np.float32) * cfg.DEPTH_SCALE / 1000.0 + 0.085

    whole_pts = get_point_cloud_from_depth(depth, camera_k)

    # 定义transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    all_pts = []
    all_rgb = []
    all_rgb_choose = []
    all_model_points = []
    all_obj = []
    whole_model_points = []

    for m, o in zip(mask, obj):
        # 加载/检查mask
        m = load_image_if_path(m, flags=cv2.IMREAD_GRAYSCALE)
        if m is None or not isinstance(m, np.ndarray):
            raise ValueError("Mask is not a valid numpy array")

        # 检查mask值范围，若在0-1之间则转换到0-255
        if m.max() <= 1.0:
            m = (m * 255).astype(np.uint8)
        else:
            m = m.astype(np.uint8)

        # 加载模型点
        cls_name = class_names[o]
        model_points = get_models(model_dir, cls_name, cfg.N_SAMPLE_MODEL_POINT)
        if model_points is None:
            continue
        radius = np.max(np.linalg.norm(model_points, axis=1))

        # 获取mask的bbox
        m = np.logical_and(m > 0, depth > 0)
        bbox = get_bbox(m > 0)
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
        if len(choose) < cfg.N_SAMPLE_OBSERVED_POINT:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.N_SAMPLE_OBSERVED_POINT, replace=True)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.N_SAMPLE_OBSERVED_POINT, replace=False)

        choose = choose[choose_idx]
        pts = pts[choose_idx]

        # 裁剪并处理RGB
        sub_rgb = rgb_img[y1:y2, x1:x2, :]
        if cfg.RGB_MASK_FLAG:
            sub_rgb = sub_rgb * (sub_mask[:, :, None] > 0).astype(np.uint8)
        sub_rgb = cv2.resize(sub_rgb, (cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        sub_rgb = transform(sub_rgb)  # [C,H,W]

        # 获取resize后对应的rgb_choose
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.IMG_SIZE)

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
    whole_model_points = np.stack(whole_model_points, axis=0)
    return ret_dict, image, depth_image, mask, whole_model_points
