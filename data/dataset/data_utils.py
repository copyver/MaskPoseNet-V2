import os
import numpy as np
import json
import time
import cv2
from pycocotools import mask as mask_utils

"""
Helper functions for loading data.
"""
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
    print(f'Loading {annotation_key} into memory...')
    tic = time.time()
    with open(annotations_file, 'r') as f:
        dataset = json.load(f)
    assert isinstance(dataset, dict), f'Annotation file format {type(dataset)} not supported.'
    print(f'Done {annotation_key} (t={time.time() - tic:0.2f}s)')
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
    for i in range(len(counts)-1):
        start += counts[i]
        end = start + counts[i+1]
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get('size'), order='F')

    return binary_mask


def get_point_cloud_from_depth(depth, k, bbox=None):
    cam_fx, cam_fy, cam_cx, cam_cy = k[0,0], k[1,1], k[0,2], k[1,2]

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

    cloud = np.stack([pt0,pt1,pt2]).transpose((1,2,0))
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
        [1,0,0],
        [0,np.cos(angles[0]),-np.sin(angles[0])],
        [0,np.sin(angles[0]), np.cos(angles[0])]
    ]) @ np.array([
        [np.cos(angles[1]),0,np.sin(angles[1])],
        [0,1,0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ]) @ np.array([
        [np.cos(angles[2]),-np.sin(angles[2]),0],
        [np.sin(angles[2]), np.cos(angles[2]),0],
        [0,0,1]
    ])
    return rand_rotation
