import itertools
import os
from collections import defaultdict

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import torchvision.transforms as transforms
from imgaug.augmenters import (Sequential, Sometimes, Grayscale,
                               GaussianBlur, Add, AdditiveGaussianNoise, Multiply,
                               CoarseDropout, Invert, pillike)

from base_dataset import DatasetBase
from data_utils import (
    _isArrayLike,
    convert_blender_to_pyrender,
    load_color_image,
    load_depth_image,
    load_mask,
    load_anns,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
    get_random_rotation,
    get_bbox,
)


class PoseNetDataset(DatasetBase):
    """
    PoseNet数据集类，继承自DatasetBase。
    该类负责加载和处理PoseNet模型所需的数据集。
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.cfg = cfg

        self.data_dir = cfg.DATA_DIR
        self.num_img_per_epoch = cfg.NUM_IMG_PER_EPOCH
        self.min_visib_px = cfg.MIN_PX_COUNT_VISIB
        self.min_visib_frac = cfg.MIN_VISIB_FRACT
        self.dilate_mask = cfg.DILATE_MASK
        self.rgb_mask_flag = cfg.RGB_MASK_FLAG
        self.shift_range = cfg.SHIFT_RANGE
        self.img_size = cfg.IMG_SIZE
        self.n_sample_observed_point = cfg.N_SAMPLE_OBSERVED_POINT
        self.n_sample_model_point = cfg.N_SAMPLE_MODEL_POINT
        self.n_sample_template_point = cfg.N_SAMPLE_TEMPLATE_POINT

        self.dataset_dir = os.path.join(self.data_dir, cfg.DATASET_DIR)
        self.model_dir = os.path.join(self.data_dir, cfg.MODEL_DIR)
        self.templates_dir = os.path.join(self.data_dir, cfg.TEMPLATE_DIR)

        self.color_augmentor = (
            Sequential([
                Sometimes(0.5, CoarseDropout(p=0.2, size_percent=0.05)),
                Sometimes(0.4, GaussianBlur((0., 3.))),
                Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),
                Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),
                Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),
                Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),
                Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
                Sometimes(0.3, Invert(0.2, per_channel=True)),
                Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
                Sometimes(0.5, Multiply((0.6, 1.4))),
                Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),
                Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
                Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),
            ], random_order=True)
        )
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def load_data(self, subset: str = "train"):
        image_dir = os.path.join(self.dataset_dir, subset, "images", "color_ims")
        depth_dir = os.path.join(self.dataset_dir, subset, "images", "depth_ims")
        self.scene_camera_info = load_anns(self.dataset_dir, subset, "scene_camera.json")
        self.scene_instances_gt_info = load_anns(self.dataset_dir, subset, "scene_instances_gt.json")
        self.scene_pose_gt_info = load_anns(self.dataset_dir, subset, "scene_pose_gt.json")
        self.createIndex()
        class_ids = self.getCatIds()
        image_ids = list(self.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("dataset", i, self.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            image_file = self.imgs[i]['file_name']
            self.add_image(
                "dataset", image_id=i,
                path=os.path.join(image_dir, image_file),
                depth_path=os.path.join(depth_dir, image_file),
                width=self.imgs[i]["width"],
                height=self.imgs[i]["height"],
                annotations=self.loadAnns(self.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

        self.prepare()

    def createIndex(self):
        print('Creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        annotations = self.scene_instances_gt_info.get('annotations', [])
        images = self.scene_instances_gt_info.get('images', [])
        categories = self.scene_instances_gt_info.get('categories', [])

        for ann in annotations:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

        for img in images:
            imgs[img['id']] = img

        for cat in categories:
            cats[cat['id']] = cat

        for ann in annotations:
            catToImgs[ann['category_id']].append(ann['image_id'])

        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        print('Index created!')

    def getCatIds(self):
        cats = self.scene_instances_gt_info.get('categories', [])
        ids = [cat['id'] for cat in cats]
        return ids

    def loadCats(self, ids):
        if isinstance(ids, list):
            return [self.cats[id] for id in ids]
        else:
            return [self.cats[ids]]

    def loadAnns(self, ids=None):
        if ids is None:
            ids = []
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif isinstance(ids, int):
            return [self.anns[ids]]

    def getAnnIds(self, imgIds=None, catIds=None, iscrowd=None):
        if catIds is None:
            catIds = []
        if imgIds is None:
            imgIds = []
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            anns = list(self.anns.values())
        else:
            if len(imgIds) > 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = list(self.anns.values())
            if len(catIds) > 0:
                anns = [ann for ann in anns if ann['category_id'] in catIds]

        if iscrowd is not None:
            ids = [ann['id'] for ann in anns if ann.get('iscrowd', 0) == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def load_image_annotations(self, image_id):
        return self.image_info[image_id]['annotations']

    def load_pose_Rt(self, image_id, annotation_id):
        image_id_str = str(image_id)
        if image_id_str in self.scene_pose_gt_info:
            for annotation in self.scene_pose_gt_info[image_id_str]:
                if annotation['annotation_id'] == annotation_id:
                    return annotation['cam_R_m2c'], annotation['cam_t_m2c']
        return None, None

    def load_camera_k(self, image_id):
        image_id_str = str(image_id)
        if image_id_str in self.scene_camera_info:
            return self.scene_camera_info[image_id_str]['cam_K']
        return None

    def get_template(self, file_base, category_id, tem_index=1):
        category_id_str = f"obj_{int(category_id):02d}"
        rgb_path = os.path.join(file_base, category_id_str, f'rgb_{tem_index}.png')
        xyz_path = os.path.join(file_base, category_id_str, f'xyz_{tem_index}.npy')
        mask_path = os.path.join(file_base, category_id_str, f'mask_{tem_index}.png')

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) == 255
        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[y1:y2, x1:x2, :]

        if np.random.rand() < 0.8 and self.color_augmentor is not None:
            rgb = self.color_augmentor.augment_image(rgb)
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(rgb)

        choose = mask.astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= self.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point, replace=False)
        choose = choose[choose_idx]

        xyz = np.load(xyz_path).astype(np.float32)
        xyz = convert_blender_to_pyrender(xyz)[y1:y2, x1:x2, :]
        xyz = xyz.reshape((-1, 3))[choose, :]
        choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        return rgb, choose, xyz

    def get_train_data(self, index):
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            image_id = index % self.num_images
            annotations = self.load_image_annotations(image_id)
            if not annotations:
                index += 1
                attempt += 1
                continue

            valid_annotation = np.random.choice(annotations)
            category_id = valid_annotation['category_id']
            annotation_id = valid_annotation['id']

            target_R, target_t = self.load_pose_Rt(image_id, annotation_id)
            if target_R is None or target_t is None:
                index += 1
                attempt += 1
                continue
            target_R = np.array(target_R).reshape(3, 3).astype(np.float32)
            target_t = np.array(target_t).reshape(3).astype(np.float32)

            camera_k = self.load_camera_k(image_id)
            if camera_k is None:
                index += 1
                attempt += 1
                continue
            camera_k = np.array(camera_k).reshape(3, 3).astype(np.float32)

            try:
                tem1_rgb, tem1_choose, tem1_pts = self.get_template(self.templates_dir, category_id, 1)
                tem2_rgb, tem2_choose, tem2_pts = self.get_template(self.templates_dir, category_id, 35)
            except FileNotFoundError:
                index += 1
                attempt += 1
                continue

            mask = load_mask(valid_annotation, self.image_info[image_id]['height'],
                             self.image_info[image_id]['width'])
            if mask is None or np.sum(mask) == 0:
                index += 1
                attempt += 1
                continue

            if self.dilate_mask and np.random.rand() < 0.5:
                mask = np.array(mask > 0).astype(np.uint8)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
            mask = (mask * 255).astype(np.uint8)

            bbox = get_bbox(mask > 0)
            y1, y2, x1, x2 = bbox
            mask = mask[y1:y2, x1:x2]
            choose = mask.astype(np.float32).flatten().nonzero()[0]

            depth_path = self.image_info[image_id]['depth_path']
            depth = load_depth_image(depth_path)
            depth = depth / 1000.0 * 5  # Assuming depth_scale is 5
            pts = get_point_cloud_from_depth(depth, camera_k, [y1, y2, x1, x2])
            pts = pts.reshape(-1, 3)[choose, :]

            target_pts = (pts - target_t[None, :]) @ target_R
            tem_pts = np.concatenate([tem1_pts, tem2_pts], axis=0)
            radius = np.max(np.linalg.norm(tem_pts, axis=1))
            target_radius = np.linalg.norm(target_pts, axis=1)
            flag = target_radius < radius * 1.2

            pts = pts[flag]
            choose = choose[flag]

            if len(choose) < 32:
                index += 1
                attempt += 1
                continue

            if len(choose) <= self.n_sample_observed_point:
                choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
            choose = choose[choose_idx]
            pts = pts[choose_idx]

            image_path = self.image_info[image_id]['path']
            rgb = load_color_image(image_path)
            rgb = rgb[..., ::-1][y1:y2, x1:x2, :]
            if np.random.rand() < 0.8 and self.color_augmentor is not None:
                rgb = self.color_augmentor.augment_image(rgb)
            if self.rgb_mask_flag:
                rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
            rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = self.transform(rgb)
            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

            rand_R = get_random_rotation()
            tem1_pts = tem1_pts @ rand_R
            tem2_pts = tem2_pts @ rand_R
            target_R = target_R @ rand_R

            add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            target_t += add_t[0]
            add_t += 0.001 * np.random.randn(pts.shape[0], 3)
            pts += add_t

            input_dict = {
                'pts': torch.FloatTensor(pts),
                'rgb': torch.FloatTensor(rgb),
                'rgb_choose': torch.IntTensor(rgb_choose).long(),
                'translation_label': torch.FloatTensor(target_t),
                'rotation_label': torch.FloatTensor(target_R),
                'tem1_rgb': torch.FloatTensor(tem1_rgb),
                'tem1_choose': torch.IntTensor(tem1_choose).long(),
                'tem1_pts': torch.FloatTensor(tem1_pts),
                'tem2_rgb': torch.FloatTensor(tem2_rgb),
                'tem2_choose': torch.IntTensor(tem2_choose).long(),
                'tem2_pts': torch.FloatTensor(tem2_pts),
            }

            return input_dict

        def get_test_data(self, index):
            return self.get_train_data(index)
