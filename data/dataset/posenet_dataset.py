import os

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from albumentations.core.transforms_interface import ImageOnlyTransform
from loguru import logger

from data.dataset.base_dataset import DatasetBase
from data.dataset.data_utils import (
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


class AdditiveGaussianNoise(ImageOnlyTransform):
    def __init__(self, scale=10.0, per_channel=False, always_apply=False, p=0.5):
        super(AdditiveGaussianNoise, self).__init__(always_apply, p)
        self.scale = scale
        self.per_channel = per_channel

    def apply(self, img, **params):
        noise = np.random.normal(0, self.scale, img.shape)
        if self.per_channel and img.ndim == 3:
            noise = noise[..., np.newaxis]
        return np.clip(img + noise, 0, 255).astype(np.uint8)


class PoseNetDataset(DatasetBase):
    """
    PoseNet数据集类，继承自DatasetBase。
    该类负责加载和处理PoseNet模型所需的数据集。
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.cfg = cfg
        self.data_dir = cfg.DATA_DIR
        # self.num_img_per_epoch = cfg.NUM_IMG_PER_EPOCH
        # self.min_visib_px = cfg.MIN_PX_COUNT_VISIB
        # self.min_visib_frac = cfg.MIN_VISIB_FRACT
        self.depth_scale = cfg.DEPTH_SCALE
        self.rgb_mask_flag = cfg.RGB_MASK_FLAG
        self.img_size = cfg.IMG_SIZE
        self.n_sample_observed_point = cfg.N_SAMPLE_OBSERVED_POINT
        self.n_sample_model_point = cfg.N_SAMPLE_MODEL_POINT
        self.n_sample_template_point = cfg.N_SAMPLE_TEMPLATE_POINT
        if self.is_train:
            self.shift_range = cfg.SHIFT_RANGE
            self.dilate_mask = cfg.DILATE_MASK

        self.dataset_dir = os.path.join(self.data_dir, cfg.DATASET_DIR)
        self.model_dir = os.path.join(self.data_dir, cfg.MODEL_DIR)
        self.templates_dir = os.path.join(self.data_dir, cfg.TEMPLATE_DIR)

        self.color_augmentor = A.Compose([
            A.OneOf([
                A.CoarseDropout(max_holes=5, max_height=20, max_width=20, p=0.5),  # 替代 CoarseDropout
                A.GaussianBlur(blur_limit=(3, 7), p=0.4),  # 替代 GaussianBlur
            ], p=0.5),
            A.OneOf([
                A.Sharpen(alpha=(0.0, 1.0), lightness=(0.5, 2.0), p=0.3),  # 替代 EnhanceSharpness
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5),  # 替代 Brightness/Contrast
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),  # 替代 EnhanceColor
            ], p=0.8),
            A.InvertImg(p=0.3),  # 替代 Invert
        ], p=1.0)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        if self.is_train:
            self.load_data("train")
        else:
            self.load_data("val")

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
            self.add_class("", i, self.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            image_file = self.imgs[i]['file_name']
            self.add_image(
                source="",
                image_id=i,
                path=os.path.join(image_dir, image_file),
                depth_path=os.path.join(depth_dir, image_file),
                width=self.imgs[i]["width"],
                height=self.imgs[i]["height"],
                annotations=self.loadAnns(self.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None))
            )

        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)
        logger.info(f"Number of images: {self.num_images}")
        logger.info(f"Number of classes: {self.num_classes}")

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
            rgb = self.color_augmentor(image=rgb)["image"]
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
        # 获取当前训练图片ID及其所有标注
        image_id = self.image_ids[index]
        annotations = self.load_image_annotations(image_id)
        if len(annotations) == 0:
            return None

        # 随机选择一个有效标注
        valid_annotation = np.random.choice(annotations)
        category_id = valid_annotation['category_id']
        annotation_id = valid_annotation['id']

        # 加载位姿信息（旋转矩阵R和平移向量t
        target_R, target_t = self.load_pose_Rt(image_id, annotation_id)
        target_R = np.array(target_R).reshape(3, 3).astype(np.float32)
        target_t = np.array(target_t).reshape(3).astype(np.float32)

        # 加载相机内参矩阵
        camera_k = self.load_camera_k(image_id)
        if camera_k is None:
            return None
        camera_k = np.array(camera_k).reshape(3, 3).astype(np.float32)

        # 加载模板数据（两种不同视角）
        tem1_rgb, tem1_choose, tem1_pts = self.get_template(self.templates_dir, category_id, 1)
        tem2_rgb, tem2_choose, tem2_pts = self.get_template(self.templates_dir, category_id, 35)
        if tem1_rgb is None or tem2_rgb is None:
            return None

        # 加载目标物体mask
        mask = load_mask(valid_annotation, self.image_info[image_id]['height'], self.image_info[image_id]['width'])
        if mask is None or np.sum(mask) == 0:
            return None

        if self.dilate_mask and np.random.rand() < 0.5:
            mask = np.array(mask > 0).astype(np.uint8)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
        mask = (mask * 255).astype(np.uint8)
        bbox = get_bbox(mask > 0)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # 加载深度图数据
        depth_path = self.image_info[image_id]['depth_path']
        depth = load_depth_image(depth_path)
        depth = depth * self.depth_scale / 1000.0
        pts = get_point_cloud_from_depth(depth, camera_k, [y1, y2, x1, x2])
        pts = pts.reshape(-1, 3)[choose, :]

        # 将点云转换到目标物体的坐标系
        target_pts = (pts - target_t[None, :]) @ target_R
        tem_pts = np.concatenate([tem1_pts, tem2_pts], axis=0)
        radius = np.max(np.linalg.norm(tem_pts, axis=1))
        target_radius = np.linalg.norm(target_pts, axis=1)
        flag = target_radius < radius * 1.2
        pts = pts[flag]
        choose = choose[flag]
        if len(choose) < 32:
            return None
        if len(choose) <= self.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        pts = pts[choose_idx]

        # rgb
        image_path = self.image_info[image_id]['path']
        rgb = load_color_image(image_path)
        rgb = rgb[..., ::-1][y1:y2, x1:x2, :]
        if np.random.rand() < 0.8 and self.color_augmentor is not None:
            rgb = self.color_augmentor(image=rgb)["image"]
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(rgb)
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        # 随机旋转增强
        rand_R = get_random_rotation()
        tem1_pts = tem1_pts @ rand_R
        tem2_pts = tem2_pts @ rand_R
        target_R = target_R @ rand_R

        # 随机平移增强
        add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
        target_t += add_t[0]
        add_t += 0.001 * np.random.randn(1, 3)
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


if __name__ == "__main__":
    import random
    from easydict import EasyDict as edict
    import yaml
    from data.dataloader.build import build_dataloader

    with open('../../cfg/base.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = edict(cfg_dict)


    # 测试代码
    def test_posenet_dataset():
        print("开始测试PoseNetDataset加载...")

        # 创建PoseNetDataset实例
        train_dataset = PoseNetDataset(cfg.TRAIN_DATA, is_train=True)
        print(f"数据集加载完成，共有 {len(train_dataset)} 张训练图片")

        # 加载数据
        index = random.randint(0, len(train_dataset) - 1)
        print(f"随机选取样本索引: {index}")

        train_data = train_dataset.get_train_data(index)
        if train_data is None:
            print(f"样本 {index} 加载失败，可能数据不足或标注缺失")
        else:
            print(f"样本 {index} 加载成功，关键字段如下：")
            print(f" - pts: {train_data['pts'].shape}")
            print(f" - rgb: {train_data['rgb'].shape}")
            print(f" - tem1_rgb: {train_data['tem1_rgb'].shape}")
            print(f" - translation_label: {train_data['translation_label']}")
            print(f" - rotation_label: {train_data['rotation_label'].shape}")

        # 使用DataLoader测试批量加载
        train_dataloader = build_dataloader(train_dataset, batch=cfg.TRAIN_DATA.BATCH_SIZE,
                                                  workers=cfg.TRAIN_DATA.WORKERS, shuffle=True)
        # collate_fn=None
        # {
        #     'rgb': [Tensor1, Tensor2, Tensor3, ...],  # 每个样本的 'rgb'
        #     'pts': [Tensor4, Tensor5, Tensor6, ...]  # 每个样本的 'pts'
        # }

        print("测试批量加载...")
        for batch_index, batch_data in enumerate(train_dataloader):
            print(f"Batch {batch_index + 1}: 数据类型: {type(batch_data)}")
            print(f"Batch {batch_index + 1}: 数据内容: {batch_data.keys()}")
            print(f"Batch {batch_index + 1}: 样本数量: {len(next(iter(batch_data.values())))}")
            if len(batch_data) > 0:
                # 遍历字典内容
                for key, value in batch_data.items():
                    print(
                        f" - Key: {key}, 第一个样本 shape: {value[0].shape if hasattr(value[0], 'shape') else type(value[0])}")

            break


    test_posenet_dataset()
