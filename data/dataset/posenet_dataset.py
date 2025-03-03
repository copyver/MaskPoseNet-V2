import os

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import trimesh
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
from scipy.spatial import cKDTree


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
    return (np.mean(dist1) + np.mean(dist2)) / 2


class PoseNetDataset(DatasetBase):
    """
    PoseNet数据集类，继承自DatasetBase。
    该类负责加载和处理PoseNet模型所需的数据集。
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.data_dir = cfg.DATA_DIR
        self.depth_scale = cfg.DEPTH_SCALE
        self.rgb_mask_flag = cfg.RGB_MASK_FLAG
        self.img_size = cfg.IMG_SIZE
        self.n_sample_observed_point = cfg.N_SAMPLE_OBSERVED_POINT
        self.n_sample_template_point = cfg.N_SAMPLE_TEMPLATE_POINT
        self.dataset_dir = os.path.join(self.data_dir, cfg.DATASET_DIR)
        self.templates_dir = os.path.join(self.data_dir, cfg.TEMPLATE_DIR)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        if is_train:
            self.dilate_mask = cfg.DILATE_MASK
            self.shift_range = cfg.SHIFT_RANGE
            self.color_augmentor = A.Compose([
                A.OneOf([
                    A.CoarseDropout(max_holes=5, max_height=20, max_width=20, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.4),
                ], p=0.5),
                A.OneOf([
                    A.Sharpen(alpha=(0.0, 1.0), lightness=(0.5, 2.0), p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                ], p=0.8),
                A.InvertImg(p=0.3),
            ], p=1.0)
            self.load_data("train")

        else:
            self.n_sample_model_point = cfg.N_SAMPLE_MODEL_POINT
            self.model_dir = os.path.join(self.data_dir, cfg.MODEL_DIR)
            self.n_template_view = cfg.N_TEMPLATE_VIEW
            self.color_augmentor = None
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
        if subset == "val" and len(image_ids) > 100:
            image_ids = image_ids[:100]

        for i in image_ids:
            image_file = self.imgs[i]['file_name']
            self.add_image(
                source="",
                id=i,
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

    def get_all_templates(self, cls_name, device):
        n_template_view = self.n_template_view
        all_tem_rgb = [[] for i in range(n_template_view)]
        all_tem_choose = [[] for i in range(n_template_view)]
        all_tem_pts = [[] for i in range(n_template_view)]

        for i in range(n_template_view):
            tem_rgb, tem_choose, tem_pts = self.get_template(self.templates_dir, cls_name, i)
            all_tem_rgb[i].append(torch.FloatTensor(tem_rgb))
            all_tem_choose[i].append(torch.IntTensor(tem_choose).long())
            all_tem_pts[i].append(torch.FloatTensor(tem_pts))

        for i in range(n_template_view):
            all_tem_rgb[i] = torch.stack(all_tem_rgb[i]).to(device, non_blocking=True)
            all_tem_choose[i] = torch.stack(all_tem_choose[i]).to(device, non_blocking=True)
            all_tem_pts[i] = torch.stack(all_tem_pts[i]).to(device, non_blocking=True)

        return all_tem_rgb, all_tem_pts, all_tem_choose

    def get_template(self, file_base, cls_name, tem_index=0):
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

    def get_models(self, file_base, cls_name):
        mesh_path = os.path.join(file_base, cls_name + '.obj')
        mesh = trimesh.load(mesh_path, force='mesh')
        model_pts, _ = trimesh.sample.sample_surface(mesh, self.n_sample_model_point)
        return model_pts.astype(np.float32)

    def get_train_data(self, index):
        # 获取当前训练图片ID及其所有标注
        image_id = self.image_ids[index]
        annotations = self.load_image_annotations(image_id)
        if len(annotations) == 0:
            return None

        # 随机选择一个有效标注
        valid_annotation = np.random.choice(annotations)
        cls_id = valid_annotation['category_id']
        cls_name = self.class_names[cls_id]
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
        tem1_rgb, tem1_choose, tem1_pts = self.get_template(self.templates_dir, cls_name, 6)
        tem2_rgb, tem2_choose, tem2_pts = self.get_template(self.templates_dir, cls_name, 30)
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
        depth = load_depth_image(depth_path).astype(np.float32)
        depth = depth * self.depth_scale / 1000.0 + 0.085
        pts = get_point_cloud_from_depth(depth, camera_k, [y1, y2, x1, x2])
        pts = pts.reshape(-1, 3)[choose, :]

        # 将点云转换到目标物体的坐标系
        target_pts = (pts - target_t[None, :]) @ target_R
        tem_pts = np.concatenate([tem1_pts, tem2_pts], axis=0)
        # visualize_point_cloud(target_pts, tem_pts)
        radius = np.max(np.linalg.norm(tem_pts, axis=1))
        target_radius = np.linalg.norm(target_pts, axis=1)
        flag = target_radius < radius * 1.2
        pts = pts[flag]
        choose = choose[flag]
        if len(choose) < 500:
            return None
        if len(choose) <= self.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        pts = pts[choose_idx]

        add_error = compute_add_error(pts, tem_pts)
        if add_error > 1.0:
            return None

        # rgb
        image_path = self.image_info[image_id]['path']
        rgb = load_color_image(image_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[y1:y2, x1:x2, :]
        if np.random.rand() < 0.8 and self.color_augmentor is not None:
            rgb = self.color_augmentor(image=rgb)["image"]
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(rgb)
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        # # 随机旋转增强
        # rand_R = get_random_rotation()
        # tem1_pts = tem1_pts @ rand_R
        # tem2_pts = tem2_pts @ rand_R
        # target_R = target_R @ rand_R
        #
        # # 随机平移增强
        # add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
        # target_t += add_t[0]
        # add_t += 0.001 * np.random.randn(1, 3)
        # pts += add_t

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
        # 获取当前训练图片ID及其所有标注
        image_id = self.image_ids[index]
        annotations = self.load_image_annotations(image_id)
        if len(annotations) == 0:
            return None

        # 随机选择一个有效标注
        valid_annotation = np.random.choice(annotations)
        cls_id = valid_annotation['category_id']
        cls_name = self.class_names[cls_id]
        annotation_id = valid_annotation['id']

        # 加载位姿信息（旋转矩阵R和平移向量t)
        target_R, target_t = self.load_pose_Rt(image_id, annotation_id)
        target_R = np.array(target_R).reshape(3, 3).astype(np.float32)
        target_t = np.array(target_t).reshape(3).astype(np.float32)

        # 加载相机内参矩阵
        camera_k = self.load_camera_k(image_id)
        if camera_k is None:
            return None
        camera_k = np.array(camera_k).reshape(3, 3).astype(np.float32)

        # 加载模型点云数据
        model_pts = self.get_models(self.model_dir, cls_name)
        if model_pts is None:
            return None

        # 加载目标物体mask
        mask = load_mask(valid_annotation, self.image_info[image_id]['height'], self.image_info[image_id]['width'])
        if mask is None or np.sum(mask) == 0:
            return None
        mask = (mask * 255).astype(np.uint8)
        bbox = get_bbox(mask > 0)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # 加载深度图数据
        depth_path = self.image_info[image_id]['depth_path']
        depth = load_depth_image(depth_path).astype(np.float32)
        depth = depth * self.depth_scale / 1000.0 + 0.085
        pts = get_point_cloud_from_depth(depth, camera_k, [y1, y2, x1, x2])
        pts = pts.reshape(-1, 3)[choose, :]

        add_error = compute_add_error(pts, model_pts)
        if add_error > 1.0:
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
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[y1:y2, x1:x2, :]
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(rgb)
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        ret_dict = {
            'pts': torch.FloatTensor(pts),
            'rgb': torch.FloatTensor(rgb),
            'rgb_choose': torch.IntTensor(rgb_choose).long(),
            'model': torch.FloatTensor(model_pts),
            # 'obj': torch.IntTensor([cls_id]).long(),
            'obj': torch.tensor(cls_id, dtype=torch.long),
            'translation_label': torch.FloatTensor(target_t),
            'rotation_label': torch.FloatTensor(target_R),
        }
        return ret_dict


if __name__ == "__main__":
    import random
    from easydict import EasyDict as edict
    import yaml
    from data.dataloader.build import build_dataloader
    from utils.visualize import visualize_point_cloud

    with open('../../cfg/indus.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = edict(cfg_dict)

    # 测试代码
    def test_posenet_dataset():
        print("开始测试PoseNetDataset加载...")

        train_dataset = PoseNetDataset(cfg.TRAIN_DATA, is_train=True)
        print(f"数据集加载完成，共有 {len(train_dataset)} 张训练图片")

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


    def test_point_cloud(is_train: bool = True):
        if is_train:
            dataset = PoseNetDataset(cfg.TRAIN_DATA, is_train=True)

            print(f"数据集加载完成，共有 {len(dataset)} 张训练图片")
            index = random.randint(0, len(dataset) - 1)
            print(f"随机选取样本索引: {index}")

            data = dataset.get_train_data(index)

            if data is None:
                print(f"样本 {index} 加载失败，可能数据不足或标注缺失")

            pts = data['pts'].cpu().numpy()
            target_t = data['translation_label'].cpu().numpy().astype(np.float64)
            target_R = data['rotation_label'].cpu().numpy().astype(np.float64)

            tem1_pts = data['tem1_pts'].cpu().numpy().astype(np.float64)
            tem2_pts = data['tem2_pts'].cpu().numpy().astype(np.float64)

            # 使用给定的旋转和平移对 pts 进行变换
            target_pts = (pts - target_t[None, :]) @ target_R

            # 合并 tem1_pts 和 tem2_pts
            tem_pts = np.concatenate([tem1_pts, tem2_pts], axis=0)
            visualize_point_cloud(target_pts, tem_pts)

        else:
            dataset = PoseNetDataset(cfg.TEST_DATA, is_train=False)

            print(f"数据集加载完成，共有 {len(dataset)} 张训练图片")
            index = random.randint(0, len(dataset) - 1)
            print(f"随机选取样本索引: {index}")

            data = dataset.get_test_data(index)

            if data is None:
                print(f"样本 {index} 加载失败，可能数据不足或标注缺失")

            pts = data['pts'].cpu().numpy()
            target_t = data['translation_label'].cpu().numpy().astype(np.float64)
            target_R = data['rotation_label'].cpu().numpy().astype(np.float64)
            model_pts = data['model'].cpu().numpy().astype(np.float64)

            # 使用给定的旋转和平移对 pts 进行变换
            target_pts = (pts - target_t[None, :]) @ target_R

            visualize_point_cloud(target_pts, model_pts)


    test_point_cloud(is_train=False)
    # test_posenet_dataset()
