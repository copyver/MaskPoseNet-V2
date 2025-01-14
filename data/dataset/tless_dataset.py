import albumentations as A
from data.dataset.base_dataset import DatasetBase
import os
import torchvision.transforms as transforms
from loguru import logger
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
import torch
import cv2
import numpy as np
import trimesh
from utils.visualize import visualize_point_cloud


class TlessDataset(DatasetBase):
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
        self.image_len_info = {}
        if is_train:
            self.train_image_info = None
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
                    # 替代 Brightness/Contrast
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                ], p=0.8),
                A.InvertImg(p=0.3),
            ], p=1.0)
            self.load_data(is_train)

        else:
            self.test_image_info = None
            self.n_sample_model_point = cfg.N_SAMPLE_MODEL_POINT
            self.model_dir = os.path.join(self.data_dir, cfg.MODEL_DIR)
            self.n_template_view = cfg.N_TEMPLATE_VIEW
            self.color_augmentor = None
            self.load_data(is_train)

    def load_data(self, is_train=True):
        """
        遍历新数据集目录结构:
        """
        # 列出所有场景ID文件夹
        scene_ids = [
            d for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
        ]
        logger.info(f"Found {len(scene_ids)} scenes in {self.dataset_dir}.")

        # 遍历每个场景文件夹
        for scene_id in scene_ids:
            scene_camera_data = load_anns(self.dataset_dir, scene_id, "scene_camera.json")
            scene_instances_gt_data = load_anns(self.dataset_dir, scene_id, "scene_gt_coco.json")
            scene_pose_gt_data = load_anns(self.dataset_dir, scene_id, "scene_gt.json")

            self.scene_camera_info.update({scene_id: scene_camera_data})
            self.scene_instances_gt_info.update({scene_id: scene_instances_gt_data})
            self.scene_pose_gt_info.update({scene_id: scene_pose_gt_data})
            self.image_len_info.update({scene_id: len(scene_instances_gt_data["images"])})

        cats = scene_instances_gt_data.get('categories', [])
        class_ids = [cat['id'] for cat in cats]
        for i in class_ids:
            self.add_class(source="tless", class_id=i, class_name=f"obj_{i:06d}")

        # 添加图像信息
        for scene_id in scene_ids:
            image_count = self.image_len_info[scene_id]
            start_idx = 0 if is_train else max(0, image_count - 100)
            end_idx = image_count - 100 if is_train else image_count

            for image_id in range(start_idx, end_idx):
                image_info = self.scene_instances_gt_info[scene_id]["images"][image_id]
                annotations = self.scene_instances_gt_info[scene_id]["annotations"][image_id]
                image_file = image_info["file_name"].split("/")[-1]

                self.add_image(
                    source="tless",
                    id=int(scene_id) * 1296 + image_id,
                    path=os.path.join(self.dataset_dir, scene_id, "rgb", image_file),
                    scene_id=scene_id,
                    image_id=image_id,
                    depth_path=os.path.join(self.dataset_dir, scene_id, "depth", image_file),
                    width=image_info["width"],
                    height=image_info["height"],
                    annotations=annotations,
                )

        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)
        logger.info(f"Number of images: {self.num_images}")
        logger.info(f"Number of classes: {self.num_classes}")

    def add_train_image_info(self, source: str, path: str, **kwargs):
        image_info = {
            "id": id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.train_image_info.append(image_info)

    def add_val_image_info(self, source: str, path: str, **kwargs):
        image_info = {
            "id": id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.test_image_info.append(image_info)

    def load_pose_Rt(self, scene_id: str, image_id: int):
        pose_dict = self.scene_pose_gt_info[scene_id][str(image_id)][0]
        return pose_dict["cam_R_m2c"], pose_dict["cam_t_m2c"]

    def load_camera_k(self, scene_id: str, image_id: int):
        cam_dict = self.scene_camera_info[scene_id][str(image_id)]
        return cam_dict["cam_K"], cam_dict["depth_scale"]

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
        # xyz = convert_blender_to_pyrender(xyz)[y1:y2, x1:x2, :]
        xyz = xyz[y1:y2, x1:x2, :]
        xyz = xyz.reshape((-1, 3))[choose, :] * 0.001
        choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        return rgb, choose, xyz

    def get_models(self, file_base, cls_name):
        mesh_path = os.path.join(file_base, cls_name + '.ply')
        mesh = trimesh.load(mesh_path, force='mesh')
        model_pts, _ = trimesh.sample.sample_surface(mesh, self.n_sample_model_point)
        return model_pts.astype(np.float32) / 1000.0

    def get_train_data(self, index):
        random_image_info = self.image_info[index]
        scene_id = random_image_info["scene_id"]
        annotations = random_image_info["annotations"]
        image_id = random_image_info["image_id"]
        if len(annotations) == 0:
            return None

        cls_id = annotations['category_id']
        cls_name = self.class_names[cls_id]

        # 加载位姿信息（旋转矩阵R和平移向量t
        target_R, target_t = self.load_pose_Rt(scene_id, image_id)
        target_R = np.array(target_R).reshape(3, 3).astype(np.float32)
        target_t = np.array(target_t).reshape(3).astype(np.float32) / 1000.0

        # 加载相机内参矩阵
        camera_k, depth_scale = self.load_camera_k(scene_id, image_id)
        if camera_k is None:
            return None
        camera_k = np.array(camera_k).reshape(3, 3).astype(np.float32)

        # 加载模板数据（两种不同视角）
        tem1_rgb, tem1_choose, tem1_pts = self.get_template(self.templates_dir, cls_name, 6)
        tem2_rgb, tem2_choose, tem2_pts = self.get_template(self.templates_dir, cls_name, 30)
        if tem1_rgb is None or tem2_rgb is None:
            return None

        # 加载目标物体mask
        mask = load_mask(annotations, annotations['height'], annotations['width'])
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
        depth_path = random_image_info["depth_path"]
        depth = load_depth_image(depth_path).astype(np.float32)
        depth = depth / 50.0 + 0.15
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

        # rgb
        image_path = random_image_info['path']
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
        # 同理
        random_image_info = self.image_info[index]
        scene_id = random_image_info["scene_id"]
        annotations = random_image_info["annotations"]
        image_id = random_image_info["image_id"]
        if len(annotations) == 0:
            return None

        cls_id = annotations['category_id']
        cls_name = self.class_names[cls_id]

        # 加载位姿信息（旋转矩阵R和平移向量t
        target_R, target_t = self.load_pose_Rt(scene_id, image_id)
        target_R = np.array(target_R).reshape(3, 3).astype(np.float32)
        target_t = np.array(target_t).reshape(3).astype(np.float32) / 1000.0

        # 加载相机内参矩阵
        camera_k, depth_scale = self.load_camera_k(scene_id, image_id)
        if camera_k is None:
            return None
        camera_k = np.array(camera_k).reshape(3, 3).astype(np.float32)

        # 加载模型点云数据
        model_points = self.get_models(self.model_dir, cls_name)
        if model_points is None:
            return None

        # 加载目标物体mask
        mask = load_mask(annotations, annotations['height'], annotations['width'])
        if mask is None or np.sum(mask) == 0:
            return None
        mask = (mask * 255).astype(np.uint8)
        bbox = get_bbox(mask > 0)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # 加载深度图数据
        depth_path = random_image_info["depth_path"]
        depth = load_depth_image(depth_path).astype(np.float32)
        depth = depth / 50.0 + 0.15
        pts = get_point_cloud_from_depth(depth, camera_k, [y1, y2, x1, x2])
        pts = pts.reshape(-1, 3)[choose, :]

        # target_pts = (pts - target_t[None, :]) @ target_R
        # visualize_point_cloud(target_pts, model_points)

        if len(choose) <= self.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        pts = pts[choose_idx]

        # rgb
        image_path = random_image_info['path']
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

        ret_dict = {
            'pts': torch.FloatTensor(pts),
            'rgb': torch.FloatTensor(rgb),
            'rgb_choose': torch.IntTensor(rgb_choose).long(),
            'model': torch.FloatTensor(model_points),
            'obj': torch.tensor(cls_id, dtype=torch.long),
            'translation_label': torch.FloatTensor(target_t),
            'rotation_label': torch.FloatTensor(target_R),
        }
        return ret_dict


if __name__ == '__main__':
    import random
    from easydict import EasyDict as edict
    import yaml

    with open('../../cfg/tless.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = edict(cfg_dict)

    def test_point_cloud(is_train: bool = True):
        from utils.visualize import visualize_point_cloud
        if is_train:
            dataset = TlessDataset(cfg.TRAIN_DATA, is_train=True)

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
            dataset = TlessDataset(cfg.TEST_DATA, is_train=False)

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


    test_point_cloud(is_train=False,)
