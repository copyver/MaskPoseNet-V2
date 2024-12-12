import torch
import torch.nn as nn
from models.pose.feature_extraction import ViTEncoder
from models.pose.coarse_point_matching import CoarsePointMatching
from models.pose.fine_point_matching import FinePointMatching
from models.pose.transformer import GeometricStructureEmbedding
from models.pose.model_utils import sample_pts_feats


class PoseModel(nn.Module):
    def __init__(self, cfg):
        super(PoseModel, self).__init__()
        self.cfg = cfg
        self.coarse_npoint = cfg.COARSE_NPOINT
        self.fine_npoint = cfg.FINE_NPOINT

        self.feature_extraction = ViTEncoder(cfg.FEATURE_EXTRACTION, cfg.FINE_NPOINT)
        self.geo_embedding = GeometricStructureEmbedding(cfg.GEO_EMBEDDING)
        self.coarse_point_matching = CoarsePointMatching(cfg.COARSE_POINT_MATCHING)
        self.fine_point_matching = FinePointMatching(cfg.FINE_POINT_MATCHING)

    def forward(self, end_points):
        dense_pm, dense_fm, dense_po, dense_fo, radius = self.feature_extraction(end_points)

        # pre-compute geometric embeddings for geometric transformer
        bg_point = torch.ones(dense_pm.size(0), 1, 3).float().to(dense_pm.device) * 100

        sparse_pm, sparse_fm, fps_idx_m = sample_pts_feats(
            dense_pm, dense_fm, self.coarse_npoint, return_index=True
        )
        geo_embedding_m = self.geo_embedding(torch.cat([bg_point, sparse_pm], dim=1))

        sparse_po, sparse_fo, fps_idx_o = sample_pts_feats(
            dense_po, dense_fo, self.coarse_npoint, return_index=True
        )
        geo_embedding_o = self.geo_embedding(torch.cat([bg_point, sparse_po], dim=1))

        # coarse_point_matching
        end_points = self.coarse_point_matching(
            sparse_pm, sparse_fm, geo_embedding_m,
            sparse_po, sparse_fo, geo_embedding_o,
            radius, end_points,
        )

        # fine_point_matching
        end_points = self.fine_point_matching(
            dense_pm, dense_fm, geo_embedding_m, fps_idx_m,
            dense_po, dense_fo, geo_embedding_o, fps_idx_o,
            radius, end_points
        )

        return end_points


if __name__ == '__main__':
    from easydict import EasyDict as edict
    import yaml
    from data.dataloader.build import build_dataloader
    from data.dataset.posenet_dataset import PoseNetDataset

    with open('cfg/base.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = edict(cfg_dict)

    # 创建PoseNetDataset实例
    train_dataset = PoseNetDataset(cfg.TRAIN_DATA, is_train=True)
    print(f"数据集加载完成，共有 {len(train_dataset)} 张训练图片")

    train_dataloader = build_dataloader(train_dataset, batch=cfg.TRAIN_DATA.BATCH_SIZE,
                                        workers=cfg.TRAIN_DATA.WORKERS, shuffle=True)

    model = PoseModel(cfg.POSE_MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("测试批量加载并运行模型...")
    for batch_index, batch_data in enumerate(train_dataloader):
        print(f"Batch {batch_index + 1}: 数据类型: {type(batch_data)}")
        print(f"Batch {batch_index + 1}: 数据内容: {batch_data.keys()}")
        print(f"Batch {batch_index + 1}: 样本数量: {len(next(iter(batch_data.values())))}")
        if len(batch_data) > 0:
            for key, value in batch_data.items():
                print(f" - Key: {key}, shape: {value.shape}")

            # 准备模型输入
            # 将 batch_data 中的所有张量移动到设备上
            end_points = {}
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    end_points[key] = value.to(device)
                else:
                    # 如果不是张量，可能需要特殊处理
                    pass

            # 运行模型前向传播
            output = model(end_points)

            # 打印输出
            print("模型输出:")
            print(output)

            # 由于只是测试，只运行一个批次
            break
