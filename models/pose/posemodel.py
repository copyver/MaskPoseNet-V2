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
        self.coarse_npoint = cfg.coarse_npoint
        self.fine_npoint = cfg.fine_npoint

        self.feature_extraction = ViTEncoder(cfg.feature_extraction, self.fine_npoint)
        self.geo_embedding = GeometricStructureEmbedding(cfg.geo_embedding)
        self.coarse_point_matching = CoarsePointMatching(cfg.coarse_point_matching)
        self.fine_point_matching = FinePointMatching(cfg.fine_point_matching)


