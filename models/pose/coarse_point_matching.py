import torch
import torch.nn as nn

from models.pose.transformer import GeometricTransformer
from models.pose.model_utils import (
    compute_feature_similarity,
    aug_pose_noise,
    compute_coarse_Rt,
)
from models.pose.loss_utils import compute_correspondence_loss


class CoarsePointMatching(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super(CoarsePointMatching, self).__init__()
        self.cfg = cfg
        self.return_feat = return_feat
        self.nblock = self.cfg.NBLOCK

        self.in_proj = nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_DIM)
        self.out_proj = nn.Linear(cfg.HIDDEN_DIM, cfg.OUT_DIM)

        self.bg_token = nn.Parameter(torch.randn(1, 1, cfg.HIDDEN_DIM) * .02)

        self.transformers = []
        for _ in range(self.nblock):
            self.transformers.append(GeometricTransformer(
                blocks=['self', 'cross'],
                d_model=cfg.HIDDEN_DIM,
                num_heads=4,
                dropout=None,
                activation_fn='ReLU',
                return_attention_scores=False,
            ))
        self.transformers = nn.ModuleList(self.transformers)

    def forward(self, p1, f1, geo1, p2, f2, geo2, radius, end_points):
        B = f1.size(0)

        f1 = self.in_proj(f1)
        f1 = torch.cat([self.bg_token.repeat(B, 1, 1), f1], dim=1)  # adding bg
        f2 = self.in_proj(f2)
        f2 = torch.cat([self.bg_token.repeat(B, 1, 1), f2], dim=1)  # adding bg

        atten_list = []
        for idx in range(self.nblock):
            f1, f2 = self.transformers[idx](f1, geo1, f2, geo2)

            if self.training or idx == self.nblock - 1:
                atten_list.append(compute_feature_similarity(
                    self.out_proj(f1),
                    self.out_proj(f2),
                    self.cfg.SIM_TYPE,
                    self.cfg.TEMP,
                    self.cfg.NORMALIZE_FEAT
                ))

        if self.training:
            gt_R = end_points['rotation_label']
            gt_t = end_points['translation_label'] / (radius.reshape(-1, 1) + 1e-6)
            init_R, init_t = aug_pose_noise(gt_R, gt_t)

            end_points = compute_correspondence_loss(
                end_points, atten_list, p1, p2, gt_R, gt_t,
                dis_thres=self.cfg.LOSS_DIS_THRES,
                loss_str='coarse'
            )
        else:
            init_R, init_t = compute_coarse_Rt(
                atten_list[-1], p1, p2,
                end_points['model'] / (radius.reshape(-1, 1, 1) + 1e-6),
                self.cfg.NPROPOSAL1, self.cfg.NPROPOSAL2,
            )
        end_points['init_R'] = init_R
        end_points['init_t'] = init_t

        if self.return_feat:
            return end_points, self.out_proj(f1), self.out_proj(f2)
        else:
            return end_points
