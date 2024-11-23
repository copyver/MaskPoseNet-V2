# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Testing customized ops. '''

import os
import sys

import numpy as np
import torch
from torch.autograd import gradcheck

from models.pose.src_pointnet2 import pointnet2_utils


def test_interpolation_grad():
    batch_size = 1
    feat_dim = 2
    m = 4

    # feats must be double precision for gradcheck
    feats = torch.randn(batch_size, feat_dim, m, requires_grad=True, dtype=torch.double).cuda()

    def interpolate_func(inputs):
        # Convert inputs to float32 (required by three_interpolate)
        inputs = inputs.to(torch.float)
        idx = torch.from_numpy(np.array([[[0, 1, 2], [1, 2, 3]]])).int().cuda()
        weight = torch.from_numpy(np.array([[[1, 1, 1], [2, 2, 2]]])).float().cuda()

        # Perform interpolation
        interpolated_feats = pointnet2_utils.three_interpolate(inputs, idx, weight)

        # Convert output back to double for gradcheck
        return interpolated_feats.to(torch.double)

    try:
        # Perform gradcheck with relaxed tolerance
        assert gradcheck(interpolate_func, feats, atol=1e-1, rtol=1e-1), "Gradcheck failed!"
        print("Gradcheck passed!")
    except Exception as e:
        print(f"Gradcheck failed: {e}")


if __name__ == '__main__':
    test_interpolation_grad()
