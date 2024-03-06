"""
Point-Net.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, init_feat_dim=3, conv_dims=[32, 64, 64, 128, 128], student_optional_hyper_param=None):
        """
        Students:
        You can make a generic function that instantiates a point-net with arbitrary hyper-parameters,
        or go for an implemetnations working only with the hyper-params of the HW.
        Do not use batch-norm, drop-out and other not requested features.
        Just nn.Linear/Conv1D/ReLUs and the (max) poolings.
        
        :param init_feat_dim: input point dimensionality (default 3 for xyz)
        :param conv_dims: output point dimensionality of each layer
        """
        super().__init__()
        self.dims = [init_feat_dim] + conv_dims
        self.k = len(self.dims)

        self.convs = nn.ParameterList()
        for i in range(1, self.k):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(self.dims[i-1], self.dims[i], 1),
                    nn.ReLU() if i < (self.k - 1) else nn.Identity()
                )
            )
        self.pool = nn.MaxPool1d(1)
        
        
    def forward(self, pointclouds):
        """
        Run forward pass of the PointNet model on a given point cloud.
        :param pointclouds: (B x N x 3) point cloud
        """
        x = pointclouds
        for layer in self.convs:
            x = layer(x)
        x = x.permute((0, 2, 1))
        return self.pool(x)
