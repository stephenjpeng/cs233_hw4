"""
Multi-layer perceptron.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn


class MLP(nn.Module):
    """ Multi-layer perceptron. That is a k-layer deep network where each layer is a fully-connected (nn.Linear) layer, with
    (optionally) batch-norm, a non-linearity and dropout.

    Students: again, you can use this scaffold to make a generic MLP that can be used with multiple-hyper parameters
    or, opt for a perhaps simpler custom variant that just does so for HW4. For HW4 do not use batch-norm, drop-out
    or other non-requested features, for the non-bonus question.
    """

    def __init__(self, in_feat_dim, out_channels, b_norm=False, dropout_rate=0, non_linearity=nn.ReLU(inplace=True)):
        """Constructor
        :param in_feat_dim: input feature dimension
        :param out_channels: list of ints describing each the number hidden/final neurons.
        :param b_norm: True/False, or list of booleans
        :param dropout_rate: int, or list of int values
        :param non_linearity: nn.Module
        """
        super().__init__()
        # parameters
        self.k = len(out_channels) + 1
        self.b_norm = b_norm if isinstance(b_norm, list) else [b_norm] * len(out_channels) + [False]
        self.c_dims = [in_feat_dim] + out_channels
        assert not self.b_norm[-1], "Batch norm applied after output!"

        # modules
        self.nonlin = non_linearity

        self.layers = nn.ParameterList()
        self.layers.append(nn.Dropout(dropout_rate))
        for i in range(1, self.k - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.c_dims[i-1], self.c_dims[i]),
                    nn.BatchNorm1d(self.c_dims[i]) if self.b_norm[i] else nn.Identity(),
                    self.nonlin,
                )
            )
        # output
        self.layers.append(
            nn.Linear(self.c_dims[-2], self.c_dims[-1])
        )



    def forward(self, x, reshape_3d=False):
        """
        Run forward pass of MLP
        :param x: (B x in_feat_dim) point cloud
        """
        B, k = x.shape
        for layer in self.layers:
            x = layer(x)
        if reshape_3d:
            return x.view(B, -1, 3)
        else:
            return x


    def l1_loss(self):
        """
        Calculate L1 of parameters
        """
        params = torch.cat([x.view(-1) for x in self.parameters()])
        return torch.norm(params, 1)
