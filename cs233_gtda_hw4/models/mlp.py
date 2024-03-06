"""
Multi-layer perceptron.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""


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
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ParameterList()
        for i in range(1, self.k):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.c_dims[i-1], self.c_dims[i]),
                    self.nonlin,
                    self.dropout if i < (self.k - 1) else nn.Identity(),
                    nn.BatchNorm1d(out_channels[i]) if self.b_norm[i] else nn.Identity()
                )
            )


        
    def forward(self, x):
        """
        Run forward pass of MLP
        :param x: (B x in_feat_dim) point cloud
        """
        for layer in self.layers:
            x = layer(x)
        return x
