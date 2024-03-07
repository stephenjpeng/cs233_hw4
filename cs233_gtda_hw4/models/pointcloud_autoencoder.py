"""
PC-AE.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn
from tqdm.autonotebook import tqdm
from ..in_out.utils import AverageMeter
# from ..losses.chamfer import chamfer_loss

# In the unlikely case where you cannot use the JIT chamfer implementation (above) you can use the slower
# one that is written in pure pytorch:
from ..losses.nn_distance import chamfer_loss


class PointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """ AE constructor.
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, pointclouds):
        """Forward pass of the AE
            :param pointclouds: B x N x 3
        """
        x = self.encoder(pointclouds)
        x = self.decoder(x)
        return x
        

    def train_for_one_epoch(self, loader, optimizer, device='cuda'):
        """ Train the autoencoder for one epoch based on the Chamfer loss.
        :param loader: (train) pointcloud_dataset loader
        :param optimizer: torch.optimizer
        :param device: cuda? cpu?
        :return: (float), average loss for the epoch.
        """
        self.train()
        loss_meter = AverageMeter()

        for load in tqdm(loader):
            optimizer.zero_grad()

            pointclouds = load['point_cloud'].to(device)
            reconstructions = self.forward(pointclouds)
            loss = chamfer_loss(pointclouds, reconstructions).sum()

            loss.backward()
            optimizer.step()

            loss_meter.update(loss, pointclouds.shape[0])

        return loss_meter.avg
    
    @torch.no_grad()
    def embed(self, pointclouds):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :return: B x latent-dimension of AE
        """
        return self.encoder(pointclouds)
        

    @torch.no_grad()
    def reconstruct(self, loader, device='cuda'):
        """ Reconstruct the point-clouds via the AE.
        :param loader: pointcloud_dataset loader
        :param device: cpu? cuda?
        :return: (reconstructions, float), average loss for the loader
        """
        self.eval()
        loss_meter = AverageMeter()

        reconstructions = []
        for load in tqdm(loader):
            pointclouds = load['point_cloud'].to(device)
            reconstruction = self.forward(pointclouds)
            loss = chamfer_loss(pointclouds, reconstruction).sum()

            loss_meter.update(loss, pointclouds.shape[0])
            reconstructions += list(reconstruction)

        return reconstructions, loss_meter.avg

    @torch.no_grad()
    def reconstruct_single(self, pointcloud, device='cuda'):
        """ Reconstruct the point-cloud via the AE.
        :param pointcloud: pointcloud_data
        :param device: cpu? cuda?
        :return: (reconstruction, float), average loss for the loader
        """
        r, l = self.reconstruct([{'point_cloud': pointcloud.unsqueeze(0)}], device)
        return r[0].squeeze(0), l
