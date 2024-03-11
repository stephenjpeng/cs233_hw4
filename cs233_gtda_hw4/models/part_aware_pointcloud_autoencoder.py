"""
Part-Aware PC-AE.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from ..in_out.utils import AverageMeter

if torch.cuda.is_available():
    from ..losses.chamfer import chamfer_loss
else:
    # In the unlikely case where you cannot use the JIT chamfer implementation (above) you can use the slower
    # one that is written in pure pytorch:
    from ..losses.nn_distance import chamfer_loss


class PartAwarePointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, part_classifier, part_lambda):
        """ Part-aware AE initialization
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        :param part_classifier: nn.Module acting as the second decoding branch that classifies the point part
        labels.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.part_classifier = part_classifier
        self.part_lambda = part_lambda


    def forward(self, pointclouds):
        """Forward pass of the AE
            :param pointclouds: B x N x 3
            :return: (reconstruction, label)
        """
        B, N, _ = pointclouds.shape
        h = self.encoder(pointclouds)  # B x latent_dim

        # AE branch
        x = self.decoder(h, reshape_3d=True)  # B x out_pts x 3

        # part prediction branch
        class_input = torch.cat((
            pointclouds,
            h.unsqueeze(1).expand(-1, N, -1)
        ), dim=2)
        y = self.part_classifier(class_input)  # (B, C, N)
        # y = y.permute(0, 2, 1)

        return x, y


    def train_for_one_epoch(self, loader, optimizer, device='cuda'):
        """ Train the autoencoder for one epoch based on the Chamfer loss.
        :param loader: (train) pointcloud_dataset loader
        :param optimizer: torch.optimizer
        :param device: cuda? cpu?
        :return: (float, float, float), average loss for the epoch.
        """
        self.train()
        loss_meter = AverageMeter()
        recon_loss_meter = AverageMeter()
        xentr_loss_meter = AverageMeter()

        for load in tqdm(loader):
            optimizer.zero_grad()

            pointclouds = load['point_cloud'].to(device)
            true_labels = load['part_mask'].to(device) # .flatten()
            reconstructions, labels = self.forward(pointclouds)

            recon_loss = chamfer_loss(pointclouds, reconstructions).mean()
            xentr_loss = F.cross_entropy(labels, true_labels, reduction='none')
            xentr_loss = xentr_loss.mean()

            loss = recon_loss + self.part_lambda * xentr_loss
            loss.backward()
            optimizer.step()

            loss_meter.update(loss, pointclouds.shape[0])
            recon_loss_meter.update(recon_loss, pointclouds.shape[0])
            xentr_loss_meter.update(xentr_loss, pointclouds.shape[0])

        return loss_meter.avg, recon_loss_meter.avg, xentr_loss_meter.avg
    
    @torch.no_grad()
    def embed(self, pointclouds):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :return: B x latent-dimension of AE
        """
        return self.encoder(pointclouds)
        

    @torch.no_grad()
    def reconstruct(self, loader, device='cuda', return_all_recon_loss=False):
        """ Reconstruct the point-clouds via the AE.
        :param loader: pointcloud_dataset loader
        :param device: cpu? cuda?
        :param return_all_recon_loss: whether to log and return all losses
        :return: (reconstructions, float), average loss for the loader
        """
        self.eval()
        loss_meter = AverageMeter()
        recon_loss_meter = AverageMeter()
        xentr_loss_meter = AverageMeter()

        reconstructions = []
        if return_all_recon_loss:
            all_recon_loss = []

        for load in loader:
            pointclouds = load['point_cloud'].to(device)
            true_labels = load['part_mask'].to(device) 
            reconstruction, labels = self.forward(pointclouds)

            recon_losses = chamfer_loss(pointclouds, reconstruction)
            recon_loss = recon_losses.mean()
            xentr_loss = F.cross_entropy(labels, true_labels, reduction='none')
            xentr_loss = xentr_loss.mean()

            loss = recon_loss + self.part_lambda * xentr_loss

            loss_meter.update(loss, pointclouds.shape[0])
            recon_loss_meter.update(recon_loss, pointclouds.shape[0])
            xentr_loss_meter.update(xentr_loss, pointclouds.shape[0])

            reconstructions += list(reconstruction)
            if return_all_recon_loss:
                all_recon_loss += list(recon_losses)

        if return_all_recon_loss:
            return reconstructions, labels, loss_meter.avg, recon_loss_meter.avg, xentr_loss_meter.avg, all_recon_loss
        return reconstructions, labels, loss_meter.avg, recon_loss_meter.avg, xentr_loss_meter.avg

    @torch.no_grad()
    def reconstruct_single(self, pointcloud, true_labels, device='cuda', return_logits=False):
        """ Reconstruct the point-cloud via the AE.
        :param pointcloud: pointcloud_data
        :param device: cpu? cuda?
        :return: (reconstruction, float), average loss for the loader
        """
        r, l, l1, l2, l3 = self.reconstruct([{
            'point_cloud': pointcloud.unsqueeze(0),
            'part_mask': true_labels.unsqueeze(0),
        }], device)

        if not return_logits:
            l = torch.argmax(l, 1)
        return r[0].squeeze(0), l, l1, l2, l3
