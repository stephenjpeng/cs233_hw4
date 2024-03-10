"""
Fancy Part-Aware PC-AE.
"""

import numpy as np
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


class FancyPartAwarePointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, part_classifier, part_exister, part_lambda, device,
            class_decay=1, class_decay_cadence=50, decode_alpha=0.0,
            predict_parts=True, predict_part_exist=False,
            variational=False, kl_lambda=0, exist_lambda=0, kl_decay=1, kl_decay_cadence=50, noise=0,
            encode_parts=False):
        """ Part-aware AE initialization
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        :param part_classifier: nn.Module acting as the second decoding branch that classifies the point part
        :param part_lambda: scalar multiple for the part classification loss
        :param class_decay: scalar multiple to decay the classification loss contribution
        :param class_decay_cadence: decay class loss contribution every _ epochs
        :param decode_alpha: coefficient of L1 regularization on decoder
        :param predict_parts: predict parts
        :param exist_lambda: scalar multiple for the part existence loss
        :param predict_part_exist: predict a {0, 1}^4 vector for existence of each part
        :param variational: use variational AE
        :param vae_decay: scalar multiple to decay the KL divergence contribution
        :param vae_decay_cadence: decay KL loss contribution every _ epochs
        :param noise: variance of normal noise to add to inputs
        :param encode_parts: autoencode the parts
        """
        super().__init__()
        # not changing anytime soon
        self.num_parts = 4

        self.encoder = encoder
        self.decoder = decoder
        if predict_parts:
            self.part_classifier = part_classifier
        if predict_part_exist:
            self.part_exister = part_exister

        self.part_lambda = part_lambda
        self.class_decay = class_decay
        self.class_decay_cadence = class_decay_cadence

        self.decode_alpha = decode_alpha

        self.predict_parts = predict_parts
        self.exist_lambda = exist_lambda
        self.predict_part_exist = predict_part_exist

        self.variational = variational
        self.kl_lambda = kl_lambda
        self.kl_decay = kl_decay
        self.kl_decay_cadence = kl_decay_cadence
        self.noise = noise

        self.encode_parts = encode_parts

        self.device = device

        self.epoch = 0


    def forward(self, pointclouds):
        """Forward pass of the AE
            :param pointclouds: B x N x 3
            :return: (reconstruction, label)
        """
        B, N, _ = pointclouds.shape

        if self.training:
            pointclouds = pointclouds + self.noise * torch.randn(pointclouds.shape).to(self.device)
        h = self.encoder(pointclouds)  # B x latent_dim

        if self.variational:
            mu, logvar = torch.split(h, h.shape[-1] // 2, dim=-1) # each B x latent_dim / 2
            sigma = torch.exp(logvar / 2)
            if self.training:
                z = mu + sigma * torch.randn(sigma.shape).to(self.device)
            else:
                z = mu
        else:
            z = mu = h

        # AE branch
        x = self.decoder(z, reshape_3d=True)  # B x out_pts x 3

        # part prediction branch
        if self.predict_parts:
            class_input = torch.cat((
                pointclouds,
                mu.unsqueeze(1).expand(-1, N, -1)
            ), dim=2)
            y = self.part_classifier(class_input)  # (B, C, N)

        # part existence prediction branch
        if self.predict_part_exist:
            w = self.part_exister(mu)

        res = [x]
        if self.predict_parts:
            res += [y]

        if self.predict_part_exist:
            res += [w]

        if self.variational:
            res += [mu, sigma]
        return res

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
        bce_loss_meter = AverageMeter()
        kl_loss_meter = AverageMeter()
        self.epoch += 1

        for load in loader:
            optimizer.zero_grad()

            pointclouds = load['point_cloud'].to(device)
            true_label = load['part_mask'].to(device) # .flatten()

            output = self.forward(pointclouds)
            reconstruction = output[0]
            idx = 1
            if self.predict_parts:
                label = output[idx]
                idx += 1
            if self.predict_part_exist:
                existence = output[idx]
                idx += 1
            if self.variational:
                mu, sigma = output[idx:idx+2]

            if self.encode_parts:
                # use out_points / 4 points to construct each part
                recon_loss = 0
                recon_parts = torch.chunk(reconstruction,
                        reconstruction.shape[-2] // self.num_parts,
                        -2)
                for part in range(self.num_parts):
                    true_pc = (pointclouds * (true_label==part).unsqueeze(-1).expand(-1, -1, 3)
                        + 0.5 * (true_label!=part).unsqueeze(-1).expand(-1, -1, 3))
                    recon_loss += chamfer_loss(true_pc, recon_parts[part]).mean()
            else:
                recon_loss = chamfer_loss(pointclouds, reconstruction).mean()

            if self.predict_parts:
                xentr_loss = F.cross_entropy(label, true_label, reduction='none')
                xentr_loss = xentr_loss.mean()
            else:
                xentr_loss = 0

            if self.predict_part_exist:
                target = torch.vstack(
                        [(true_label==i).max(axis=-1)[0] for i in range(self.num_parts)]
                    ).T.float()
                bce_loss = F.binary_cross_entropy_with_logits(existence, target, reduction='mean')
            else:
                bce_loss = 0


            if self.variational:
                kl_loss = - (1 + (torch.log(sigma) * 2) - torch.pow(mu, 2) - sigma * 2).mean() / 2
            else:
                kl_loss = 0

            loss = recon_loss + \
                    self.part_lambda * xentr_loss + \
                    self.exist_lambda * bce_loss + \
                    self.kl_lambda * kl_loss + \
                    self.decode_alpha * self.decoder.l1_loss()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss, pointclouds.shape[0])
            recon_loss_meter.update(recon_loss, pointclouds.shape[0])
            xentr_loss_meter.update(xentr_loss, pointclouds.shape[0])
            bce_loss_meter.update(bce_loss, pointclouds.shape[0])
            kl_loss_meter.update(kl_loss, pointclouds.shape[0])

        if self.epoch % self.class_decay_cadence == 0:
            self.part_lambda *= self.class_decay
        if self.epoch % self.kl_decay_cadence == 0:
            self.kl_lambda *= self.kl_decay
        return loss_meter.avg, recon_loss_meter.avg, xentr_loss_meter.avg, bce_loss_meter.avg, kl_loss_meter.avg
    
    @torch.no_grad()
    def embed(self, pointclouds):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :return: B x latent-dimension of AE
        """
        h = self.encoder(pointclouds)
        if self.variational:  # return mean of distribution
            return torch.split(h, h.shape[-1] // 2, -1)[0]
        return h
        

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
        bce_loss_meter = AverageMeter()
        kl_loss_meter = AverageMeter()

        reconstructions = []
        labels = []
        if return_all_recon_loss:
            all_recon_loss = []

        for load in loader:
            pointclouds = load['point_cloud'].to(device)
            true_label = load['part_mask'].to(device) 

            output = self.forward(pointclouds)
            reconstruction = output[0]
            idx = 1
            if self.predict_parts:
                label = output[idx]
                idx += 1
            if self.predict_part_exist:
                existence = output[idx]
                idx += 1
            if self.variational:
                mu, sigma = output[idx:idx+2]

            if self.encode_parts:
                # use out_points / 4 points to construct each point
                recon_losses = 0
                recon_parts = torch.chunk(reconstruction,
                        reconstruction.shape[-2] // self.num_parts,
                        -2)
                for part in range(self.num_parts):
                    true_pc = (pointclouds * (true_label==part).unsqueeze(-1).expand(-1, -1, 3)
                        + 0.5 * (true_label!=part).unsqueeze(-1).expand(-1, -1, 3))
                    recon_losses += chamfer_loss(true_pc, recon_parts[part])
            else:
                recon_losses = chamfer_loss(pointclouds, reconstruction)
            recon_loss = recon_losses.mean()


            if self.predict_parts:
                xentr_loss = F.cross_entropy(label, true_label, reduction='none')
                xentr_loss = xentr_loss.mean()
            else:
                xentr_loss = 0

            if self.predict_part_exist:
                target = torch.vstack(
                        [(true_label==i).max(axis=-1)[0] for i in range(self.num_parts)]
                    ).T.float()
                bce_loss = F.binary_cross_entropy_with_logits(existence, target, reduction='mean')
            else:
                bce_loss = 0

            if self.variational:
                kl_loss = - (1 + 2 * torch.log(sigma) - (torch.pow(mu, 2) + torch.pow(sigma, 2))).mean() / 2
            else:
                kl_loss = 0

            loss = recon_loss + \
                    self.part_lambda * xentr_loss + \
                    self.kl_lambda * kl_loss + \
                    self.exist_lambda * bce_loss + \
                    self.decode_alpha * self.decoder.l1_loss()

            loss_meter.update(loss, pointclouds.shape[0])
            recon_loss_meter.update(recon_loss, pointclouds.shape[0])
            xentr_loss_meter.update(xentr_loss, pointclouds.shape[0])
            bce_loss_meter.update(bce_loss, pointclouds.shape[0])
            kl_loss_meter.update(kl_loss, pointclouds.shape[0])

            reconstructions += list(reconstruction)
            labels += list(label)
            if return_all_recon_loss:
                all_recon_loss += list(recon_losses)
        labels = torch.cat(labels)
        if return_all_recon_loss:
            return (reconstructions, labels,
                    loss_meter.avg, recon_loss_meter.avg, xentr_loss_meter.avg,
                    bce_loss_meter.avg, kl_loss_meter.avg, all_recon_loss)
        return (reconstructions, labels,
                loss_meter.avg, recon_loss_meter.avg,
                xentr_loss_meter.avg, bce_loss_meter.avg, kl_loss_meter.avg)

    @torch.no_grad()
    def reconstruct_single(self, pointcloud, true_labels, device='cuda', return_logits=False):
        """ Reconstruct the point-cloud via the AE.
        :param pointcloud: pointcloud_data
        :param device: cpu? cuda?
        :return: (reconstruction, float), average loss for the loader
        """
        out = self.reconstruct([{
            'point_cloud': pointcloud.unsqueeze(0),
            'part_mask': true_labels.unsqueeze(0),
            }], device)

        r = out[0]

        if self.predict_parts:
            l = out[1]
            if not return_logits:
                l = torch.argmax(l, 0)
        else:
            l = None
        return r[0].squeeze(0), l
