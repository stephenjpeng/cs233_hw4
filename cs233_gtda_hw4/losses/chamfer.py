from .ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
import torch

chamfer_raw = dist_chamfer_3D.chamfer_3DDist()

def chamfer_loss(pc_a, pc_b, mask=None):
    """ Compute the chamfer loss for batched pointclouds.
    :param pc_a: torch.Tensor B x Na-points per point-cloud x 3
    :param pc_b: torch.Tensor B x Nb-points per point-cloud x 3
    :param mask: torch.Tensor B x N-points per point-cloud
        - ONLY WORKS FOR PC_A AND BINARY WEIGHTS
    :return: B floats, indicating the chamfer distances
    """
    dist_a, dist_b, idx_a, idx_b = chamfer_raw(pc_a, pc_b)
    if mask is None:
        dist = dist_a.mean(1) + dist_b.mean(1) # reduce separately, sizes of points can be different
    else:
        denom = mask.sum(1)
        denom += torch.where(denom == 0, 1, 0)
        dist = (dist_a * mask).sum(1) / denom + (dist_b).mean(1)
        assert torch.all(torch.isfinite(dist))

    return dist
