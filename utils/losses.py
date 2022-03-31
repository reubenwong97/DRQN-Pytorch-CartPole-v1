import torch
import torch.functional as F


def cosine_loss(x1, x2):
    criterion = torch.nn.CosineSimilarity(dim=1)
    loss = -criterion(x1, x2).mean()

    return loss


def L1_loss(x1, x2):
    loss = F.smooth_l1_loss(x1, x2)

    return loss
