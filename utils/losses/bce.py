import torch.nn as nn


def BCE_loss(pred, label, reduction="mean"):
    bce_loss = nn.BCELoss(reduction=reduction)
    bce_out = bce_loss(pred, label)
    return bce_out
