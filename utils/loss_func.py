import torch
from .losses.dice import dice_coeff_batch
from .losses.bce import BCE_loss


def bce_dice(input: torch.Tensor, target: torch.Tensor):
    """
    return loss as dictionary form
    """
    dice = 1 - dice_coeff_batch(input, target)
    bce = BCE_loss(input, target)
    loss_dict = {
        "Dice Loss": dice,
        "Binary Cross Entropy": bce,
    }
    return loss_dict
