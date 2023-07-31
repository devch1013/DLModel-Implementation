import torch
from torch import optim
from .loss_func import bce_dice
from losses.dice import dice_coeff_batch


def get_optimizer(model, cfg):
    """
    Return torch optimizer

    Args:
        model: Model you want to train
        cfg: Dictionary of optimizer configuration

    Returns:
        optimizer
    """
    optim_name = cfg["name"].lower()
    args = cfg["args"]
    print("Optimizer: ", optim_name, " lr=", args["lr"])
    optim_dict = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    try:
        return optim_dict[optim_name](model.parameters(), **args)
    except:
        raise NotImplementedError


def get_scheduler(optimizer, cfg):
    """
    get ["lr_scheduler"] cfg dictionary
    """
    scheduler_name = cfg["name"].lower()
    print("Scheduler: ", scheduler_name)
    args = cfg["args"]
    scheduler_dict = {
        "plateau": optim.lr_scheduler.ReduceLROnPlateau,
        "multisteplr": optim.lr_scheduler.MultiStepLR,
    }
    try:
        return scheduler_dict[scheduler_name](optimizer=optimizer, **args)
    except:
        NotImplementedError


def get_criterion(cfg):
    """
    Return torch criterion

    Args:
        cfg: Dictionary of criterion configuration

    Returns:
        criterion
    """
    criterion_name = cfg["name"].lower()
    print("Criterion: ", criterion_name)
    criterion_dict = {
        "crossentropyloss": torch.nn.BCEWithLogitsLoss(),
        "bce-dice": bce_dice,
    }
    try:
        return criterion_dict[criterion_name]
    except:
        raise NotImplementedError


def get_metric(cfg):
    """
    Return metric function for validation or test
    """
    metric_name = cfg["name"].lower()
    metric_dict = {
        "dice": dice_coeff_batch,
    }
    try:
        return metric_dict[metric_name]
    except:
        NotImplementedError
