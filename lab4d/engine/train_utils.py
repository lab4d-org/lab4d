# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

import torch


def get_local_rank():
    try:
        return int(os.environ["LOCAL_RANK"])
    except:
        # print("LOCAL_RANK not found, set to 0")
        return 0


def match_param_name(name, param_lr, type):
    """
    Match the param name with the param_lr dict

    Args:
        name (str): the name of the param
        param_lr (Dict): the param_lr dict
        type (str): "with" or "startwith"

    Returns:
        bool, lr
    """
    matched_param = []
    matched_lr = []

    for params_name, lr in param_lr.items():
        if type == "with":
            if params_name in name:
                matched_param.append(params_name)
                matched_lr.append(lr)
        elif type == "startwith":
            if name.startswith(params_name):
                matched_param.append(params_name)
                matched_lr.append(lr)
        else:
            raise ValueError("type not found")

    if len(matched_param) == 0:
        return False, 0.0
    elif len(matched_param) == 1:
        return True, matched_lr[0]
    else:
        print("multiple matches found for %s" % name)
        raise ValueError(matched_param)


class DataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    """For multi-GPU access, forward attributes to the inner module."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __delattr__(self, name):
        try:
            return super().__delattr__(name)
        except AttributeError:
            return delattr(self.module, name)
