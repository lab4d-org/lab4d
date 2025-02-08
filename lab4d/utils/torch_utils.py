# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch
import torch.nn as nn


def reinit_model(model, std=1):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if hasattr(m.weight, "data"):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            if hasattr(m.bias, "data"):
                m.bias.data.zero_()


def flip_pair(tensor):
    """Flip the tensor along the pair dimension

    Args:
        tensor: (M*2, ...) Inputs [x0, x1, x2, x3, ..., x_{2k}, x_{2k+1}]

    Returns:
        tensor: (M*2, ...) Outputs [x1, x0, x3, x2, ..., x_{2k+1}, x_{2k}]
    """
    if torch.is_tensor(tensor):
        if len(tensor) < 2:
            return tensor
        return tensor.view(tensor.shape[0] // 2, 2, -1).flip(1).view(tensor.shape)
    elif isinstance(tensor, tuple):
        return tuple([flip_pair(t) for t in tensor])
    elif isinstance(tensor, dict):
        return {k: flip_pair(v) for k, v in tensor.items()}


def compute_gradients_sdf(fn, x, training=False, sdf=None, mode="numerical", eps=1e-3):
    """
    Taken from https://github.com/nvlabs/neuralangelo
    """
    x = x.detach()
    if mode == "analytical":
        requires_grad = x.requires_grad
        with torch.enable_grad():
            # 1st-order gradient
            x.requires_grad_(True)
            sdf = fn(x)
            gradient = torch.autograd.grad(sdf.sum(), x, create_graph=True)[0]
            # 2nd-order gradient (hessian)
            if training:
                pass
            else:
                gradient = gradient.detach()
        x.requires_grad_(requires_grad)
    elif mode == "numerical":
        k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
        k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
        k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
        k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
        sdf1 = fn(x + k1 * eps)  # [...,1]
        sdf2 = fn(x + k2 * eps)  # [...,1]
        sdf3 = fn(x + k3 * eps)  # [...,1]
        sdf4 = fn(x + k4 * eps)  # [...,1]
        gradient = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps)
    return gradient


@torch.enable_grad()
def compute_gradient(fn, x):
    """
    gradient of mlp params wrt pts
    """
    x.requires_grad_(True)
    y = fn(x)

    # get gradient for each size-1 output
    gradients = []
    for i in range(y.shape[-1]):
        y_sub = y[..., i : i + 1]
        d_output = torch.ones_like(y_sub, requires_grad=False, device=y.device)
        gradient = torch.autograd.grad(
            outputs=y_sub,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients.append(gradient[..., None])
    gradients = torch.cat(gradients, -1)  # ...,input-dim, output-dim
    return gradients


def frameid_to_vid(fid, frame_offset):
    """Given absolute frame ids [0, ..., N], compute the video id of each frame.

    Args:
        fid: (nframes,) Absolute frame ids
          e.g. [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203]
        frame_offset: (nvideos + 1,) Offset of each video
          e.g., [0, 100, 200, 300]
    Returns:
        vid: (nframes,) Maps idx to video id
        tid: (nframes,) Maps idx to relative frame id
    """
    vid = torch.zeros_like(fid)
    for i in range(frame_offset.shape[0] - 1):
        assign = torch.logical_and(fid >= frame_offset[i], fid < frame_offset[i + 1])
        vid[assign] = i
    return vid


def remove_ddp_prefix(state_dict):
    """Remove distributed data parallel prefix from model checkpoint

    Args:
        state_dict (Dict): Model checkpoint
    Returns:
        new_state_dict (Dict): New model checkpoint
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def remove_state_startwith(state_dict, prefix):
    """Remove model parameters that start with a prefix

    Args:
        state_dict (Dict): Model checkpoint
        prefix (str): Prefix to filter
    Returns:
        new_state_dict (Dict): New model checkpoint
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            continue
        else:
            new_state_dict[key] = value
    return new_state_dict


def remove_state_with(state_dict, string):
    """Remove model parameters that contain a string

    Args:
        state_dict (Dict): Model checkpoint
        string (str): String to filter
    Returns:
        new_state_dict (Dict): New model checkpoint
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if string in key:
            continue
        else:
            new_state_dict[key] = value
    return new_state_dict


def compress_state_with(state_dict, string):
    """Initialize model parameters with the mean of the instance embedding if
    the parameter name contains a string

    Args:
        state_dict (Dict): Model checkpoint, modified in place
        string (str): String to filter
    """
    # init with the mean of inst_embedding
    for key, value in state_dict.items():
        if string in key:
            state_dict[key] = value.mean(dim=0, keepdim=True)
