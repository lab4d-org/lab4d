import sys, os
import glob
import pdb
import torch
import numpy as np
import cv2
from torch.functional import F
from scipy.interpolate import griddata


sys.path.insert(0, os.getcwd())
from preprocess.third_party.vcnplus.flowutils.flowlib import point_vec


def fill_with_interpolation(mask, data_map):
    """
    Fill the masked regions of the map using linear interpolation from the unmasked regions.

    Parameters:
    mask (numpy.ndarray): A 2D boolean array where True indicates masked regions.
    data_map (numpy.ndarray): A 2D array containing the map data.

    Returns:
    numpy.ndarray: A 2D array with the masked regions filled.
    """
    # Get the coordinates of the masked and unmasked points
    x, y = np.indices(data_map.shape)
    x, y = x.flatten(), y.flatten()

    masked_points = mask.flatten()
    unmasked_points = ~masked_points

    # Extract the values of the unmasked points
    data_values = data_map.flatten()
    unmasked_values = data_values[unmasked_points]

    # Get the coordinates of the unmasked points
    unmasked_coords = np.array((x[unmasked_points], y[unmasked_points])).T

    # Interpolate over the masked points
    interpolated_values = griddata(
        unmasked_coords,
        unmasked_values,
        (x[masked_points], y[masked_points]),
        method="linear",
    )

    # Create the filled map
    filled_map = data_map.copy()
    filled_map[mask] = interpolated_values

    return filled_map


def compute_tracks(seqname, outdir, dframe):
    """
    dframe is a list of frames intervals, e.g, [1,2,4,8]
    """
    device = "cuda"
    grid_size = 100
    max_res = 4e5
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)

    for df in dframe:
        fw_path = "%s/FlowFW_%d/Full-Resolution/%s/" % (outdir, df, seqname)
        bw_path = "%s/FlowBW_%d/Full-Resolution/%s/" % (outdir, df, seqname)
        os.system("mkdir -p %s" % (fw_path))
        os.system("mkdir -p %s" % (bw_path))
    max_df = max(dframe)

    img_paths = sorted(
        glob.glob("%s/JPEGImages/Full-Resolution/%s/*.jpg" % (outdir, seqname))
    )

    # load all images and compute resize ratio
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)[:, :, ::-1].copy()
        img = torch.tensor(img, device=device, dtype=torch.float32).permute(2, 0, 1)
        imgs.append(img)
    imgs = torch.stack(imgs, 0)
    _, _, inp_h, inp_w = imgs.shape
    res_fac = np.sqrt(max_res / (inp_h * inp_w))
    max_h = int(np.ceil(inp_h * res_fac))
    max_w = int(np.ceil(inp_w * res_fac))
    imgs = F.interpolate(imgs, (max_h, max_w), mode="bilinear")

    # compute forward flow
    empth_mask = torch.ones(max_h, max_w, device=device)
    empty_flow = torch.zeros(max_h, max_w, 3, device=device)
    empty_flow[..., 2] = 1
    # Queried points of shape (B, N, 3) in format (t, x, y) for frame index and pixel coordinates.
    hxy = torch.meshgrid(
        torch.linspace(0, max_h - 1, grid_size).long(),
        torch.linspace(0, max_w - 1, grid_size).long(),
    )
    queries = torch.stack(
        [torch.zeros_like(hxy[0].flatten()), hxy[1].flatten(), hxy[0].flatten()], 1
    )
    queries = queries.to(device)[None].float()

    # forward
    for idx in range(len(img_paths)):
        print(f"Processing frame {idx}/{len(img_paths)-1}")
        video = imgs[idx : idx + max_df + 1][None]  # 1 T C H W
        # fill in to 5 frames at least
        if video.shape[1] < 5:
            video = torch.cat(
                [video, video[:, -1:].repeat(1, 5 - video.shape[1], 1, 1, 1)], 1
            )

        # forward
        # B T N 2,  B T N 1
        pred_tracks, pred_visibility = cotracker(video, queries=queries)
        pred_tracks = pred_tracks[0]  # T N 2
        pred_visibility = pred_visibility[0]  # T N 1
        pred_uncertainty = 1 - pred_visibility.float() * 2  # visible -> low uncertainty
        pred_uncertainty[:] = -1 # TODO find a threshold

        # assign flow: fw
        for df in dframe:
            if idx + df >= len(img_paths):
                continue
            fw_path = "%s/FlowFW_%d/Full-Resolution/%s/" % (outdir, df, seqname)
            flowfw_path = os.path.join(fw_path, f"{idx:05d}.npy")

            flowfw = empty_flow.clone()
            flowfw[hxy[0].flatten(), hxy[1].flatten(), :2] = (
                pred_tracks[df] - queries[0, :, 1:]
            )
            flowfw[hxy[0].flatten(), hxy[1].flatten(), 2] = pred_uncertainty[df]
            flowfw = flowfw.cpu().numpy()

            # fill in empty flow with linear interpolation
            mask = empth_mask.clone().cpu().numpy()
            mask[hxy[0].flatten(), hxy[1].flatten()] = 0  # 1: empty, 0: filled
            mask = mask.astype(bool)
            flowfw[..., 0] = fill_with_interpolation(mask, flowfw[..., 0])
            flowfw[..., 1] = fill_with_interpolation(mask, flowfw[..., 1])
            flowfw[..., 2] = fill_with_interpolation(mask, flowfw[..., 2])

            np.save(flowfw_path, flowfw)
            flowvis = point_vec(
                imgs[idx].permute(1, 2, 0).cpu().numpy(), flowfw, skip=10
            )
            cv2.imwrite("%s/visflo-%05d.jpg" % (fw_path, idx), flowvis)

        # backward
        idy = len(img_paths) - idx - 1
        print(f"Processing frame {idy}/{len(img_paths)-1}")
        video = imgs[max(0, idy - max_df) : idy + 1].flip(0)[None]  # fliped
        # fill in to 5 frames at least
        if video.shape[1] < 5:
            video = torch.cat(
                [video, video[:, -1:].repeat(1, 5 - video.shape[1], 1, 1, 1)], 1
            )

        pred_tracks, pred_visibility = cotracker(video, queries=queries)
        pred_tracks = pred_tracks[0]  # T N 2
        pred_visibility = pred_visibility[0]
        pred_uncertainty = 1 - pred_visibility.float() * 2
        pred_uncertainty[:] = -1 # TODO find a threshold

        # assign flow: bw
        for df in dframe:
            if idy - df < 0:
                continue
            bw_path = "%s/FlowBW_%d/Full-Resolution/%s/" % (outdir, df, seqname)
            flowbw_path = os.path.join(bw_path, f"{idy:05d}.npy")

            flowbw = empty_flow.clone()
            flowbw[hxy[0].flatten(), hxy[1].flatten(), :2] = (
                pred_tracks[df] - queries[0, :, 1:]
            )
            flowbw[hxy[0].flatten(), hxy[1].flatten(), 2] = pred_uncertainty[df]
            flowbw = flowbw.cpu().numpy()

            # fill in empty flow with linear interpolation
            mask = empth_mask.clone().cpu().numpy()
            mask[hxy[0].flatten(), hxy[1].flatten()] = 0
            mask = mask.astype(bool)
            flowbw[..., 0] = fill_with_interpolation(mask, flowbw[..., 0])
            flowbw[..., 1] = fill_with_interpolation(mask, flowbw[..., 1])
            flowbw[..., 2] = fill_with_interpolation(mask, flowbw[..., 2])

            np.save(flowbw_path, flowbw)
            flowvis = point_vec(
                imgs[idy].permute(1, 2, 0).cpu().numpy(), flowbw, skip=10
            )
            cv2.imwrite("%s/visflo-%05d.jpg" % (bw_path, idy), flowvis)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <seqname> <outdir> <dframe>")
        print(
            f"  Example: python {sys.argv[0]} cat-pikachu-0-0000 'database/processed/' 1"
        )
        exit()
    seqname = sys.argv[1]
    outdir = sys.argv[2]
    dframe = sys.argv[3]
    dframe = [int(i) for i in dframe[1:-1].split(",")]
    compute_tracks(seqname, outdir, dframe)
