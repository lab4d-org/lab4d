import torch
import numpy as np
import soft_renderer as sr


def render_color(renderer, in_verts, faces, colors, texture_type="vertex"):
    """
    verts in ndc
    in_verts: ...,N,3/4
    faces: ...,N,3
    rendered: ...,4,...
    """

    verts = in_verts.clone()
    verts = verts.view(-1, verts.shape[-2], 3)
    faces = faces.view(-1, faces.shape[-2], 3)
    if texture_type == "vertex":
        colors = colors.view(-1, colors.shape[-2], 3)
    elif texture_type == "surface":
        colors = colors.view(-1, colors.shape[1], colors.shape[2], 3)
    device = verts.device

    offset = torch.Tensor(renderer.transform.transformer._eye).to(device)[
        np.newaxis, np.newaxis
    ]
    verts_pre = verts[:, :, :3] - offset
    verts_pre[:, :, 1] = -1 * verts_pre[:, :, 1]  # pre-flip
    rendered = renderer.render_mesh(
        sr.Mesh(verts_pre, faces, textures=colors, texture_type=texture_type)
    )
    return rendered


def render_flow(renderer, verts, faces, verts_n):
    """
    rasterization
    verts in ndc
    verts: ...,N,3/4
    verts_n: ...,N,3/4
    faces: ...,N,3
    """
    verts = verts.view(-1, verts.shape[1], 3)
    verts_n = verts_n.view(-1, verts_n.shape[1], 3)
    faces = faces.view(-1, faces.shape[1], 3)
    device = verts.device

    rendered_ndc_n = render_color(renderer, verts, faces, verts_n)
    _, _, h, w = rendered_ndc_n.shape
    rendered_sil = rendered_ndc_n[:, -1]

    ndc = np.meshgrid(range(w), range(h))
    ndc = torch.Tensor(ndc).to(device)[None]
    ndc[:, 0] = ndc[:, 0] * 2 / (w - 1) - 1
    ndc[:, 1] = ndc[:, 1] * 2 / (h - 1) - 1

    flow = rendered_ndc_n[:, :2] - ndc
    flow = flow.permute(0, 2, 3, 1)  # x,h,w,2
    flow = torch.cat([flow, rendered_sil[..., None]], -1)

    flow[rendered_sil < 1] = 0.0
    flow[..., -1] = 0.0  # discard the last channel
    return flow
