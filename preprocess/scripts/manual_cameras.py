# Copyright (c) 2023 Alex Lyons, Carnegie Mellon University.
import configparser
import glob
import io
import json
import os
import sys
from functools import partial
from math import cos, pi, sin

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import trimesh
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lab4d.utils.geom_utils import K2mat, compute_crop_params
from preprocess.libs.io import read_mask


def read_mask_img(img_path, crop_size, use_full, use_minvis):
    mask_img_path = img_path.replace("JPEGImages", "Annotations")
    mask_img = cv2.imread(mask_img_path)[..., ::-1] / 255.0
    shape = mask_img.shape
    mask_path = mask_img_path.replace(".jpg", ".npy")
    mask, _ = read_mask(mask_path, shape)

    if use_minvis:
        img = cv2.imread(img_path)[..., ::-1] / 255.0

        orange_mask = np.stack((mask,) * 3, axis=-1).squeeze().astype(np.float64)
        orange_mask[:, :, 1] *= 165.0 / 255.0
        orange_mask[:, :, 2] = 0

        foreground = cv2.addWeighted(img * mask, 0.4, orange_mask, 0.6, 0)
        background = img * (1 - mask)
        mask_img = foreground + background

    if mask.shape[0] != shape[0] or mask.shape[1] != shape[1]:
        mask = cv2.resize(mask, shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(mask, -1)
    crop2raw = compute_crop_params(mask, crop_size=crop_size, use_full=use_full)

    x0, y0 = np.meshgrid(range(crop_size), range(crop_size))
    hp_crop = np.stack([x0, y0, np.ones_like(x0)], -1)  # augmented coord
    hp_crop = hp_crop.astype(np.float32)
    hp_raw = hp_crop @ K2mat(crop2raw).T  # raw image coord
    x0 = hp_raw[..., 0].astype(np.float32)
    y0 = hp_raw[..., 1].astype(np.float32)
    mask_img = cv2.remap(mask_img, x0, y0, interpolation=cv2.INTER_LINEAR)

    return mask_img


def adjust_bounds(fig, v_adj):
    maxBoundMargin = 0.5
    maxBound = np.max(np.abs(v_adj))
    maxBound = maxBound * (1 + maxBoundMargin)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-maxBound, maxBound]),
            yaxis=dict(range=[-maxBound, maxBound]),
            zaxis=dict(range=[-maxBound, maxBound]),
        ),
        # scene_dragmode=False,
        autosize=False,
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0,
        ),
    )

    return fig


def get_annotation(annotations, index):
    return Image.fromarray(np.uint8(annotations[index] * 255))


def gen_fig_img(source):
    return dict(
        source=source,
        xref="paper",
        yref="paper",
        x=0,
        y=0,
        sizex=1,
        sizey=1,
        sizing="contain",
        opacity=0.5,
        layer="below",
        xanchor="left",
        yanchor="bottom",
    )


def init_fig(annots, mesh_path):
    mesh = trimesh.load_mesh(mesh_path)
    v = mesh.vertices
    f = mesh.faces

    center = np.mean(v, axis=0)
    v_adj = v - center

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v_adj[:, 0],
                y=v_adj[:, 1],
                z=v_adj[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color="blue",
            )
        ]
    )

    fig.add_layout_image(gen_fig_img(get_annotation(annots, 0)))

    fig = adjust_bounds(fig, v_adj)

    annot_height, annot_width, _ = annots[0].shape
    fig.update_layout(
        scene_aspectmode="cube", width=2 * annot_width, height=2 * annot_height
    )
    # fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # fig.update_zaxes(scaleanchor="x", scaleratio=1)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return fig, v_adj


def debug_format(R):
    s = [str(round(r, 5)) for r in R.flatten()]
    print(
        s[0]
        + " "
        + s[1]
        + " "
        + s[2]
        + " 0 "
        + s[3]
        + " "
        + s[4]
        + " "
        + s[5]
        + " 0 "
        + s[6]
        + " "
        + s[7]
        + " "
        + s[8]
        + " 0 0 0 0 1 "
    )


class FigureTracker:
    def __init__(self, annots, annot_ids, mesh_path, seqnames, use_minvis):
        self.fig, self.v_adj = init_fig(annots, mesh_path)
        self.curr_frame = annot_ids[0]
        self.se3_dict = {}
        self.terminated = False
        self.annots = annots
        self.annot_ids = annot_ids
        self.vid = 0
        self.seqnames = seqnames
        self.use_minvis = use_minvis

        self.R = np.eye(3)
        self.Rx = np.eye(3)
        self.Ry = np.eye(3)
        self.Rz = np.eye(3)

    def numFrames(self):
        return self.annots.shape[0] - 1


def exit_gradio(track, demo):
    if track.terminated:
        demo.close()
    return


def terminate(track):
    print("Use Ctrl+C to continue the program.")
    print("TODO: implement exit function")
    track.terminated = True
    endButton = gr.Button.update(visible=False)
    endSlider = gr.Slider.update(visible=False)
    endPlot = gr.Plot.update(visible=False)
    endText = gr.Textbox.update(visible=False)
    return (
        gr.Textbox.update(visible=True),
        endPlot,
        endSlider,
        endSlider,
        endSlider,
        endSlider,
        endButton,
        endButton,
        endText,
        endButton,
        endButton,
        endButton,
    )


def load_annotations(config, vidid, use_minvis):
    crop_size = 256
    img_path = config.get("data_%d" % vidid, "img_path") + "*.jpg"
    imglist = sorted(glob.glob(img_path))
    masks = np.zeros((len(imglist) - 1, crop_size, crop_size, 3))
    frameids = np.zeros((len(imglist) - 1))
    for im0idx in range(len(imglist) - 1):
        use_full = False
        mask_img = read_mask_img(imglist[im0idx], crop_size, use_full, use_minvis)
        masks[im0idx] = mask_img
        # frameids[im0idx] = int(imglist[im0idx].split("/")[-1].split(".")[0])
        frameids[im0idx] = im0idx
    frameids = frameids.astype(int)
    return masks, frameids


def trig(degree):
    return cos(degree * pi / 180), sin(degree * pi / 180)


def update_rotx(track, angle):
    ca, sa = trig(angle)
    track.Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    return update_rot(track)


def update_roty(track, angle):
    ca, sa = trig(angle)
    track.Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    return update_rot(track)


def update_rotz(track, angle):
    ca, sa = trig(angle)
    track.Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    return update_rot(track)


def image_convert(fig):
    fig_bytes = fig.to_image(format="png")
    return Image.open(io.BytesIO(fig_bytes))


def update_rot(track):
    track.R = track.Rz @ track.Ry @ track.Rx
    new_v = track.v_adj @ track.R.T

    track.fig.data[0].x = new_v[:, 0]
    track.fig.data[0].y = new_v[:, 1]
    track.fig.data[0].z = new_v[:, 2]
    return load_fig(track)


def load_fig(track):
    unit_vec = np.asarray([1, 0.01, 0.01])
    zoom = 2

    eye_x = unit_vec[0] * zoom
    eye_y = unit_vec[1] * zoom
    eye_z = unit_vec[2] * zoom

    up_x = 0
    up_y = 1
    up_z = 0

    center_x = 0
    center_y = 0
    center_z = 0

    eye = dict(x=eye_x, y=eye_y, z=eye_z)
    center = dict(x=center_x, y=center_y, z=center_z)
    up = dict(x=up_x, y=up_y, z=up_z)

    camera = dict(eye=eye, up=up, center=center)
    track.fig.update_layout(scene=dict(camera=camera))
    return track.fig


def caminfo_to_rotation(track):
    caminfo = track.fig.layout.scene.camera
    final_R = np.zeros((4, 4))
    final_R[2, 3] = 3
    final_R[3, 3] = 1
    eye = np.asarray([caminfo.eye.x, caminfo.eye.y, caminfo.eye.z])
    center = np.asarray([caminfo.center.x, caminfo.center.y, caminfo.center.z])
    up = np.asarray([caminfo.up.x, caminfo.up.y, caminfo.up.z])
    L = center - eye
    L = L / np.linalg.norm(L)
    s = np.cross(L, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, L)
    cam_R = np.array([s, u, -L])
    final_R[0:3, 0:3] = cam_R @ track.R
    final_R[0:3, 0:3] = final_R[0:3, 0:3].T  # field to camera
    # debug_format(cam_R @ track.R)

    track.se3_dict[int(track.curr_frame)] = final_R.tolist()
    print("Saved rotation for frame %d" % track.curr_frame)
    save_dir = (
        "database/processed/Cameras/Full-Resolution/%s" % track.seqnames[track.vid]
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = "%s/01-manual.json" % save_dir
    with open(save_path, "w") as fp:
        json.dump(track.se3_dict, fp, indent=4)

    return track.fig


def update_fig_annot(fig, annots, index):
    new_annot = get_annotation(annots, index)
    fig.update_layout_images(gen_fig_img(new_annot))
    return fig


def annot_slider_update(track, index):
    annot_id = track.annot_ids[index]
    new_fig = update_fig_annot(track.fig, track.annots, index)
    track.curr_frame = annot_id
    return load_fig(track), str(annot_id)


def switch_video(track, config):
    track.vid += 1
    track.se3_dict = {}
    if track.vid >= len(track.seqnames):
        return (gr.Button.update(visible=False), None, 0, "0")
    else:
        track.annots, track.annot_ids = load_annotations(
            config, track.vid, track.use_minvis
        )
        track.curr_frame = track.annot_ids[0]
        new_fig = update_fig_annot(track.fig, track.annots, 0)
        track.fig = new_fig
        if track.vid == len(track.seqnames) - 1:
            return (
                gr.Button.update(visible=False),
                track.fig,
                gr.Slider.update(value=0, maximum=track.annots.shape[0] - 1),
                str(track.annot_ids[0]),
            )
        else:
            return (
                gr.Button.update(visible=True),
                track.fig,
                gr.Slider.update(value=0, maximum=track.annots.shape[0] - 1),
                str(track.annot_ids[0]),
            )


def next_frame(track, index):
    new_index = index + 1
    if new_index >= len(track.annot_ids):
        new_index = 0
    annot_id = track.annot_ids[new_index]
    track.curr_frame = annot_id
    new_fig = update_fig_annot(track.fig, track.annots, new_index)
    return load_fig(track), new_index, str(annot_id)


def prev_frame(track, index):
    new_index = index - 1
    if new_index < 0:
        new_index = len(track.annot_ids) - 1
    annot_id = track.annot_ids[new_index]
    track.curr_frame = annot_id
    new_fig = update_fig_annot(track.fig, track.annots, new_index)
    return load_fig(track), new_index, str(annot_id)


def manual_camera_interface(vidname, use_minvis, mesh_path):
    config = configparser.RawConfigParser()
    config.read("database/configs/%s.config" % vidname)
    seqnames = []
    for vidid in range(len(config.sections()) - 1):
        seqname = config.get("data_%d" % vidid, "img_path").strip("/").split("/")[-1]
        seqnames.append(seqname)

    annots, annot_ids = load_annotations(config, 0, use_minvis)

    track = FigureTracker(annots, annot_ids, mesh_path, seqnames, use_minvis)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                meshMap = gr.Plot(
                    label="Mesh (Warning: Don't rotate the mesh by dragging)",
                )
                demo.load(partial(load_fig, track), [], meshMap)

            with gr.Column():
                rx = gr.Slider(
                    0, 360, label="Elevation (Degree)", value=0, interactive=True
                )
                ry = gr.Slider(
                    0, 360, label="Azimuth (Degree)", value=0, interactive=True
                )
                rz = gr.Slider(0, 360, label="Roll (Degree)", value=0, interactive=True)
                frame = gr.Slider(
                    0,
                    track.numFrames(),
                    label="Frame Index",
                    value=0,
                    interactive=True,
                    step=1,
                )
                frame_text = gr.Textbox(
                    value=str(track.annot_ids[0]), label="Frame Index"
                )

                rx.release(partial(update_rotx, track), [rx], [meshMap])
                ry.release(partial(update_roty, track), [ry], [meshMap])
                rz.release(partial(update_rotz, track), [rz], [meshMap])
                frame.release(
                    partial(annot_slider_update, track),
                    [frame],
                    [meshMap, frame_text],
                )

                with gr.Row():
                    prevFrame = gr.Button(value="Previous Frame")
                    prevFrame.click(
                        partial(prev_frame, track),
                        [frame],
                        [meshMap, frame, frame_text],
                    )
                    nextFrame = gr.Button(value="Next Frame")
                    nextFrame.click(
                        partial(next_frame, track),
                        [frame],
                        [meshMap, frame, frame_text],
                    )

                rot = gr.Button(value="Calculate and Save Rotation")
                rot.click(partial(caminfo_to_rotation, track), [], [meshMap])

        with gr.Row():
            lastVid = track.vid == len(track.seqnames) - 1
            nextVid = gr.Button(value="Next Video", visible=not lastVid)
            nextVid.click(
                partial(switch_video, track, config),
                [],
                [nextVid, meshMap, frame, frame_text],
            )

        with gr.Row():
            terminate_Text = "Use Ctrl+C in terminal to continue the program"
            exited = gr.Textbox(
                label="Gradio Exited", value=terminate_Text, visible=False
            )
            exited.change(partial(exit_gradio, track, demo), [], [])

            exit = gr.Button(value="Exit")
            exit.click(
                partial(terminate, track),
                [],
                [
                    exited,
                    meshMap,
                    rx,
                    ry,
                    rz,
                    frame,
                    nextVid,
                    rot,
                    frame_text,
                    exit,
                    nextFrame,
                    prevFrame,
                ],
            )

        demo.queue().launch(share=True)


if __name__ == "__main__":
    # The video(s) name (ex. cat-pikachu-0)
    vidname = sys.argv[1]
    # If used minvis for creating annotations (1 or 0)
    use_minvis = bool(int(sys.argv[2]))

    mesh_path = "database/mesh-templates/cat-pikachu-remeshed.obj"
    manual_camera_interface(vidname, use_minvis, mesh_path)
