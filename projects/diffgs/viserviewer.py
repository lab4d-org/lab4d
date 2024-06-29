import os, sys
from absl import app
from threading import Thread
import torch
import numpy as np
import time
import viser
import viser.transforms as tf
import pdb
import cv2
from collections import deque

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.utils.camera_utils import construct_batch
from lab4d.utils.vis_utils import img2color
from lab4d.utils.geom_utils import K2inv, K2mat, mat2K
from lab4d.utils.profile_utils import torch_profile


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w


def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c


class RenderThread(Thread):
    pass


class ViserViewer:
    def __init__(self, device, viewer_port, data_info):
        self.device = device
        self.port = viewer_port

        frame_offset = data_info["frame_info"]["frame_offset"]
        self.sublen = frame_offset[1:] - frame_offset[:-1]
        self.frame_offset = frame_offset

        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)

        self.need_update = False

        self.pause_time = True
        self.pause_training = False
        self.train_viewer_update_period_slider = self.server.add_gui_slider(
            "Train Viewer Update Period",
            min=1,
            max=100,
            step=1,
            initial_value=10,
            disabled=self.pause_training,
        )

        self.pause_training_button = self.server.add_gui_button("Pause Training")
        self.pause_time_button = self.server.add_gui_button("Pause Time")
        self.sh_order = self.server.add_gui_slider(
            "SH Order", min=1, max=4, step=1, initial_value=1
        )
        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )
        self.near_plane_slider = self.server.add_gui_slider(
            "Near", min=0.1, max=30, step=0.5, initial_value=0.1
        )
        self.far_plane_slider = self.server.add_gui_slider(
            "Far", min=30.0, max=1000.0, step=10.0, initial_value=1000.0
        )
        self.inst_id_slider = self.server.add_gui_slider(
            "Video ID", min=0, max=len(self.sublen)-1, step=1, initial_value=0
        )

        self.frameid_sub_slider = self.server.add_gui_slider(
            "Frame ID", min=0, max=max(self.sublen)-1, step=1, initial_value=0
        )

        self.show_train_camera = self.server.add_gui_checkbox(
            "Show Train Camera", initial_value=False
        )

        self.toggle_outputs = self.server.add_gui_dropdown(
            "Toggle outputs", ('rgb', 'depth', 'alpha', 'xyz', 'flow', 'mask_fg'), initial_value="rgb"
        )

        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

        @self.show_train_camera.on_update
        def _(_):
            self.need_update = True

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.near_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.far_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.frameid_sub_slider.on_update
        def _(_):
            self.need_update = True

        @self.inst_id_slider.on_update
        def _(_):
            self.need_update = True

        @self.toggle_outputs.on_update
        def _(_):
            self.need_update = True

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training = not self.pause_training
            self.train_viewer_update_period_slider.disabled = not self.pause_training
            self.pause_training_button.name = (
                "Resume Training" if self.pause_training else "Pause Training"
            )

        @self.pause_time_button.on_click
        def _(_):
            self.pause_time = not self.pause_time
            self.pause_time_button.name = (
                "Resume Time" if self.pause_time else "Pause Time"
            )

        self.c2ws = []
        self.camera_infos = []

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

            # initialize cameras
            client.camera.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            client.camera.position = np.array([0.0, 0.0, 1.0])
        self.debug_idx = 0

    def set_renderer(self, renderer):
        self.renderer = renderer
        self.intrinsics = renderer.get_intrinsics(0)[None].cpu().numpy()
        self.extrinsics = renderer.gaussians.get_extrinsics().detach().cpu().numpy()

    @torch.no_grad()
    def update(self):
        if self.need_update:
            for client in self.server.get_clients().values():
                camera = client.camera
                w2c = get_w2c(camera)
                # rot_offset = np.asarray([ 0.9624857, 2.3236458, -1.2028077])
                # w2c[:3,:3] = w2c[:3,:3] @ cv2.Rodrigues(rot_offset)[0].T
                try:
                    W = self.resolution_slider.value
                    H = int(self.resolution_slider.value / camera.aspect)
                    focal_x = W / 2 / np.tan(camera.fov / 2)
                    focal_y = H / 2 / np.tan(camera.fov / 2)

                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()

                    inst_id = self.inst_id_slider.value
                    frameid_sub = [
                        min(self.frameid_sub_slider.value, self.sublen[inst_id]-1)
                    ]
                    if not self.pause_time:
                        self.frameid_sub_slider.value = min(self.frameid_sub_slider.value+1, self.sublen[inst_id]-1)
                    frameid = self.frame_offset[inst_id] + frameid_sub

                    intrinsics = self.intrinsics
                    if "render_res" in self.renderer.config:
                        res = self.renderer.config["render_res"]
                    else:
                        res = self.renderer.config["eval_res"]

                    raw_size = self.renderer.data_info["raw_size"][0]
                    crop2raw = np.zeros((1, 4))
                    ratio = raw_size[0] / H # heigh to be max
                    crop2raw[:, 0] = W * ratio / res
                    crop2raw[:, 1] = H * ratio / res
                    intrinsics = mat2K(K2inv(crop2raw) @ K2mat(intrinsics))
                    field2cam = {"fg": w2c[None] @ self.extrinsics[frameid]}
                    # field2cam = None
                    # crop2raw=np.asarray([[focal_x, focal_y, W/2, H/2]])
                    crop2raw = None

                    batch = construct_batch(
                        inst_id,
                        frameid_sub,
                        res,
                        field2cam,
                        intrinsics,
                        crop2raw,
                        self.device,
                    )

                    outputs, _ = self.renderer.evaluate(
                        batch, is_pair=False, augment_nv=False
                    )
                    # with torch_profile("tmp/", "profile", enabled=self.renderer.config["profile"]):
                    #     outputs, _ = self.renderer.evaluate(
                    #         batch, is_pair=False, augment_nv=False
                    #     )
                    end_cuda.record()
                    torch.cuda.synchronize()
                    interval = start_cuda.elapsed_time(end_cuda) / 1000.0

                    toggle_outputs = self.toggle_outputs.value
                    out = outputs[toggle_outputs][0].astype(np.float32)
                    out = img2color(toggle_outputs, out)
                except RuntimeError as e:
                    print(e)
                    interval = 1
                    continue
                client.set_background_image(out, format="jpeg")
                self.debug_idx += 1
                # if self.debug_idx % 100 == 0:
                #     cv2.imwrite(
                #         f"./tmp/viewer/debug_{self.debug_idx}.png",
                #         cv2.cvtColor(out * 255, cv2.COLOR_RGB2BGR),
                #     )

                self.render_times.append(interval)
                self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
                # print(f"Update time: {end - start:.3g}")