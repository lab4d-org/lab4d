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

from lab4d.render import render, get_config, construct_batch_from_opts
from lab4d.utils.camera_utils import construct_batch
from lab4d.utils.geom_utils import K2inv, K2mat, mat2K
from projects.diffgs.trainer import GSplatTrainer as Trainer

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
    def __init__(self, device, viewer_port):
        self.device = device
        self.port = viewer_port

        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)

        self.need_update = False

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
            "Video ID", min=0, max=100, step=1, initial_value=0
        )
        self.frameid_sub_slider = self.server.add_gui_slider(
            "Frame ID", min=0, max=100, step=1, initial_value=0
        )

        self.show_train_camera = self.server.add_gui_checkbox(
            "Show Train Camera", initial_value=False
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

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training = not self.pause_training
            self.train_viewer_update_period_slider.disabled = not self.pause_training
            self.pause_training_button.name = (
                "Resume Training" if self.pause_training else "Pause Training"
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
            # # initialize cameras
            # for client in self.server.get_clients().values():
            #     client.camera.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            #     print(client.camera.wxyz)

        self.debug_idx = 0

    def set_renderer(self, renderer):
        self.renderer = renderer

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
                    H = int(self.resolution_slider.value/camera.aspect)
                    focal_x = W/2/np.tan(camera.fov/2)
                    focal_y = H/2/np.tan(camera.fov/2)

                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()

                    intrinsics = self.renderer.get_intrinsics(0)[None]
                    res = self.renderer.config["render_res"]
                    raw_size = self.renderer.data_info["raw_size"][0]  # full range of pixels
                    crop2raw = torch.zeros((1, 4), device=self.device)
                    crop2raw[:, 0] = raw_size[1] / res
                    crop2raw[:, 1] = raw_size[0] / res
                    intrinsics = mat2K(K2inv(crop2raw) @ K2mat(intrinsics))
                    field2cam = {"fg": w2c[None]}
                    # field2cam = None
                    # crop2raw=np.asarray([[focal_x, focal_y, W/2, H/2]])
                    crop2raw = None
                    inst_id = self.inst_id_slider.value
                    frameid_sub = [self.frameid_sub_slider.value]

                    batch = construct_batch(inst_id, frameid_sub, res, field2cam, intrinsics, crop2raw, self.device)

                    outputs,_ = self.renderer.evaluate(batch, is_pair=False, augment_nv=False)
                    end_cuda.record()
                    torch.cuda.synchronize()
                    interval = start_cuda.elapsed_time(end_cuda)/1000.

                    out = outputs["rgb"][0].astype(np.float32)
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


def main(_):
    opts = get_config()
    # load model/data
    opts["logroot"] = sys.argv[1].split("=")[1].rsplit("/", 2)[0]
    model, data_info, ref_dict = Trainer.construct_test_model(opts)

    gui = ViserViewer(device=model.device, viewer_port=6789)
    gui.set_renderer(model)
    while(True):
        gui.update()

if __name__ == "__main__":
    app.run(main)
