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

        self.pause_training_button = self.server.add_gui_button("Pause Training")
        self.pause_time_button = self.server.add_gui_button("Pause Time")
        # self.sh_order = self.server.add_gui_slider(
        #     "SH Order", min=0, max=3, step=1, initial_value=0
        # )
        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=64, max=1024, step=2, initial_value=512
        )
        self.fov_slider = self.server.add_gui_slider(
            "FoV", min=60, max=150, step=1, initial_value=90
        )
        self.inst_id_slider = self.server.add_gui_slider(
            "Video ID", min=0, max=len(self.sublen)-1, step=1, initial_value=0
        )

        self.frameid_sub_slider = self.server.add_gui_slider(
            "Frame ID", min=0, max=max(self.sublen)-1, step=1, initial_value=0
        )

        self.frameid_sub_slider_appr = self.server.add_gui_slider(
            "Frame ID (appr)", min=0, max=max(self.sublen)-1, step=1, initial_value=0
        )

        self.toggle_outputs = self.server.add_gui_dropdown(
            "Toggle outputs", ('rgb', 'depth', 'alpha', 'xyz', 'feature', 'mask_fg', 'vis2d', "gauss_mask"), initial_value="rgb"
        )

        self.toggle_view_sel = self.server.add_gui_dropdown("Toggle view control", ('rotation', 'all'), initial_value="all")
        self.toggle_viewpoint = self.server.add_gui_dropdown("Toggle viewpoint", ('ref', 'bev'), initial_value="ref")

        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.frameid_sub_slider.on_update
        def _(_):
            self.need_update = True

        @self.frameid_sub_slider_appr.on_update
        def _(_):
            self.need_update = True

        @self.inst_id_slider.on_update
        def _(_):
            self.need_update = True

        @self.toggle_outputs.on_update
        def _(_):
            self.need_update = True

        @self.toggle_view_sel.on_update
        def _(_):
            self.need_update = True

        @self.toggle_viewpoint.on_update
        def _(_):
            self.need_update = True

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training = not self.pause_training

        @self.pause_time_button.on_click
        def _(_):
            self.pause_time = not self.pause_time

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

            # initialize cameras
            client.camera.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            client.camera.position = np.array([0.0, 0.0, 0.0])
            # look at affects the rotation of the viewer
            client.camera.look_at = np.array([0.0, 0.0, 3.0])
        self.debug_idx = 0

    def set_renderer(self, renderer):
        self.renderer = renderer

    def frameid_inc(self, value, inst_id):
        curr_frame_value = value + 1
        if curr_frame_value >= self.sublen[inst_id]:
            curr_frame_value = 0
        return int(curr_frame_value)

    def get_frameid(self, value, inst_id):
        return self.frame_offset[inst_id] + min(value, self.sublen[inst_id]-1)


    def read_batch_from_gui(self, client):
        camera = client.camera
        w2c = get_w2c(camera)
        # rot_offset = np.asarray([ 0.9624857, 2.3236458, -1.2028077])
        # w2c[:3,:3] = w2c[:3,:3] @ cv2.Rodrigues(rot_offset)[0].T
        self.renderer.config["render_res"] = self.resolution_slider.value
        camera.fov = self.fov_slider.value * np.pi / 180
        W = self.resolution_slider.value
        H = int(self.resolution_slider.value / camera.aspect)
        focal_x = W / 2 / np.tan(camera.fov / 2)
        focal_y = H / 2 / np.tan(camera.fov / 2)

        inst_id = self.inst_id_slider.value
        frameid = self.get_frameid(self.frameid_sub_slider.value, inst_id)
        if not self.pause_time:
            time.sleep(0.1)
            self.frameid_sub_slider.value = self.frameid_inc(self.frameid_sub_slider.value, inst_id)
            self.frameid_sub_slider_appr.value = self.frameid_inc(self.frameid_sub_slider_appr.value, inst_id)

        intrinsics = self.renderer.get_intrinsics(frameid).cpu().numpy()
        extrinsics = self.renderer.gaussians.get_extrinsics(frameid).cpu().numpy()

        if "render_res" in self.renderer.config:
            res = self.renderer.config["render_res"]
        else:
            res = self.renderer.config["eval_res"]

        raw_size = self.renderer.data_info["raw_size"][0]
        # crop2raw = np.zeros(4)
        # # ratio = raw_size[0] / H # heights to be max
        # # crop2raw[0] = W * ratio / res
        # # crop2raw[1] = H * ratio / res
        # crop2raw[0] = W / res
        # crop2raw[1] = H / res
        # tan(fov/2) = res/2 / focal
        intrinsics = np.asarray([res / 2 / np.tan(camera.fov / 2) / camera.aspect,
                                 res / 2 / np.tan(camera.fov / 2),
                                 res / 2,
                                 res / 2])
        # intrinsics = mat2K(K2inv(crop2raw) @ K2mat(intrinsics))

        if self.toggle_view_sel.value == "all":
            extrinsics = w2c @ extrinsics
        elif self.toggle_view_sel.value == "rotation":
            extrinsics[:3,:3] = w2c[:3,:3] @ extrinsics[:3,:3]
        else:
            raise ValueError("Invalid view selection")

        if self.renderer.config["field_type"] == "fg":
            # bev obj
            rot_offset = np.asarray([np.pi/2,0.,0.])
            rot_offset = cv2.Rodrigues(rot_offset)[0]        
            extrinsics_2ndscreen = extrinsics.copy()
            extrinsics_2ndscreen[:3,:3] = rot_offset @ extrinsics[:3,:3]
        else:
            # bev scene
            rot_offset = cv2.Rodrigues(np.asarray([0., 0., np.pi/2]))[0] @ \
                        cv2.Rodrigues(np.asarray([np.pi/2,0.,0.]))[0]        
            extrinsics_2ndscreen = extrinsics.copy()
            extrinsics_2ndscreen[:3,:3] = rot_offset # @ extrinsics[:3,:3]
            extrinsics_2ndscreen[:3,3] = rot_offset @ extrinsics[:3,:3].T @ extrinsics[:3,3]
            extrinsics_2ndscreen[2, 3] = 2

        if self.toggle_viewpoint.value == "bev":
            extrinsics = np.stack([extrinsics_2ndscreen, extrinsics],0)
        else:
            extrinsics = np.stack([extrinsics, extrinsics_2ndscreen],0)

        # field2cam = None
        # crop2raw=np.asarray([[focal_x, focal_y, W/2, H/2]])
        crop2raw = None
        frameid = self.frameid_sub_slider.value
        intrinsics = np.tile(intrinsics[None],(2,1))
        batch = construct_batch(
            inst_id,
            [frameid, frameid],
            res,
            {"fg": extrinsics},
            intrinsics,
            crop2raw,
            self.device,
        )
        return batch

    def update_gaussians_by_frameid(self, batch):
        inst_id = self.inst_id_slider.value
        frameid_appr = self.get_frameid(self.frameid_sub_slider_appr.value, inst_id)
        self.renderer.process_frameid(batch) # absolute
        self.renderer.gaussians.update_motion(batch["frameid"])
        self.renderer.gaussians.update_appearance(batch["frameid"] * 0 + frameid_appr)
        self.renderer.gaussians.update_extrinsics(batch["frameid"])


    @torch.no_grad()
    def update(self, update_gaussian_func = update_gaussians_by_frameid):
        if self.need_update:
            for client in self.server.get_clients().values():
                try:
                    batch = self.read_batch_from_gui(client)

                    # update gaussians
                    if update_gaussian_func is not None:
                        update_gaussian_func(self, batch)
                    else:
                        # self.renderer.process_frameid(batch) # absolute
                        self.update_gaussians_by_frameid(batch)

                    # add additional controls
                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()
                    batch["frameid"][:] = -1 # this queries most recent update
                    outputs = self.renderer.evaluate_simple(batch, update_traj=False)
                    # with torch_profile("tmp/", "profile", enabled=self.renderer.config["profile"]):
                    #     outputs = self.renderer.evaluate(
                    #         batch, is_pair=False, augment_nv=False
                    #     )
                    end_cuda.record()
                    torch.cuda.synchronize()
                    interval = start_cuda.elapsed_time(end_cuda) / 1000.0

                    toggle_outputs = self.toggle_outputs.value
                    out = outputs[toggle_outputs].astype(np.float32)
                    if "apply_pca_fn" in self.renderer.data_info:
                        pca_fn = self.renderer.data_info["apply_pca_fn"]
                    else:
                        pca_fn = None
                    out = img2color(toggle_outputs, out, pca_fn=pca_fn)
                    out_big = cv2.resize(out[0], (out[0].shape[1]*3, out[1].shape[0]*3))
                    out_small = out[1]
                    out_small = cv2.copyMakeBorder(out_small, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255,0,0))
                    if out_big.ndim != out_small.ndim:
                        out_small = out_small[...,None]
                    out_big[:out_small.shape[0], :out_small.shape[1]] = out_small
                    # out = out_big # cv2.resize(out_big, (out_big.shape[1]//4, out_big.shape[0]//4))
                    out = cv2.resize(out_big, (out_big.shape[1]//2, out_big.shape[0]//2))
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