import cv2
import numpy as np
import sys, os
import trimesh
import math
import pdb
import matplotlib.pyplot as plt
from celluloid import Camera
import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import viser
import viser.transforms as tf
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time


sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import get_pts_traj
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.io import save_vid
from lab4d.config import load_flags_from_file
from projects.behavior.articulation_loader import ArticulationLoader


def get_img_from_plt(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        int(height), int(width), 3
    )
    return image


def spline_interp(sample_wp, forecast_size, num_kps=1, state_size=3, interp_size=56):
    if not torch.is_tensor(sample_wp):
        device = "cpu"
        sample_wp = torch.tensor(sample_wp, device=device)
    else:
        device = sample_wp.device

    t = torch.linspace(0, 1, forecast_size, device=device)
    coeffs = natural_cubic_spline_coeffs(
        t, sample_wp.reshape(-1, forecast_size, state_size * num_kps)
    )
    spline = NaturalCubicSpline(coeffs)
    point = torch.linspace(0, 1, interp_size, device=device)
    sample_wp_dense = spline.evaluate(point).reshape(sample_wp.shape[0], -1)
    return sample_wp_dense


def put_text(img, text, pos, color=(255, 0, 0)):
    img = img.astype(np.uint8)
    img = cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )
    return img


def concatenate_points(pointcloud1, pointcloud2):
    pts = np.concatenate([pointcloud1.vertices, pointcloud2.vertices], axis=0)
    colors = np.concatenate(
        [pointcloud1.visual.vertex_colors, pointcloud2.visual.vertex_colors],
        axis=0,
    )
    return trimesh.PointCloud(pts, colors=colors)


class DiffusionVisualizer:
    def __init__(
        self,
        xzmax,
        xzmin,
        state_size=3,
        num_timesteps=50,
        bg_field=None,
        logdir="base",
        lab4d_dir=None,
    ):
        self.state_size = state_size
        self.xzmax = xzmax
        self.xzmin = xzmin
        self.num_timesteps = num_timesteps
        self.bg_field = bg_field
        self.logdir = logdir

        # mesh rendering
        raw_size = [512, 512]
        renderer = PyRenderWrapper(raw_size)
        renderer.set_camera_bev(depth=(xzmax - xzmin) * 2)
        # set camera intrinsics
        fl = max(raw_size)
        intr = np.asarray([fl * 2, fl * 2, raw_size[1] / 2, raw_size[0] / 2])
        renderer.set_intrinsics(intr)
        renderer.align_light_to_camera()
        self.renderer = renderer

        # to store camera position
        self.userwp_list = []
        self.goal_list = []

        # TODO joint parsing
        if lab4d_dir is not None:
            opts = load_flags_from_file("%s/opts.log" % lab4d_dir)
            opts["load_suffix"] = "latest"
            opts["logroot"] = "logdir"
            opts["inst_id"] = 1
            opts["grid_size"] = 128
            opts["level"] = 0
            opts["vis_thresh"] = -10
            opts["extend_aabb"] = False
            self.articulation_loader = ArticulationLoader(opts)
        else:
            self.articulation_loader = None

    def delete(self):
        self.renderer.delete()

    # rotate the sample
    def render_rotate_sample(self, shape, n_frames=10, pts_traj=None, pts_color=None):
        sample_raw = shape.vertices
        frames = []
        for i in range(n_frames):
            progress = -0.25 + 0.5 * i / n_frames
            rot = cv2.Rodrigues(np.array([progress * math.pi, 0, 0]))[0]
            sample = np.dot(sample_raw, rot.T)
            shape = trimesh.PointCloud(sample, colors=shape.visual.vertex_colors)
            input_dict = {"shape": shape}
            if pts_traj is not None and pts_color is not None:
                input_dict["pts_traj"] = (pts_traj.reshape(-1, 3) @ rot.T).reshape(
                    pts_traj.shape
                )
                input_dict["pts_color"] = pts_color
            color = self.renderer.render(input_dict)[0]
            frames.append(color)
        return frames

    def render_trajectory(
        self,
        forward_samples,
        reverse_samples,
        past,
        x0_to_world,
        rotate=True,
        prefix="base",
    ):
        """
        forward_samples: list of torch.Tensor

        reverse_samples: M, bs, T,K,3
        x0_to_world: 1,1,3
        past: T,1,3
        """
        num_timesteps = self.num_timesteps
        frames = []
        if len(forward_samples) > 0:
            # forward
            num_wps = int(len(forward_samples[0][0]) / self.state_size)
            colors = np.ones((len(forward_samples[0]), num_wps, 4))  # N, K, 4
            colors[..., 1:3] = 0
            if num_wps > 1:
                colors = colors * np.linspace(0, 1, num_wps)[None, :, None]
            colors = colors.reshape(-1, 4)

            shape_clean = trimesh.PointCloud(
                forward_samples[0].reshape(-1, 3), colors=colors
            )
            if rotate:
                frames += self.render_rotate_sample(shape_clean)

            for i, sample in enumerate(forward_samples):
                shape = trimesh.PointCloud(sample.reshape(-1, 3), colors=colors)
                input_dict = {"shape": shape}
                color = self.renderer.render(input_dict)[0]
                color = put_text(color, f"step {i: 4} / {num_timesteps}", (10, 30))
                color = put_text(color, "Forward process", (10, 60))
                frames.append(color)
        else:
            shape_clean = None

        # reverse
        reverse_samples = reverse_samples.cpu().numpy()
        num_wps = reverse_samples.shape[2]
        num_kps = reverse_samples.shape[3]
        colors = np.ones((len(reverse_samples[0]), num_wps, num_kps, 4))
        colors[..., 1:3] = 0
        if num_wps > 1:
            colors = colors * np.linspace(0, 1, num_wps)[None, :, None, None]
        colors = colors.reshape(-1, 4)

        # get past location as a sphere
        past = past.reshape(-1, 3)
        past_shape = []
        for past_idx in range(len(past)):
            shape_sub = trimesh.creation.uv_sphere(radius=0.01, count=[4, 4])
            shape_sub = shape_sub.apply_translation(past[past_idx].cpu().numpy())
            past_shape.append(shape_sub)
        past_shape = trimesh.util.concatenate(past_shape)
        past_colors = np.array([[0, 1, 0, 1]]).repeat(len(past_shape.vertices), axis=0)

        # get bg mesh
        bg_pts = self.bg_field.bg_mesh.vertices - x0_to_world[:, 0].cpu().numpy()
        bg_colors = np.ones((len(bg_pts), 4)) * 0.6

        for i, sample in enumerate(reverse_samples):
            sample_vis = np.concatenate(
                [sample.reshape(-1, 3), past_shape.vertices, bg_pts], axis=0
            )
            sample_colors = np.concatenate([colors, past_colors, bg_colors], axis=0)

            shape = trimesh.PointCloud(sample_vis, colors=sample_colors)
            if i == 0:
                shape_noise = trimesh.PointCloud(sample.reshape(-1, 3), colors=colors)
            pts_traj, pts_color = get_pts_traj(
                np.transpose(sample.reshape(-1, num_wps, 3), (1, 0, 2)),
                num_wps - 1,
                traj_len=num_wps,
            )
            input_dict = {"shape": shape, "pts_traj": pts_traj, "pts_color": pts_color}
            color = self.renderer.render(input_dict)[0]
            color = put_text(
                color, f"step {i: 4} / {num_timesteps}", (10, 30), color=(0, 0, 255)
            )
            color = put_text(color, "Reverse process", (10, 60), color=(0, 0, 255))
            frames.append(color)

        # concat two shapes
        if rotate:
            # shape = concatenate_points(shape_clean, shape)
            frames += self.render_rotate_sample(
                shape, pts_traj=pts_traj, pts_color=pts_color
            )

        filename = "%s/%s-rendering" % (self.logdir, prefix)
        shape = trimesh.PointCloud(sample.reshape(-1, 3), colors=colors)
        shape.vertices = shape.vertices + x0_to_world[:, 0].cpu().numpy()
        shape.export("%s.obj" % filename)
        if shape_clean is not None:
            shape_clean.export("%s-clean.obj" % filename)
        shape_noise.export("%s-noise.obj" % filename)
        save_vid(filename, frames)
        print("Animation saved to %s.mp4" % filename)
        print("Mesh saved to %s.obj" % filename)

    def plot_trajectory_2d(
        self,
        forward_samples,
        reverse_samples,
        reverse_grad,
        gt_goal,
        xyz,
        yshape,
        past,
        cam,
        prefix="base",
    ):
        """
        reverse_samples: M, bs, T,K,3
        reverse_grad: M, bs, TK3
        gt_goal: T,K,3
        xyz: N,3
        past: T,K,3
        cam: T,1,3
        """
        num_wps = reverse_samples.shape[3]
        num_timesteps = self.num_timesteps
        xzmin = self.xzmin
        xzmax = self.xzmax
        past = past.reshape(past.shape[0], -1, 3)
        cam = cam.reshape(cam.shape[0], -1, 3)

        if torch.is_tensor(gt_goal):
            gt_goal = gt_goal.cpu().numpy()
        if torch.is_tensor(past):
            past = past.cpu().numpy()
        if torch.is_tensor(cam):
            cam = cam.cpu().numpy()

        # 2D plot
        frames = []
        # forward
        for i, sample in enumerate(forward_samples):
            fig, ax = plt.subplots()
            sample = sample.reshape(-1, num_wps, 3)
            plt.scatter(
                sample[:, -1, 0],
                sample[:, -1, 2],
                alpha=0.5,
                s=15,
                color="blue",
            )
            plt.scatter(
                sample[0, -1, 0],
                sample[0, -1, 2],
                alpha=0.5,
                s=30,
                color="red",
            )
            ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
            ax.text(0.0, 1.01, "Forward process", transform=ax.transAxes, size=15)
            plt.xlim(xzmin, xzmax)
            plt.ylim(xzmin, xzmax)
            plt.axis("off")
            image = get_img_from_plt(fig)
            frames.append(image)
            plt.close(fig)

        # reverse
        reverse_samples = reverse_samples.cpu().numpy()
        if torch.is_tensor(reverse_grad):
            reverse_grad = reverse_grad.cpu().numpy()
        num_wps = reverse_samples.shape[2]
        num_kps = reverse_samples.shape[3]
        for i, sample in enumerate(reverse_samples):
            fig, ax = plt.subplots()
            sample = sample.reshape(-1, num_wps, num_kps, 3)  # bs,t,k,3
            # draw all the waypoints
            for j in range(num_wps):
                alpha = np.clip(1 - j / num_wps, 0.1, 1)
                s = np.clip(30 + 100 * j / num_wps, 30, 100)
                if num_kps > 1:
                    s = s * 0.1
                plt.scatter(
                    sample[:, j, :, 0].flatten(),
                    sample[:, j, :, 2].flatten(),
                    alpha=alpha,
                    s=s,
                    color="red",
                )
            # draw line passing through the waypoints
            sample = sample.reshape(-1, num_wps * num_kps, 3)  # bs,t,k,3
            plt.plot(sample[:, :, 0].T, sample[:, :, 2].T, color="red", alpha=0.5)
            # past
            s = 100
            if num_kps > 1:
                s = s * 0.1
            plt.scatter(
                past[..., 0].flatten(),
                past[..., 2].flatten(),
                s=s,
                color="green",
                marker="x",
            )
            # cam
            plt.scatter(cam[:, 0, 0], cam[:, 0, 2], s=100, color="blue", marker="x")
            # goal
            plt.scatter(
                gt_goal[..., 0].flatten(),
                gt_goal[..., 2].flatten(),
                s=50,
                color="black",
                marker="o",
                alpha=0.5,
            )
            if i < len(reverse_samples) - 1:
                grad = reverse_grad[i]
                # aver over height
                grad = grad.reshape((yshape, -1, num_wps * num_kps, 3)).mean(0)
                xyz_sliced = xyz.reshape((yshape, -1, 3))[0]
                plt.quiver(
                    xyz_sliced[:, 0],
                    xyz_sliced[:, 2],
                    -grad[:, -1, 0],
                    -grad[:, -1, 2],
                    angles="xy",
                    scale_units="xy",
                    scale=30,  # inverse scaling
                    color=(0.5, 0.5, 0.5),
                )
            ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
            ax.text(0.0, 1.01, "Reverse process", transform=ax.transAxes, size=15)
            plt.xlim(xzmin, xzmax)
            plt.ylim(xzmin, xzmax)
            plt.axis("off")

            image = get_img_from_plt(fig)
            frames.append(image)
            plt.close(fig)

        filename = "%s/%s-animation" % (self.logdir, prefix)
        print("Animation saved to %s.mp4" % filename)
        save_vid(filename, frames)

    def run_viser(self):
        # visualizations
        server = viser.ViserServer(port=8081)
        # Setup root frame
        base_handle = server.add_frame(
            "/frames",
            wxyz=tf.SO3.exp(np.array([-np.pi / 2, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            show_axes=False,
        )

        server.add_mesh_trimesh(
            name="/frames/environment",
            mesh=self.bg_field.bg_mesh,
        )
        self.server = server

        self.root_visitation_boxes = self.bg_field.voxel_grid.to_boxes(
            mode="root_visitation"
        )
        self.cam_visitation = self.bg_field.voxel_grid.to_boxes(mode="cam_visitation")
        # server.add_mesh_trimesh(
        #     name="/frames/root_visitation",
        #     mesh=self.root_visitation_boxes,
        # )
        # server.add_mesh_trimesh(
        #     name="/frames/cam_visitation",
        #     mesh=self.cam_visitation,
        # )

        add_userwp_handle = server.add_gui_button("Add camera path")
        stop_add_user_handle = server.add_gui_button("Stop adding camera path")
        stop_add_user_handle.disabled = True

        add_goal_handle = server.add_gui_button("Add goal")
        stop_add_goal_handle = server.add_gui_button("Stop adding goal")
        stop_add_goal_handle.disabled = True

        delete_goal_handle = server.add_gui_button("Delete goal")

        @stop_add_user_handle.on_click
        def _(_):
            add_userwp_handle.disabled = False
            stop_add_user_handle.disabled = True
            server.remove_scene_click_callback(userwp_click_callback)

        @add_userwp_handle.on_click
        def _(_):
            add_userwp_handle.disabled = True
            stop_add_user_handle.disabled = False
            server.on_scene_click(userwp_click_callback)

        @stop_add_goal_handle.on_click
        def _(_):
            add_goal_handle.disabled = False
            stop_add_goal_handle.disabled = True
            server.remove_scene_click_callback(goal_click_callback)

        @add_goal_handle.on_click
        def _(_):
            add_goal_handle.disabled = True
            stop_add_goal_handle.disabled = False
            server.on_scene_click(goal_click_callback)

        @delete_goal_handle.on_click
        def _(_):
            self.goal_list = []
            server.remove_frame("/frames/control/goal_0")

        def get_intersection(message, mesh):
            # Check for intersection with the mesh, using trimesh's ray-mesh intersection.
            # Note that mesh is in the mesh frame, so we need to transform the ray.
            R_world_mesh = tf.SO3(base_handle.wxyz)
            R_mesh_world = R_world_mesh.inverse()
            origin = (R_mesh_world @ np.array(message.ray_origin)).reshape(1, 3)
            direction = (R_mesh_world @ np.array(message.ray_direction)).reshape(1, 3)
            # mesh = self.bg_field.bg_mesh
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            hit_pos, _, _ = intersector.intersects_location(origin, direction)

            if len(hit_pos) == 0:
                return []

            # Get the first hit position (based on distance from the ray origin).
            hit_pos = min(hit_pos, key=lambda x: np.linalg.norm(x - origin))
            # print(f"Hit position: {hit_pos}")
            return hit_pos

        # @server.on_scene_click
        def userwp_click_callback(message: viser.ScenePointerEvent) -> None:
            hit_pos = get_intersection(message, self.cam_visitation)
            if len(hit_pos) == 0:
                return
            # always maintain 2 hit points
            if len(self.userwp_list) == 2:
                self.userwp_list.pop(0)
            self.userwp_list.append(hit_pos)
            self.show_control_points()

        def goal_click_callback(message: viser.ScenePointerEvent) -> None:
            hit_pos = get_intersection(message, self.root_visitation_boxes)
            if len(hit_pos) == 0:
                return
            # always maintain 1 hit points
            if len(self.goal_list) == 1:
                self.goal_list.pop(0)
            self.goal_list.append(hit_pos)
            self.show_control_points(mode="goal")

    def show_control_points(self, mode="userwp"):
        if mode == "userwp":
            pts = self.userwp_list
            mesh = trimesh.creation.axis(
                origin_size=0.01, axis_length=0.1, axis_radius=0.005
            )
        elif mode == "goal":
            pts = self.goal_list
            mesh = trimesh.creation.uv_sphere(radius=0.05)
            mesh.visual.vertex_colors = np.array([[0, 0, 1.0, 1.0]]).repeat(
                len(mesh.vertices), axis=0
            )
        else:
            raise ValueError("Unknown mode")

        # add points
        if len(pts) > 0:
            for it, hit in enumerate(pts):
                if it == 0 and mode == "userwp":
                    continue
                mesh = mesh.apply_translation(hit)
                self.server.add_mesh_trimesh(f"/frames/control/{mode}_{it}", mesh)
        if len(pts) == 2:
            self.server.add_spline_catmull_rom(
                f"/frames/control/{mode}", np.array(pts), color=(1.0, 0.0, 1.0)
            )

    def render_trajectory_viser(self, goal, wp, past_wp, joints, angles, x0_to_world):
        """
        goal: bsx3
        future: bsxTx3
        past: T'x3
        past_joints: T'xKx3
        x0_to_world: 1x3
        """
        if torch.is_tensor(past_wp):
            past_wp = past_wp.cpu().numpy()
        if torch.is_tensor(wp):
            wp = wp.cpu().numpy()
        if torch.is_tensor(x0_to_world):
            x0_to_world = x0_to_world.cpu().numpy()
        if torch.is_tensor(goal):
            goal = goal.cpu().numpy()
        if torch.is_tensor(joints):
            joints = joints.cpu().numpy()
        if torch.is_tensor(angles):
            angles = angles.cpu().numpy()
        x0_to_world = x0_to_world.reshape(3)

        # add past
        self.server.add_point_cloud(
            f"/frames/past_pts",
            past_wp + x0_to_world,
            colors=(0.5, 0.5, 0.5),
            point_size=0.05,
        )
        # self.server.add_point_cloud(
        #     f"/frames/past_joints",
        #     past_joints + x0_to_world,
        #     colors=(0.8, 0.8, 0.8),
        #     point_size=0.01,
        # )

        # add goal
        goal = goal + x0_to_world
        for i in range(len(goal)):
            goal_mesh = trimesh.creation.uv_sphere(radius=0.05)
            goal_mesh = goal_mesh.apply_translation(goal[i])
            goal_mesh.visual.vertex_colors = np.array([[1.0, 0, 0, 0.5]]).repeat(
                len(goal_mesh.vertices), axis=0
            )
            self.server.add_mesh_trimesh(f"/frames/goal/{i}", goal_mesh)

        # self.server.add_point_cloud(
        #     f"/frames/goal",
        #     goal,
        #     point_size=0.1,
        #     colors=(1.0, 0, 0),
        # )

        # add future
        cmap = cm.get_cmap("cool")
        bs = len(wp)
        wp = wp.reshape(bs, -1, 3) + x0_to_world
        colors = cmap(np.linspace(0, 1, wp.shape[1]))[:, :3]
        for t in range(len(wp[0])):
            self.server.add_point_cloud(
                f"/frames/future/{t}",
                wp[:, t],
                point_size=0.02,
                colors=colors[t].reshape(1, 3).repeat(bs, axis=0),
            )

            if self.articulation_loader is not None:
                so3_wp_angles = np.concatenate(
                    [angles[t], wp[0, t, None], joints[t]], axis=0
                )
                so3_wp_angles = so3_wp_angles.reshape(1, -1)
                self.articulation_loader.load_files(so3_wp_angles)
                mesh = self.articulation_loader.mesh_dict[0]
                self.server.add_mesh_trimesh(f"/frames/joints", mesh)
                if t in np.asarray([15, 31, 47, 63]) - 8:
                    self.server.add_mesh_trimesh(f"/frames/joints_{t}", mesh)
            else:
                self.server.add_point_cloud(
                    f"/frames/joints",
                    joints[t] + x0_to_world,
                    colors=(0, 1.0, 0.0),
                    point_size=0.05,
                )
                time.sleep(0.05)
