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


sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import get_pts_traj
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.io import save_vid


def spline_interp(sample_wp, forecast_size=4, num_kps=1, state_size=3, interp_size=100):
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
        logdir="base",
        prefix="base",
    ):
        self.state_size = state_size
        self.xzmax = xzmax
        self.xzmin = xzmin
        self.num_timesteps = num_timesteps
        self.logdir = logdir
        self.prefix = prefix

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
        bg_field,
        x0_to_world,
    ):
        num_wps = int(len(forward_samples[0][0]) / self.state_size)
        num_timesteps = self.num_timesteps
        past = past.reshape(past.shape[0], -1, 3)
        # forward
        frames = []
        colors = np.ones((len(forward_samples[0]), num_wps, 4))  # N, K, 4
        colors[..., 1:3] = 0
        if num_wps > 1:
            colors = colors * np.linspace(0, 1, num_wps)[None, :, None]
        colors = colors.reshape(-1, 4)

        shape_clean = trimesh.PointCloud(
            forward_samples[0].reshape(-1, 3), colors=colors
        )
        frames += self.render_rotate_sample(shape_clean)

        for i, sample in enumerate(forward_samples):
            shape = trimesh.PointCloud(sample.reshape(-1, 3), colors=colors)
            input_dict = {"shape": shape}
            color = self.renderer.render(input_dict)[0]
            color = put_text(color, f"step {i: 4} / {num_timesteps}", (10, 30))
            color = put_text(color, "Forward process", (10, 60))
            frames.append(color)

        # reverse
        num_wps = int(len(reverse_samples[0][0]) / self.state_size)
        colors = np.ones((len(reverse_samples[0]), num_wps, 4))
        colors[..., 0:2] = 0
        if num_wps > 1:
            colors = colors * np.linspace(0, 1, num_wps)[None, :, None]
        colors = colors.reshape(-1, 4)

        # get past location as a sphere
        past_shape = []
        for past_idx in range(len(past[0])):
            shape_sub = trimesh.creation.uv_sphere(radius=0.03)
            shape_sub = shape_sub.apply_translation(past[0][past_idx].cpu().numpy())
            past_shape.append(shape_sub)
        past_shape = trimesh.util.concatenate(past_shape)
        past_colors = np.array([[0, 1, 0, 1]]).repeat(len(past_shape.vertices), axis=0)

        # get bg mesh
        bg_pts = bg_field.bg_mesh.vertices - x0_to_world[0].cpu().numpy()
        # bg_pts = (
        #     voxel_grid.to_boxes(mode="root_visitation").vertices - x0_to_world[0].cpu().numpy()
        # )
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
        # shape = concatenate_points(shape_clean, shape)
        frames += self.render_rotate_sample(
            shape, pts_traj=pts_traj, pts_color=pts_color
        )

        filename = "%s/%s-rendering" % (self.logdir, self.prefix)
        shape.export("%s.obj" % filename)
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
        sample_idx,
        xyz,
        y,
        past,
        cam,
    ):
        num_wps = int(len(forward_samples[0][0]) / self.state_size)
        num_timesteps = self.num_timesteps
        xzmin = self.xzmin
        xzmax = self.xzmax
        past = past.reshape(past.shape[0], -1, 3)
        cam = cam.reshape(cam.shape[0], -1, 3)

        # 2D plot
        fig, ax = plt.subplots()
        camera = Camera(fig)

        # forward
        for i, sample_goal in enumerate(forward_samples):
            sample_goal = sample_goal.reshape(-1, num_wps, 3)
            plt.scatter(
                sample_goal[:, -1, 0],
                sample_goal[:, -1, 2],
                alpha=0.5,
                s=15,
                color="blue",
            )
            plt.scatter(
                sample_goal[sample_idx, -1, 0],
                sample_goal[sample_idx, -1, 2],
                alpha=0.5,
                s=30,
                color="red",
            )
            ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
            ax.text(0.0, 1.01, "Forward process", transform=ax.transAxes, size=15)
            plt.xlim(xzmin, xzmax)
            plt.ylim(xzmin, xzmax)
            plt.axis("off")
            camera.snap()

        # reverse
        num_wps = int(len(reverse_samples[0][0]) / self.state_size)
        for i, sample_goal in enumerate(reverse_samples):
            sample_goal = sample_goal.reshape(-1, num_wps, 3)
            # draw the goal
            plt.scatter(
                sample_goal[:, -1, 0],
                sample_goal[:, -1, 2],
                alpha=0.5,
                s=100,
                color="red",
            )
            # draw all the waypoints
            for j in range(num_wps - 1):
                alpha = np.clip(1 - j / num_wps, 0.1, 1)
                s = np.clip(5 + 100 * j / num_wps, 5, 100)
                plt.scatter(
                    sample_goal[:, j, 0],
                    sample_goal[:, j, 2],
                    alpha=alpha,
                    s=s,
                    color="green",
                )
            # draw line passing through the waypoints
            plt.plot(
                sample_goal[:, :, 0].T, sample_goal[:, :, 2].T, color="green", alpha=0.5
            )

            plt.scatter(
                past[0, :, 0].cpu(),
                past[0, :, 2].cpu(),
                s=100,
                color="green",
                marker="x",
            )
            plt.scatter(
                cam[0, :, 0].cpu(), cam[0, :, 2].cpu(), s=100, color="blue", marker="x"
            )
            plt.scatter(
                gt_goal[0].cpu(), gt_goal[2].cpu(), s=100, color="black", marker="o"
            )
            if i < len(reverse_samples) - 1:
                grad = reverse_grad[i]
                # aver over height
                grad = grad.reshape(y.shape + (-1, num_wps, 3)).mean(0)
                xyz_sliced = xyz.reshape(y.shape + (-1, 3))[0]
                plt.quiver(
                    xyz_sliced[:, 0],
                    xyz_sliced[:, 2],
                    -grad[:, -1, 0],
                    -grad[:, -1, 2],
                    angles="xy",
                    scale_units="xy",
                    scale=10,
                    color="red",
                )
            ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
            ax.text(0.0, 1.01, "Reverse process", transform=ax.transAxes, size=15)
            plt.xlim(xzmin, xzmax)
            plt.ylim(xzmin, xzmax)
            plt.axis("off")
            camera.snap()

        animation = camera.animate(blit=True, interval=35)
        filename = "%s/%s-animation" % (self.logdir, self.prefix)
        animation.save("%s.mp4" % filename)
        print("Animation saved to %s.mp4" % filename)
