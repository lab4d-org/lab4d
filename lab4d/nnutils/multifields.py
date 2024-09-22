# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from collections import defaultdict

import numpy as np
import torch
import trimesh
from torch import nn

from lab4d.nnutils.deformable import Deformable
from lab4d.nnutils.nerf import NeRF
from lab4d.nnutils.feature import FeatureNeRF
from lab4d.nnutils.bgnerf import BGNeRF
from lab4d.nnutils.pose import ArticulationSkelMLP, CameraMLP_so3
from lab4d.nnutils.warping import ComposedWarp, SkinningWarp
from lab4d.utils.quat_transform import quaternion_translation_to_se3
from lab4d.utils.vis_utils import draw_cams, mesh_cat
from lab4d.utils.geom_utils import extend_aabb


class MultiFields(nn.Module):
    """A container of neural fields.

    Args:
        data_info (Dict): Dataset metadata from get_data_info()
        field_type (str): Field type ("comp", "fg", or "bg")
        fg_motion (str): Foreground motion type ("rigid", "dense", "bob",
            "skel-{human,quad}", or "comp_skel-{human,quad}_{bob,dense}")
        single_inst (bool): If True, assume the same morphology over videos
        scene_type (str): one of {share-1, share-x, sep-x}
    """

    def __init__(
        self,
        data_info,
        field_type="bg",
        fg_motion="rigid",
        single_inst=True,
        scene_type="share-1",
        extrinsics_type="mlp",
        feature_channels=16,
        init_scale_fg=0.2,
        init_scale_bg=0.05,
        init_beta=0.1,
        num_freq_xyz=10,
        use_timesync=False,
        bg_vid=-1,
        use_cc=True,
    ):
        vis_info = data_info["vis_info"]

        super().__init__()
        field_params = nn.ParameterDict()
        self.field_type = field_type
        self.fg_motion = fg_motion
        self.single_inst = single_inst
        self.scene_type = scene_type
        self.extrinsic_type = extrinsics_type
        self.feature_channels = feature_channels
        self.init_scale_fg = init_scale_fg
        self.init_scale_bg = init_scale_bg
        self.init_beta=init_beta
        self.num_freq_xyz = num_freq_xyz
        self.use_timesync = use_timesync
        self.bg_vid = bg_vid
        self.use_cc = use_cc # whether to use connected components for fg

        # specify field type
        if field_type == "comp":
            # define a field per-category
            for category, tracklet_id in vis_info.items():
                field_params[category] = self.define_field(
                    category, data_info, tracklet_id
                )
        else:
            tracklet_id = vis_info[field_type]
            field_params[field_type] = self.define_field(
                field_type, data_info, tracklet_id
            )
        self.field_params = field_params

    def define_field(self, category, data_info, tracklet_id):
        """Define a new foreground or background neural field.

        Args:
            category (str): Field type ("fg" or "bg")
            data_info (Dict): Dataset metadata from get_data_info().
                This includes `data_info["rtmat"]`: NxTx4x4 camera matrices
            tracklet_id (int): Track index within a video
        """
        data_info = data_info.copy()
        # TODO: build a map from tracklet framid to video frameid
        # right now, the first dimension of rtmat is tracklet frameid
        # which is identical to video frameid if # instance=1
        data_info["rtmat"] = data_info["rtmat"][tracklet_id]
        data_info["geom_path"] = data_info["geom_path"][tracklet_id]
        num_inst = len(data_info["frame_info"]["frame_offset"]) - 1
        if category == "fg":
            # TODO add a flag to decide rigid fg vs deformable fg
            nerf = Deformable(
                self.fg_motion,
                data_info,
                num_freq_dir=-1,
                appr_channels=32,
                num_inst=1 if self.single_inst else num_inst,
                init_scale=self.init_scale_fg,
                init_beta=self.init_beta,
                extrinsics_type=self.extrinsic_type,
                feature_channels=self.feature_channels,
                use_timesync=self.use_timesync,
                invalid_vid=self.bg_vid,
            )
            # no directional encoding
        elif category == "bg":
            if self.scene_type.startswith("share"):
                bg_arch = FeatureNeRF
                num_inst_bg = 1 if self.scene_type.endswith("1") else num_inst
            else:
                bg_arch = BGNeRF
                num_inst_bg = num_inst
            # increase freq according to scale
            num_freq_xyz = int(np.log2(self.init_scale_bg / 0.05) + self.num_freq_xyz)
            nerf = bg_arch(
                data_info,
                D=8,
                skips=[1, 2, 3, 4, 5, 6, 7],
                num_freq_dir=0,
                appr_channels=0,
                num_inst=num_inst_bg,
                init_scale=self.init_scale_bg,
                init_beta=self.init_beta,
                num_freq_xyz=num_freq_xyz,
                extrinsics_type=self.extrinsic_type,
                feature_channels=self.feature_channels,
            )
        else:  # exit with an error
            raise ValueError("Invalid category")

        # mark type
        nerf.category = category
        nerf.use_cc = self.use_cc
        return nerf

    def mlp_init(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics for all child fields from external priors
        """
        for field in self.field_params.values():
            field.mlp_init()

    def set_alpha(self, alpha):
        """Set alpha values for all child fields

        Args:
            alpha (float or None): 0 to 1
        """
        for field in self.field_params.values():
            field.pos_embedding.set_alpha(alpha)
            field.pos_embedding_color.set_alpha(alpha)

    def set_importance_sampling(self, use_importance_sampling):
        """
        Set inverse sampling for all child fields
        """
        for field in self.field_params.values():
            field.use_importance_sampling = use_importance_sampling

    def set_symm_ratio(self, symm_ratio):
        """Set symmetry ratio for all child fields

        Args:
            symm_ratio (float): 0 to 1
        """
        for field in self.field_params.values():
            field.symm_ratio = symm_ratio

    def set_beta_prob(self, beta_prob_fg, beta_prob_bg):
        """Set beta probability for all child fields. This determines the
        probability of instance code swapping

        Args:
            beta_prob (float): Instance code swapping probability, 0 to 1
        """
        for category, field in self.field_params.items():
            if category=="fg":
                field.basefield.inst_embedding.set_beta_prob(beta_prob_fg)
            elif category=="bg":
                field.basefield.inst_embedding.set_beta_prob(beta_prob_bg)
            else:
                raise NotImplementedError

    def update_geometry_aux(self):
        """Update proxy geometry and bounds for all child fields"""
        for field in self.field_params.values():
            field.update_proxy()
            field.update_aabb()
            field.update_near_far()

    # def reset_geometry_aux(self):
    #     """Reset proxy geometry and bounds for all child fields"""
    #     for field in self.field_params.values():
    #         print("resetting geometry aux for %s" % field.category)
    #         field.update_proxy()
    #         field.update_aabb(beta=0)
    #         field.update_near_far(beta=0)

    def reset_beta(self, beta):
        """Reset beta for all child fields"""
        for field in self.field_params.values():
            field.reset_beta(beta)

    @torch.no_grad()
    def extract_canonical_meshes(
        self,
        grid_size=64,
        level=0.0,
        inst_id=None,
        vis_thresh=0.0,
        use_extend_aabb=True,
    ):
        """Extract canonical mesh using marching cubes for all child fields

        Args:
            grid_size (int): Marching cubes resolution
            level (float): Contour value to search for isosurfaces on the signed
                distance function
            inst_id: (int) Instance id. If None, extract for the average instance
            vis_thresh (float): threshold for visibility value to mask out invisible points.
            use_extend_aabb (bool): If True, extend aabb by 50% to get a loose proxy.
              Used at training time.
        Returns:
            meshes (Dict): Maps field types ("fg or bg") to extracted meshes
        """
        meshes = {}
        for category, field in self.field_params.items():
            mesh = field.extract_canonical_mesh(
                grid_size=grid_size,
                level=level,
                inst_id=inst_id,
                vis_thresh=vis_thresh,
                use_extend_aabb=use_extend_aabb,
            )
            if len(mesh.vertices) == 0:
                mesh = field.get_proxy_geometry()
            meshes[category] = mesh
        return meshes

    @torch.no_grad()
    def export_geometry_aux(self, path):
        """Export proxy geometry for all neural fields

        Args:
            path (str): Output path
        """
        for category, field in self.field_params.items():
            # print(field.near_far)
            mesh_geo = field.get_proxy_geometry()
            quat, trans = field.camera_mlp.get_vals()
            rtmat = quaternion_translation_to_se3(quat, trans).cpu()
            # evenly pick max 200 cameras
            if rtmat.shape[0] > 200:
                idx = np.linspace(0, rtmat.shape[0] - 1, 200).astype(np.int32)
                rtmat = rtmat[idx]
            mesh_cam = draw_cams(rtmat)
            mesh = mesh_cat(mesh_geo, mesh_cam)
            if category == "fg":
                aabb = extend_aabb(field.aabb, factor=0.5)
                mesh_gauss, mesh_sdf = field.warp.get_template_vis(aabb=aabb)
                mesh_gauss.export("%s-%s-gauss.obj" % (path, category))
                mesh_sdf.export("%s-%s-sdf.obj" % (path, category))
            mesh.export("%s-%s-proxy.obj" % (path, category))

    def visibility_decay_loss(self):
        """Compute mean visibility decay loss over all child fields.
        Encourage visibility to be low at random points within the aabb. The
        effect is that invisible / occluded points are assigned -inf visibility

        Returns:
            loss: (0,) Visibility decay loss
        """
        loss = []
        for field in self.field_params.values():
            loss.append(field.visibility_decay_loss())
        loss = torch.stack(loss, 0).sum(0).mean()
        return loss

    def gauss_skin_consistency_loss(self):
        """Compute mean Gauss skin consistency loss over all child fields.
        Enforce consistency between the NeRF's SDF and the SDF of Gaussian bones

        Returns:
            loss: (0,) Mean Gauss skin consistency loss
        """
        loss = []
        for field in self.field_params.values():
            if isinstance(field, Deformable) and isinstance(field.warp, SkinningWarp):
                loss.append(field.gauss_skin_consistency_loss())
        if len(loss) > 0:
            loss = torch.stack(loss, 0).mean()
        else:
            loss = torch.tensor(0.0, device=self.parameters().__next__().device)
        return loss

    def soft_deform_loss(self):
        """Compute average soft deformation loss over all child fields.
        Minimize soft deformation so it doesn't overpower the skeleton.
        Compute L2 distance of points before and after soft deformation

        Returns:
            loss: (0,) Soft deformation loss
        """
        loss = []
        for field in self.field_params.values():
            if isinstance(field, Deformable) and isinstance(field.warp, ComposedWarp):
                loss.append(field.soft_deform_loss())
        if len(loss) > 0:
            loss = torch.stack(loss, 0).mean()
        else:
            loss = torch.tensor(0.0, device=self.parameters().__next__().device)
        return loss

    def cam_prior_loss(self):
        """Compute mean camera prior loss over all child fields.
        Encourage camera transforms over time to match external priors.

        Returns:
            loss: (0,) Mean camera prior loss
        """
        loss = []
        for field in self.field_params.values():
            loss.append(field.cam_prior_loss())
        loss = torch.stack(loss, 0).sum(0).mean()
        return loss

    def cam_prior_relative_loss(self):
        """Compute mean camera prior loss over all child fields.
        Encourage camera transforms over time to match external priors.

        Returns:
            loss: (0,) Mean camera prior loss
        """
        loss = []
        for field in self.field_params.values():
            loss.append(field.cam_prior_relative_loss())
        loss = torch.stack(loss, 0).sum(0).mean()
        return loss

    def cam_smooth_loss(self):
        """Compute mean camera smoothness loss over all child fields.
        Encourage camera transforms over time to be smooth.

        Returns:
            loss: (0,) Mean camera smoothness loss
        """
        loss = []
        for field in self.field_params.values():
            loss.append(field.cam_smooth_loss())
        loss = torch.stack(loss, 0).sum(0).mean()
        return loss

    def skel_prior_loss(self):
        """Compute mean skeleton prior loss over all child fields.
        Encourage the skeleton rest pose to be near the pose initialization.
        Computes L2 loss on joint axis-angles and bone lengths.

        Returns:
            loss: (0,) Mean skeleton prior loss
        """
        loss = []
        for field in self.field_params.values():
            if (
                isinstance(field, Deformable)
                and isinstance(field.warp, SkinningWarp)
                and isinstance(field.warp.articulation, ArticulationSkelMLP)
            ):
                loss.append(field.warp.articulation.skel_prior_loss())
        if len(loss) > 0:
            loss = torch.stack(loss, 0).mean()
        else:
            loss = torch.tensor(0.0, device=self.parameters().__next__().device)
        return loss

    def get_samples(self, Kinv, batch):
        """Compute time-dependent camera and articulation parameters for all
        child fields.

        Args:
            Kinv: (N,3,3) Inverse camera intrinsics matrix
            batch (Dict): Batch of input metadata. Keys: "dataid",
                "frameid_sub", "crop2raw", "feature", "hxy", and "frameid"
        Returns:
            samples_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts containing input metadata and time-dependent outputs.
                Each dict has keys: "Kinv" (M,3,3), "field2cam" (M,4,4),
                "frame_id" (M,), "inst_id" (M,), "near_far" (M,2),
                "hxy" (M,N,2), and "feature" (M,N,16).
        """
        samples_dict = defaultdict(dict)
        for category, field in self.field_params.items():
            batch_sub = batch.copy()
            if "field2cam" in batch.keys():
                batch_sub["field2cam"] = batch["field2cam"][category]
            samples_dict[category] = field.get_samples(Kinv, batch_sub)
        return samples_dict

    def query_multifields(self, samples_dict, flow_thresh=None, n_depth=64):
        """Render outputs from all child fields.

        Args:
            samples_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts containing input metadata and time-dependent outputs.
                Each dict has keys: "Kinv" (M,3,3), "field2cam" (M,4,4),
                "frame_id" (M,), "inst_id" (M,), "near_far" (M,2),
                "hxy" (M,N,2), and "feature" (M,N,16).
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            multifields_dict (Dict): Maps neural field types to TODO
            deltas_dict (Dict): Maps neural field types to TODO
            aux_dict (Dict): Maps neural field types to
        """

        multifields_dict = {}
        deltas_dict = {}
        aux_dict = {}
        for category, field in self.field_params.items():
            (
                multifields_dict[category],
                deltas_dict[category],
                aux_dict[category],
            ) = field.query_field(
                samples_dict[category],
                flow_thresh=flow_thresh,
                n_depth=n_depth
            )
        return multifields_dict, deltas_dict, aux_dict

    @staticmethod
    def compose_fields(multifields_dict, deltas_dict):
        """Compose fields based on depth

        Args:
            multifields_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts of field outputs. Each dict has keys: "rgb" (M,N,D,3),
                "density" (M,N,D,1), "density_{fg,bg}" (M,N,D,1), "vis" (M,N,D,1),
                "cyc_dist" (M,N,D,1), "xyz" (M,N,D,3), "xyz_cam" (M,N,D,3),
                "depth" (M,1,D,1)
            deltas (Dict): Maps neural field types ("bg" or "fg") to (M,N,D,1)
                the distance along rays between adjacent samples.
        Returns:
            field_dict (Dict): Neural field outputs composed along rays. Keys:
                "rgb" (M, N, sum(D), 3), "density" (M, N, sum(D), 1),
                "density_{fg,bg}" (M, N, sum(D), 1), "vis" (M, N, sum(D), 1),
                "cyc_dist" (M, N, sum(D), 1), "xyz" (M, N, sum(D), 3),
                "xyz_cam" (M, N, sum(D), 3), "depth" (M, 1, sum(D), 1)
            deltas: (M, N, sum(D), 1) Distance along rays between adjacent samples
        """
        field_dict = {}
        deltas = []
        all_keys = []
        for i in multifields_dict.values():
            all_keys.extend(i.keys())
        all_keys = list(set(all_keys))

        for k in all_keys:
            field_dict[k] = []

        # append to field_dict with the same order
        for category, field in multifields_dict.items():
            for k in all_keys:
                if k in field.keys():
                    v = field[k]
                else:
                    v = None
                field_dict[k].append(v)
            deltas.append(deltas_dict[category])

        # cat
        for k, v in field_dict.items():
            if None in v:
                # find the index where v is not None
                not_none_idx = [i for i, x in enumerate(v) if x is not None]
                not_none_tensor = v[not_none_idx[0]]

                # replace None with zeros that share the same shape as non-None values
                v = [torch.zeros_like(not_none_tensor) if i is None else i for i in v]

            field_dict[k] = torch.cat(v, 2)
        deltas = torch.cat(deltas, 2)

        # sort
        if len(deltas_dict.keys()) > 1:
            z_idx = field_dict["depth"].argsort(2)
            for k, v in field_dict.items():
                field_dict[k] = torch.gather(v, 2, z_idx.expand_as(v))
            deltas = torch.gather(deltas, 2, z_idx.expand_as(deltas))
        return field_dict, deltas

    @torch.no_grad()
    def get_cameras(self, frame_id=None):
        """Compute camera matrices in world units

        Returns:
            field2cam (Dict): Maps field names ("fg" or "bg") to (M,4,4) cameras
        """
        field2cam = {}
        for cate, field in self.field_params.items():
            quat, trans = field.camera_mlp.get_vals(frame_id=frame_id)
            trans = trans / field.logscale.exp()
            field2cam[cate] = quaternion_translation_to_se3(quat, trans)
        return field2cam

    @torch.no_grad()
    def get_aabb(self, inst_id=None):
        """Compute axis aligned bounding box
        Args:
            inst_id (int or tensor): Instance id. If None, return aabb for all instances

        Returns:
            aabb (Dict): Maps field names ("fg" or "bg") to (1/N,2,3) aabb
        """
        if inst_id is not None:
            if not torch.is_tensor(inst_id):
                inst_id = torch.tensor(inst_id, dtype=torch.long)
            inst_id = inst_id.view(-1)
        aabb = {}
        for cate, field in self.field_params.items():
            aabb[cate] = field.get_aabb(inst_id=inst_id) / field.logscale.exp()
        return aabb
