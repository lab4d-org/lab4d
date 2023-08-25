# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from lab4d.engine.train_utils import get_local_rank
from lab4d.nnutils.intrinsics import IntrinsicsMLP
from lab4d.nnutils.multifields import MultiFields
from lab4d.utils.geom_utils import K2inv, K2mat
from lab4d.utils.numpy_utils import interp_wt
from lab4d.utils.render_utils import render_pixel


class dvr_model(nn.Module):
    """A model that contains a collection of static/deformable neural fields

    Args:
        config (Dict): Command-line args
        data_info (Dict): Dataset metadata from get_data_info()
    """

    def __init__(self, config, data_info):
        super().__init__()
        self.config = config
        self.device = get_local_rank()
        self.data_info = data_info

        self.fields = MultiFields(
            data_info=data_info,
            field_type=config["field_type"],
            fg_motion=config["fg_motion"],
            single_inst=config["single_inst"],
        )
        self.intrinsics = IntrinsicsMLP(
            self.data_info["intrinsics"],
            frame_info=self.data_info["frame_info"],
            num_freq_t=0,
        )

    def mlp_init(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics for all neural fields from external priors
        """
        self.fields.mlp_init()
        self.intrinsics.mlp_init()

    def forward(self, batch):
        """Run forward pass and compute losses

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2),
                "hxy" (M,2,N,3), and "is_detected" (M,2)
        Returns:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,).
        """
        config = self.config
        self.process_frameid(batch)
        self.reshape_batch(batch)
        results = self.render(batch, flow_thresh=config["train_res"])
        loss_dict = self.compute_loss(batch, results)
        return loss_dict

    def process_frameid(self, batch):
        """Convert frameid within each video to overall frame id

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16), and
                "hxy" (M,N,3). This function modifies it in place to add key
                "frameid" (M,)
        """
        if not hasattr(self, "offset_cuda"):
            self.offset_cache = torch.tensor(
                self.data_info["frame_info"]["frame_offset_raw"],
                device=self.device,
                dtype=torch.long,
            )
        # convert frameid_sub to frameid
        batch["frameid"] = batch["frameid_sub"] + self.offset_cache[batch["dataid"]]

    def set_progress(self, current_steps):
        """Adjust loss weights and other constants throughout training

        Args:
            current_steps (int): Number of optimization steps so far
        """
        if self.config["use_freq_anneal"]:
            # positional encoding annealing
            anchor_x = (0, 4000)
            anchor_y = (0.6, 1)
            type = "linear"
            alpha = interp_wt(anchor_x, anchor_y, current_steps, type=type)
            if alpha >= 1:
                alpha = None
            self.fields.set_alpha(alpha)

        # anneal geometry/appearance code for foreground: steps(0->2k, 1->0.2), range (0.2,1)
        anchor_x = (0, 2000)
        anchor_y = (1.0, 0.2)
        type = "linear"
        beta_prob = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        self.fields.set_beta_prob(beta_prob)

        # camera prior wt: steps(0->800, 1->0), range (0,1)
        loss_name = "reg_cam_prior_wt"
        anchor_x = (0, 800)
        anchor_y = (1, 0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # reg_eikonal_wt: steps(0->24000, 1->100), range (1,100)
        loss_name = "reg_eikonal_wt"
        anchor_x = (0, 4000)
        anchor_y = (1, 100)
        type = "log"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # # skel prior wt: steps(0->4000, 1->0), range (0,1)
        # loss_name = "reg_skel_prior_wt"
        # anchor_x = (0, 4000)
        # anchor_y = (1, 0)
        # type = "linear"
        # self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # # gauss mask wt: steps(0->4000, 1->0), range (0,1)
        # loss_name = "reg_gauss_mask_wt"
        # anchor_x = (0, 4000)
        # anchor_y = (1, 0)
        # type = "linear"
        # self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

    def set_loss_weight(
        self, loss_name, anchor_x, anchor_y, current_steps, type="linear"
    ):
        """Set a loss weight according to the current training step

        Args:
            loss_name (str): Name of loss weight to set
            anchor_x: Tuple of optimization steps [x0, x1]
            anchor_y: Tuple of loss values [y0, y1]
            current_steps (int): Current optimization step
            type (str): Interpolation type ("linear" or "log")
        """
        if "%s_init" % loss_name not in self.config.keys():
            self.config["%s_init" % loss_name] = self.config[loss_name]
        factor = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        self.config[loss_name] = self.config["%s_init" % loss_name] * factor

    def evaluate(self, batch, is_pair=True):
        """Evaluate a Lab4D model

        Args:
            batch (Dict): Dataset metadata from `construct_eval_batch()`. Keys:
                "dataid" (M,), "frameid_sub" (M,), "crop2raw" (M,4),
                "feature" (M,N,16), and "hxy" (M,N,3)
            is_pair (bool): Whether to evaluate by rendering pairs
        Returns:
            rendered (Dict): Dict of rendered outputs. Keys: "mask" (M,H,W,1),
                "vis" (M,H,W,1), "depth" (M,H,W,1), "flow" (M,H,W,2),
                "feature" (M,H,W,16), "normal" (M,H,W,3), and
                "eikonal" (M,H,W,1)
        """
        if is_pair:
            div_factor = 2
        else:
            div_factor = 1
        self.process_frameid(batch)

        rendered = defaultdict(list)
        # split batch
        for i in tqdm(range(0, len(batch["frameid"]) // div_factor)):
            batch_sub = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    batch_sub[k] = {}
                    for k2, v2 in v.items():
                        batch_sub[k][k2] = v2[i * div_factor : (i + 1) * div_factor]
                else:
                    batch_sub[k] = v[i * div_factor : (i + 1) * div_factor]
            rendered_sub = self.render(batch_sub)["rendered"]
            for k, v in rendered_sub.items():
                res = int(np.sqrt(v.shape[1]))
                rendered[k].append(v.view(div_factor, res, res, -1)[0])

        for k, v in rendered.items():
            rendered[k] = torch.stack(v, 0)

        # blend with mask: render = render * mask + 0*(1-mask)
        for k, v in rendered.items():
            if "mask" in k:
                continue
            else:
                rendered[k] = rendered[k] * rendered["mask"]
        return rendered

    def update_geometry_aux(self):
        """Extract proxy geometry for all neural fields"""
        self.fields.update_geometry_aux()

    def export_geometry_aux(self, path):
        """Export proxy geometry for all neural fields"""
        return self.fields.export_geometry_aux(path)

    def render(self, batch, flow_thresh=None):
        """Render model outputs

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16),
                "hxy" (M,N,3), and "frameid" (M,)
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
            results["rendered"]: "mask" (M,N,1), "rgb" (M,N,3),
                "vis" (M,N,1), "depth" (M,N,1), "flow" (M,N,2),
                "feature" (M,N,16), "normal" (M,N,3), and "eikonal" (M,N,1)
            results["aux_dict"]["fg"]: "xy_reproj" (M,N,2) and "feature" (M,N,16)
        """
        samples_dict = self.get_samples(batch)
        results = self.render_samples_chunk(samples_dict, flow_thresh=flow_thresh)
        return results

    def get_samples(self, batch):
        """Compute time-dependent camera and articulation parameters for all
        neural fields.

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16),
                "hxy" (M,N,3), and "frameid" (M,)
        Returns:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,4,4), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,3), and
                "feature" (M,N,16).
        """
        if "Kinv" in batch.keys():
            Kinv = batch["Kinv"]
        else:
            Kmat = self.intrinsics.get_vals(batch["frameid"])
            Kinv = K2inv(Kmat) @ K2mat(batch["crop2raw"])

        samples_dict = self.fields.get_samples(Kinv, batch)
        return samples_dict

    def render_samples_chunk(self, samples_dict, flow_thresh=None, chunk_size=16384):
        """Render outputs from all neural fields. Divide in chunks along pixel
        dimension N to avoid running out of memory.

        Args:
            samples_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts of input metadata and time-dependent outputs.
                Each dict has keys: "Kinv" (M,3,3), "field2cam" (M,4,4),
                "frame_id" (M,), "inst_id" (M,), "near_far" (M,2),
                "hxy" (M,N,3), and "feature" (M,N,16).
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
            chunk_size (int): Number of pixels to render per chunk
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        # get chunk size
        category = list(samples_dict.keys())[0]
        total_pixels = (
            samples_dict[category]["hxy"].shape[0]
            * samples_dict[category]["hxy"].shape[1]
        )
        num_chunks = int(np.ceil(total_pixels / chunk_size))
        chunk_size_n = int(
            np.ceil(chunk_size // samples_dict[category]["hxy"].shape[0])
        )  # at n dimension

        results = {
            "rendered": defaultdict(list),
            "aux_dict": defaultdict(defaultdict),
        }
        for i in range(num_chunks):
            # construct chunk input
            samples_dict_chunk = defaultdict(list)
            for category, category_v in samples_dict.items():
                samples_dict_chunk[category] = defaultdict(list)
                for k, v in category_v.items():
                    if k == "hxy":
                        samples_dict_chunk[category][k] = v[
                            :, i * chunk_size_n : (i + 1) * chunk_size_n
                        ]
                    else:
                        samples_dict_chunk[category][k] = v

            # get chunk output
            results_chunk = self.render_samples(
                samples_dict_chunk, flow_thresh=flow_thresh
            )

            # merge chunk output
            for k, v in results_chunk["rendered"].items():
                if k not in results["rendered"].keys():
                    results["rendered"][k] = []
                results["rendered"][k].append(v)

            for cate in results_chunk["aux_dict"].keys():
                for k, v in results_chunk["aux_dict"][cate].items():
                    if k not in results["aux_dict"][cate].keys():
                        results["aux_dict"][cate][k] = []
                    results["aux_dict"][cate][k].append(v)

        # concat chunk output
        for k, v in results["rendered"].items():
            results["rendered"][k] = torch.cat(v, 1)

        for cate in results["aux_dict"].keys():
            for k, v in results["aux_dict"][cate].items():
                results["aux_dict"][cate][k] = torch.cat(v, 1)
        return results

    def render_samples(self, samples_dict, flow_thresh=None):
        """Render outputs from all neural fields.

        Args:
            samples_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts of input metadata and time-dependent outputs.
                Each dict has keys: "Kinv" (M,3,3), "field2cam" (M,4,4),
                "frame_id" (M,), "inst_id" (M,), "near_far" (M,2),
                "hxy" (M,N,3), and "feature" (M,N,16).
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        multifields_dict, deltas_dict, aux_dict = self.fields.query_multifields(
            samples_dict, flow_thresh=flow_thresh
        )

        field_dict, deltas = self.fields.compose_fields(multifields_dict, deltas_dict)
        rendered = render_pixel(field_dict, deltas)

        for cate in multifields_dict.keys():
            # render each field and put into aux_dict
            rendered_cate = render_pixel(multifields_dict[cate], deltas_dict[cate])
            for k, v in rendered_cate.items():
                aux_dict[cate][k] = v

        if "fg" in aux_dict.keys():
            # move for visualization
            if "xyz_matches" in aux_dict["fg"].keys():
                rendered["xyz_matches"] = aux_dict["fg"]["xyz_matches"]
                rendered["xyz_reproj"] = aux_dict["fg"]["xyz_reproj"]

        results = {"rendered": rendered, "aux_dict": aux_dict}
        return results

    @staticmethod
    def reshape_batch(batch):
        """Reshape a batch to merge the pair dimension into the batch dimension

        Args:
            batch (Dict): Arbitrary dataloader outputs (M, 2, ...). This is
                modified in place to reshape each value to (M*2, ...)
        """
        for k, v in batch.items():
            batch[k] = v.view(-1, *v.shape[2:])

    def compute_loss(self, batch, results):
        """Compute model losses

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        Returns:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,).
        """
        config = self.config
        loss_dict = {}
        self.compute_recon_loss(loss_dict, results, batch, config)
        self.mask_losses(loss_dict, batch, config)
        self.compute_reg_loss(loss_dict, results)
        self.apply_loss_weights(loss_dict, config)
        return loss_dict

    @staticmethod
    def get_mask_balance_wt(mask, vis2d, is_detected):
        """Balance contribution of positive and negative pixels in mask.

        Args:
            mask: (M,N,1) Object segmentation mask
            vis2d: (M,N,1) Whether each pixel is visible in the video frame
            is_detected: (M,) Whether there is segmentation mask in the frame
        Returns:
            mask_balance_wt: (M,N,1) Balanced mask
        """
        # all the positive labels
        mask = mask.float()
        # all the labels
        vis2d = vis2d.float() * is_detected.float()[:, None, None]
        if mask.sum() > 0 and (1 - mask).sum() > 0:
            pos_wt = vis2d.sum() / mask[vis2d > 0].sum()
            neg_wt = vis2d.sum() / (1 - mask[vis2d > 0]).sum()
            mask_balance_wt = 0.5 * pos_wt * mask + 0.5 * neg_wt * (1 - mask)
        else:
            mask_balance_wt = 1
        return mask_balance_wt

    @staticmethod
    def compute_recon_loss(loss_dict, results, batch, config):
        """Compute reconstruction losses.

        Args:
            loss_dict (Dict): Updated in place to add keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1), and
                "reg_gauss_mask" (M,N,1)
            results: Rendered outputs. Keys: "rendered", "aux_dict"
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            config (Dict): Command-line options
        """
        rendered = results["rendered"]
        aux_dict = results["aux_dict"]
        # reconstruction loss
        # get rendered fg mask
        if config["field_type"] == "fg":
            rendered_fg_mask = rendered["mask"]
        elif config["field_type"] == "comp":
            rendered_fg_mask = rendered["mask_fg"]
        elif config["field_type"] == "bg":
            rendered_fg_mask = None
        else:
            raise ("field_type %s not supported" % config["field_type"])
        # get fg mask balance factor
        mask_balance_wt = dvr_model.get_mask_balance_wt(
            batch["mask"], batch["vis2d"], batch["is_detected"]
        )
        if config["field_type"] == "bg":
            loss_dict["mask"] = (rendered["mask"] - 1).pow(2)
        elif config["field_type"] == "fg":
            loss_dict["mask"] = (rendered_fg_mask - batch["mask"].float()).pow(2)
            loss_dict["mask"] *= mask_balance_wt
        elif config["field_type"] == "comp":
            loss_dict["mask"] = (rendered_fg_mask - batch["mask"].float()).pow(2)
            loss_dict["mask"] *= mask_balance_wt
            loss_dict["mask"] += (rendered["mask"] - 1).pow(2)
        else:
            raise ("field_type %s not supported" % config["field_type"])

        if config["field_type"] == "fg" or config["field_type"] == "comp":
            loss_dict["feature"] = (aux_dict["fg"]["feature"] - batch["feature"]).norm(
                2, -1, keepdim=True
            )
            loss_dict["feat_reproj"] = (
                aux_dict["fg"]["xy_reproj"] - batch["hxy"][..., :2]
            ).norm(2, -1, keepdim=True)

        loss_dict["rgb"] = (rendered["rgb"] - batch["rgb"]).pow(2)
        loss_dict["depth"] = (
            (rendered["depth"] - batch["depth"]).norm(2, -1, keepdim=True).clone()
        )
        loss_dict["flow"] = (rendered["flow"] - batch["flow"]).norm(2, -1, keepdim=True)

        # visibility: supervise on fg and bg separately
        vis_loss = []
        # for aux_cate_dict in aux_dict.values():
        for cate, aux_cate_dict in aux_dict.items():
            if cate == "bg":
                # use smaller weight for bg
                aux_cate_dict["vis"] *= 0.01
            vis_loss.append(aux_cate_dict["vis"])
        vis_loss = torch.stack(vis_loss, 0).sum(0)
        loss_dict["vis"] = vis_loss

        # weighting
        loss_dict["flow"] = loss_dict["flow"] * (batch["flow_uct"] > 0).float()

        # consistency between rendered mask and gauss mask
        if "gauss_mask" in rendered.keys():
            loss_dict["reg_gauss_mask"] = (
                aux_dict["fg"]["gauss_mask"] - rendered_fg_mask.detach()
            ).pow(2)

    def compute_reg_loss(self, loss_dict, results):
        """Compute regularization losses.

        Args:
            loss_dict (Dict): Updated in place to add keys:
                "reg_visibility" (0,), "reg_eikonal" (0,),
                "reg_deform_cyc" (0,), "reg_soft_deform" (0,),
                "reg_gauss_skin" (0,), "reg_cam_prior" (0,), and
                "reg_skel_prior" (0,).
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        rendered = results["rendered"]
        aux_dict = results["aux_dict"]
        # regularization loss
        loss_dict["reg_visibility"] = self.fields.visibility_decay_loss()
        loss_dict["reg_eikonal"] = rendered["eikonal"]
        if "fg" in aux_dict.keys():
            loss_dict["reg_deform_cyc"] = aux_dict["fg"]["cyc_dist"]
            loss_dict["reg_delta_skin"] = aux_dict["fg"]["delta_skin"]
            loss_dict["reg_skin_entropy"] = aux_dict["fg"]["skin_entropy"]
        loss_dict["reg_soft_deform"] = self.fields.soft_deform_loss()
        loss_dict["reg_gauss_skin"] = self.fields.gauss_skin_consistency_loss()
        loss_dict["reg_cam_prior"] = self.fields.cam_prior_loss()
        loss_dict["reg_skel_prior"] = self.fields.skel_prior_loss()

    @staticmethod
    def mask_losses(loss_dict, batch, config):
        """Apply segmentation mask on dense losses

        Args:
            loss_dict (Dict): Dense losses. Keys: "mask" (M,N,1), "rgb" (M,N,3),
                "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1), "feature" (M,N,1),
                "feat_reproj" (M,N,1), and "reg_gauss_mask" (M,N,1). Modified in
                place to multiply loss_dict["mask"] with the other losses
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            config (Dict): Command-line options
        """
        # ignore the masking step
        keys_ignore_masking = ["reg_gauss_mask"]
        # always mask-out non-visible (out-of-frame) pixels
        keys_allpix = ["mask"]
        # always mask-out non-object pixels
        keys_fg = ["feature", "feat_reproj"]
        # field type specific keys
        keys_type_specific = ["rgb", "depth", "flow", "vis"]

        # type-specific masking rules
        vis2d = batch["vis2d"].float()
        maskfg = batch["mask"].float()
        if config["field_type"] == "bg":
            mask = (1 - maskfg) * vis2d
        elif config["field_type"] == "fg":
            mask = maskfg * vis2d
        elif config["field_type"] == "comp":
            mask = vis2d
        else:
            raise ("field_type %s not supported" % config["field_type"])

        # apply mask
        for k, v in loss_dict.items():
            if k in keys_ignore_masking:
                continue
            elif k in keys_allpix:
                loss_dict[k] = v * vis2d
            elif k in keys_fg:
                loss_dict[k] = v * maskfg
            elif k in keys_type_specific:
                loss_dict[k] = v * mask
            else:
                raise ("loss %s not defined" % k)

        # mask out the following losses if obj is not detected
        keys_mask_not_detected = ["mask", "feature", "feat_reproj"]
        for k, v in loss_dict.items():
            if k in keys_mask_not_detected:
                loss_dict[k] = v * batch["is_detected"].float()[:, None, None]

    @staticmethod
    def apply_loss_weights(loss_dict, config):
        """Weigh each loss term according to command-line configs

        Args:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,). Modified in
                place to multiply each term with a scalar weight.
            config (Dict): Command-line options
        """
        px_unit_keys = ["flow", "feat_reproj"]
        for k, v in loss_dict.items():
            # average over non-zero pixels
            v = v[v > 0]
            if v.numel() > 0:
                loss_dict[k] = v.mean()
            else:
                loss_dict[k] = v.sum()  # return zero

            # scale with image resolution
            if k in px_unit_keys:
                loss_dict[k] /= config["train_res"]

            # scale with loss weights
            wt_name = k + "_wt"
            if wt_name in config.keys():
                loss_dict[k] *= config[wt_name]
