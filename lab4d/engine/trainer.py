# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os
import time
from collections import defaultdict
from copy import deepcopy
import gc
import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True

from lab4d.dataloader import data_utils
from lab4d.dataloader.vidloader import VidDataset
from lab4d.engine.model import dvr_model
from lab4d.engine.train_utils import (
    DataParallelPassthrough,
    get_local_rank,
    match_param_name,
)
from lab4d.utils.profile_utils import torch_profile
from lab4d.utils.torch_utils import remove_ddp_prefix, resolve_size_mismatch, remove_state_startwith
from lab4d.utils.vis_utils import img2color, make_image_grid


class Trainer:
    def __init__(self, opts):
        """Train and evaluate a Lab4D model.

        Args:
            opts (Dict): Command-line args from absl (defined in lab4d/config.py)
        """
        # When profiling, use fewer iterations per round so trace files are smaller
        if opts["profile"]:
            opts["iters_per_round"] = 10

        self.opts = opts

        self.define_dataset()
        self.trainer_init()
        self.define_model()

        # move model to ddp
        self.move_to_ddp()

        self.optimizer_init(is_resumed=opts["load_path"] != "")

        # load model
        self.load_checkpoint_train()

    def move_to_ddp(self):
        # move model to ddp
        self.model = DataParallelPassthrough(
            self.model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=False,
        )

    def trainer_init(self):
        """Initialize logger and other misc things"""
        opts = self.opts

        logname = "%s-%s" % (opts["seqname"], opts["logname"])
        self.save_dir = os.path.join(opts["logroot"], logname)
        if get_local_rank() == 0:
            os.makedirs("tmp/", exist_ok=True)
            os.makedirs(self.save_dir, exist_ok=True)

            # tensorboard
            self.log = SummaryWriter(
                "%s/%s" % (opts["logroot"], logname), comment=logname
            )
        else:
            self.log = None

        self.current_steps = 0  # 0-total_steps
        self.current_round = 0  # 0-num_rounds
        self.first_round = 0  # 0
        self.first_step = 0  # 0

        # torch.manual_seed(8)  # do it again
        # torch.cuda.manual_seed(1)

    def define_dataset(self):
        """Construct training and evaluation dataloaders."""
        opts = self.opts
        train_dict = self.construct_dataset_opts(opts)
        self.trainloader = data_utils.train_loader(train_dict)

        eval_dict = self.construct_dataset_opts(opts, is_eval=True)
        self.evalloader = data_utils.eval_loader(eval_dict)

        self.data_info, self.data_path_dict = data_utils.get_data_info(self.evalloader)

        self.total_steps = opts["num_rounds"] * min(
            opts["iters_per_round"], len(self.trainloader)
        )

        # 0-last image in eval dataset
        self.eval_fid = np.linspace(0, len(self.evalloader) - 1, 9).astype(int)
        # self.eval_fid = np.linspace(1200, 1200, 9).astype(int)

    def init_model(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics from external priors, if this is the first run"""
        # init mlp
        if get_local_rank() == 0:
            self.model.mlp_init()

    def define_model(self, model=dvr_model):
        """Define a Lab4D model and wrap it with DistributedDataParallel"""
        opts = self.opts
        data_info = self.data_info

        self.device = torch.device("cuda:{}".format(get_local_rank()))
        self.model = model(opts, data_info)

        # ddp
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = self.model.to(self.device)

        self.init_model()

        # cache queue of length 2
        self.model_cache = [None, None]
        self.optimizer_cache = [None, None]
        self.scheduler_cache = [None, None]

        self.grad_queue = {}
        self.param_clip_startwith = {
            "module.fields.field_params.fg.camera_mlp": 10.0,
            "module.fields.field_params.fg.warp.articulation": 10.0,
            "module.fields.field_params.fg.basefield": 10.0,
            "module.fields.field_params.fg.sdf": 10.0,
            "module.fields.field_params.bg.camera_mlp": 10.0,
            "module.fields.field_params.bg.basefield": 10.0,
            "module.fields.field_params.bg.sdf": 10.0,
        }

    def get_lr_freeze_field_bg(self):
        param_lr_with_freeze_field_bg = {
            "module.fields.field_params.bg.basefield.": 0.0,
            # "module.fields.field_params.bg.colorfield.": 0.0,
            "module.fields.field_params.bg.sdf.": 0.0,
            # "module.fields.field_params.bg.rgb.": 0.0,
            "module.fields.field_params.bg.vis_mlp.": 0.0,
            "module.fields.field_params.bg.feature_field": 0.0,
        }
        return param_lr_with_freeze_field_bg

    def get_lr_freeze_field_fg(self):
        param_lr_with_freeze_field_bg = {
            "module.fields.field_params.fg.basefield.": 0.0,
            # "module.fields.field_params.fg.colorfield.": 0.0,
            "module.fields.field_params.fg.sdf.": 0.0,
            # "module.fields.field_params.fg.rgb.": 0.0,
            "module.fields.field_params.fg.vis_mlp.": 0.0,
            # "module.fields.field_params.fg.feature_field": 0.0,
        }
        return param_lr_with_freeze_field_bg

    def get_lr_freeze_field_fgbg(self):
        param_lr_with_freeze_field = {
            "module.fields.field_params.fg.basefield.": 0.0,
            # "module.fields.field_params.fg.colorfield.": 0.0,
            "module.fields.field_params.fg.sdf.": 0.0,
            # "module.fields.field_params.fg.rgb.": 0.0,
            "module.fields.field_params.fg.vis_mlp.": 0.0,
            "module.fields.field_params.fg.feature_field": 0.0,
            "module.fields.field_params.bg.basefield.": 0.0,
            # "module.fields.field_params.bg.colorfield.": 0.0,
            "module.fields.field_params.bg.sdf.": 0.0,
            # "module.fields.field_params.bg.rgb.": 0.0,
            "module.fields.field_params.bg.vis_mlp.": 0.0,
            "module.fields.field_params.bg.feature_field": 0.0,
        }
        return param_lr_with_freeze_field

    def get_lr_freeze_camera_bg(self):
        param_lr_with_freeze_camera_bg = {
            "module.fields.field_params.bg.camera_mlp": 0.0,
        }
        return param_lr_with_freeze_camera_bg

    def get_lr_freeze_camera_fg(self):
        param_lr_with_freeze_camera_fg = {
            "module.fields.field_params.fg.camera_mlp": 0.0,
        }
        return param_lr_with_freeze_camera_fg

    def get_lr_dict(self, pose_correction=False):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        opts = self.opts
        lr_base = opts["learning_rate"]
        lr_explicit = lr_base * 10
        lr_intrinsics = 0.0 if opts["freeze_intrinsics"] else lr_base

        param_lr_startwith = {
            "module.fields.field_params": lr_base,
            "module.intrinsics": lr_intrinsics,
        }
        param_lr_with = {
            ".logibeta": lr_explicit,
            ".logsigma": lr_explicit,
            ".logscale": lr_explicit,
            ".log_gauss": 0.0,
            ".base_quat": 0.0,
            ".shift": lr_explicit,
            ".orient": lr_explicit,
        }

        if opts["freeze_field_bg"]:
            param_lr_with.update(self.get_lr_freeze_field_bg())
        if opts["freeze_field_fg"]:
            param_lr_with.update(self.get_lr_freeze_field_fg())   
        if opts["freeze_field_fgbg"]:
            param_lr_with.update(self.get_lr_freeze_field_fgbg())
        if opts["freeze_camera_bg"]:
            param_lr_with.update(self.get_lr_freeze_camera_bg())
        if opts["freeze_camera_fg"]:
            param_lr_with.update(self.get_lr_freeze_camera_fg())
        if opts["freeze_scale"]:
            del param_lr_with[".logscale"]
            param_lr_with_freeze_scale = {
                "module.fields.field_params.bg.logscale": 0.0,
            }
            param_lr_with.update(param_lr_with_freeze_scale)

        if pose_correction:
            del param_lr_with[".logscale"]
            del param_lr_with[".log_gauss"]
            param_lr_with_pose_correction = {
                "module.fields.field_params.fg.basefield.": 0.0,
                "module.fields.field_params.fg.sdf.": 0.0,
                "module.fields.field_params.fg.feature_field": 0.0,
                "module.fields.field_params.fg.warp.skinning_model": 0.0,
            }
            param_lr_with.update(param_lr_with_pose_correction)

        return param_lr_startwith, param_lr_with

    def optimizer_init(self, is_resumed=False):
        """Set the learning rate for all trainable parameters and initialize
        the optimizer and scheduler.

        Args:
            is_resumed (bool): True if resuming from checkpoint
        """
        opts = self.opts
        param_lr_startwith, param_lr_with = self.get_lr_dict(
            pose_correction=opts["pose_correction"]
        )
        self.params_ref_list, params_list, lr_list = self.get_optimizable_param_list(
            param_lr_startwith, param_lr_with
        )
        # # one cycle lr
        # self.optimizer = torch.optim.AdamW(
        #     params_list,
        #     lr=opts["learning_rate"],
        #     betas=(0.9, 0.999),
        #     weight_decay=1e-4,
        # )
        # # initial_lr = lr/div_factor
        # # min_lr = initial_lr/final_div_factor
        # # if is_resumed:
        # if False:
        #     div_factor = 1.0
        #     final_div_factor = 25.0
        #     pct_start = 0.0  # cannot be 0
        # else:
        #     div_factor = 25.0
        #     final_div_factor = 1.0
        #     pct_start = min(
        #         1 - 1e-5, 2.0 / opts["num_rounds"]
        #     )  # use 2 epochs to warm up
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     lr_list,
        #     int(self.total_steps),
        #     pct_start=pct_start,
        #     cycle_momentum=False,
        #     anneal_strategy="linear",
        #     div_factor=div_factor,
        #     final_div_factor=final_div_factor,
        # )

        # cyclic lr
        assert self.total_steps // 2000 * 2000 == self.total_steps  # dividible by 2k
        self.optimizer = torch.optim.AdamW(
            params_list,
            lr=opts["learning_rate"],
            betas=(0.9, 0.99),
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            [i * 0.01 for i in lr_list],
            lr_list,
            step_size_up=10,
            step_size_down=1990,
            mode="triangular",
            gamma=1.0,
            scale_mode="cycle",
            cycle_momentum=False,
        )

    def get_optimizable_param_list(self, param_lr_startwith, param_lr_with):
        """
        Get the optimizable param list
        Returns:
            params_ref_list (List): List of params
            params_list (List): List of params
            lr_list (List): List of learning rates
        """
        params_ref_list = []
        params_list = []
        lr_list = []

        for name, p in self.model.named_parameters():
            matched_loose, lr_loose = match_param_name(name, param_lr_with, type="with")
            matched_strict, lr_strict = match_param_name(
                name, param_lr_startwith, type="startwith"
            )
            if matched_loose > 0:
                lr = lr_loose  # higher priority
            elif matched_strict > 0:
                lr = lr_strict
            else:
                lr = 0.0  # not found
                # print(name, "not found")
            if lr > 0:
                params_ref_list.append({name: p})
                params_list.append({"params": p, "lr": lr})
                lr_list.append(lr)
                if get_local_rank() == 0:
                    print(name, p.shape, lr)

        return params_ref_list, params_list, lr_list

    def train(self):
        """Training loop"""
        opts = self.opts

        # clear buffers for pytorch1.10+
        try:
            self.model._assign_modules_buffers()
        except:
            pass

        # start training loop
        self.save_checkpoint(round_count=self.current_round)
        for _ in range(self.current_round, self.current_round + opts["num_rounds"]):
            start_time = time.time()
            with torch_profile(
                self.save_dir, f"{self.current_round:03d}", enabled=opts["profile"]
            ):
                self.run_one_round()

            if get_local_rank() == 0:
                print(
                    f"Round {self.current_round:03d}: time={time.time() - start_time:.3f}s"
                )
            self.save_checkpoint(round_count=self.current_round)

    def update_aux_vars(self):
        self.model.update_geometry_aux()
        self.model.export_geometry_aux("%s/%03d" % (self.save_dir, self.current_round))
        if (
            self.current_round > self.opts["num_rounds_cam_init"]
            and self.opts["absorb_base"]
        ):
            self.model.update_camera_aux()

    def run_one_round(self):
        """Evaluation and training for a single round"""
        if get_local_rank() == 0:
            if self.current_round == self.first_round:
                self.model_eval()

        self.update_aux_vars()

        self.model.train()
        self.train_one_round()
        self.current_round += 1
        if get_local_rank() == 0:
            self.model_eval()

    def save_checkpoint(self, round_count):
        """Save model checkpoint to disk

        Args:
            round_count (int): Current round index
        """
        opts = self.opts
        # move to the left
        self.model_cache[0] = self.model_cache[1]
        self.optimizer_cache[0] = self.optimizer_cache[1]
        self.scheduler_cache[0] = self.scheduler_cache[1]
        # enqueue
        self.model_cache[1] = deepcopy(self.model.state_dict())
        self.optimizer_cache[1] = deepcopy(self.optimizer.state_dict())
        self.scheduler_cache[1] = deepcopy(self.scheduler.state_dict())

        if get_local_rank() == 0 and round_count % opts["save_freq"] == 0:
            print("saving round %d" % round_count)
            param_path = "%s/ckpt_%04d.pth" % (self.save_dir, round_count)

            checkpoint = {
                "current_steps": self.current_steps,
                "current_round": self.current_round,
                "model": self.model_cache[1],
                "optimizer": self.optimizer_cache[1],
            }

            torch.save(checkpoint, param_path)
            # copy to latest
            latest_path = "%s/ckpt_latest.pth" % (self.save_dir)
            os.system("cp %s %s" % (param_path, latest_path))

    @staticmethod
    def load_checkpoint(load_path, model, optimizer=None, load_camera=True):
        """Load a model from checkpoint

        Args:
            load_path (str): Path to checkpoint
            model (dvr_model): Model to update in place
            optimizer (torch.optim.Optimizer or None): If provided, load
                learning rate from checkpoint
        """
        checkpoint = torch.load(load_path, map_location="cpu")
        model_states = checkpoint["model"]
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_states = remove_ddp_prefix(model_states)

        if not load_camera:
            model_states = remove_state_startwith(model_states, "module.fields.field_params.fg.camera_mlp")

        resolve_size_mismatch(model, model_states)

        model.load_state_dict(model_states, strict=False)

        # # reset near_far
        # if hasattr(model, "fields"):
        #     model.fields.reset_geometry_aux()

        # if optimizer is not None:
        #     # use the new param_groups that contains the learning rate
        #     checkpoint["optimizer"]["param_groups"] = optimizer.state_dict()[
        #         "param_groups"
        #     ]
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint

    def load_checkpoint_train(self):
        """Load a checkpoint at training time and update the current step count
        and round count
        """
        # if self.opts["load_path_bg"] != "":
        #     # load background and intrinsics model
        #     checkpoint = torch.load(self.opts["load_path_bg"])
        #     model_states = checkpoint["model"]
        #     self.model.load_state_dict(model_states, strict=False)

        if self.opts["load_path"] != "":
            # training time
            checkpoint = self.load_checkpoint(
                self.opts["load_path"], self.model, optimizer=self.optimizer, load_camera=self.opts["load_fg_camera"]
            )
            if not self.opts["reset_steps"]:
                self.current_steps = checkpoint["current_steps"]
                self.current_round = checkpoint["current_round"]
                self.first_round = self.current_round
                self.first_step = self.current_steps

        if self.opts["load_path_bg"] != "":
            # load background and intrinsics model
            checkpoint = torch.load(self.opts["load_path_bg"])
            model_states = checkpoint["model"]
            resolve_size_mismatch(self.model, model_states)
            self.model.load_state_dict(model_states, strict=False)

        if self.opts["reset_beta"] > 0.0:
            self.model.module.fields.reset_beta(beta=self.opts["reset_beta"])

        # self.model.fields.reset_geometry_aux()

    def train_one_round(self):
        """Train a single round (going over mini-batches)"""
        opts = self.opts
        gc.collect()  # need to be used together with empty_cache()
        torch.cuda.empty_cache()
        self.model.train()
        self.optimizer.zero_grad()

        # necessary for shuffling
        self.trainloader.sampler.set_epoch(self.current_round)
        for i, batch in tqdm.tqdm(enumerate(self.trainloader)):
            if i == opts["iters_per_round"]:
                break

            progress = (self.current_steps - self.first_step) / self.total_steps
            self.model.set_progress(self.current_steps, progress)

            self.model.convert_img_to_pixel(batch)

            loss_dict = self.model(batch)
            total_loss = torch.sum(torch.stack(list(loss_dict.values())))
            total_loss.mean().backward()
            # print(total_loss)
            # self.print_sum_params()

            grad_dict = self.check_grad()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if get_local_rank() == 0:
                # update scalar dict
                # move all to loss
                new_loss_dict = {}
                for k, v in loss_dict.items():
                    new_loss_dict["loss/%s" % k] = v
                del loss_dict
                loss_dict = new_loss_dict
                loss_dict["loss/total"] = total_loss
                loss_dict.update(self.model.get_field_params())
                loss_dict.update(grad_dict)
                self.add_scalar(self.log, loss_dict, self.current_steps)
            self.current_steps += 1

    @staticmethod
    def construct_dataset_opts(opts, is_eval=False, dataset_constructor=VidDataset):
        """Extract train/eval dataloader options from command-line args.

        Args:
            opts (Dict): Command-line options
            is_eval (bool): When training a model (`is_eval=False`), duplicate
                the dataset to fix the number of iterations per round
            dataset_constructor (torch.utils.data.Dataset): Dataset class to use
        """
        opts_dict = {}
        opts_dict["seqname"] = opts["seqname"]
        opts_dict["load_pair"] = True
        opts_dict["data_prefix"] = "%s-%d" % (opts["data_prefix"], opts["train_res"])
        opts_dict["feature_type"] = opts["feature_type"]
        opts_dict["field_type"] = opts["field_type"]
        opts_dict["dataset_constructor"] = dataset_constructor

        if is_eval:
            opts_dict["multiply"] = False
            opts_dict["pixels_per_image"] = -1
            opts_dict["delta_list"] = []
            opts_dict["res"] = opts["eval_res"]
        else:
            # duplicate dataset to fix number of iterations per round
            opts_dict["multiply"] = True
            opts_dict["pixels_per_image"] = opts["pixels_per_image"]
            opts_dict["delta_list"] = (
                [int(x) for x in opts["delta_list"].split(",")]
                if opts["delta_list"] != ","
                else []
            )
            opts_dict["res"] = opts["train_res"]
            opts_dict["num_workers"] = opts["num_workers"]

            opts_dict["imgs_per_gpu"] = opts["imgs_per_gpu"]
            opts_dict["iters_per_round"] = opts["iters_per_round"]
            opts_dict["ngpu"] = opts["ngpu"]
            opts_dict["local_rank"] = get_local_rank()
        return opts_dict

    def print_sum_params(self):
        """Print the sum of parameters"""
        sum = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                sum += p.abs().sum()
        print(f"{sum:.16f}")

    @torch.no_grad()
    def model_eval(self):
        """Evaluate the current model"""
        self.model.eval()
        gc.collect()  # need to be used together with empty_cache()
        torch.cuda.empty_cache()
        ref_dict, batch = self.load_batch(self.evalloader.dataset, self.eval_fid)
        self.construct_eval_batch(batch)
        rendered, scalars = self.model.evaluate(batch)
        self.add_image_togrid(ref_dict)
        self.add_image_togrid(rendered)
        # if "xyz" in rendered.keys():
        #     self.visualize_matches(rendered["xyz"], rendered["xyz_matches"], tag="xyz")
        #     self.visualize_matches(
        #         rendered["xyz_cam"], rendered["xyz_reproj"], tag="xyz_cam"
        #     )
        self.add_scalar(self.log, scalars, self.current_round)

    def visualize_matches(self, xyz, xyz_matches, tag):
        """Visualize dense correspondences outputted by canonical registration

        Args:
            xyz: (M,H,W,3) Predicted xyz points
            xyz_matches: (M,H,W,3) Points to match against in canonical space.
                This is an empty list for the static background model
            tag (str): Name of export mesh
        """
        if len(xyz_matches) == 0:
            return
        xyz = xyz[6].view(-1, 3).detach().cpu().numpy()
        xyz_matches = xyz_matches[6].view(-1, 3).detach().cpu().numpy()

        nsample = 100
        idx = np.random.permutation(len(xyz))[:nsample]
        xyz = xyz[idx]
        xyz_matches = xyz_matches[idx]
        # draw lines
        lines = []
        for i in range(nsample):
            segment = np.stack([xyz[i], xyz_matches[i]], 0)
            line = trimesh.creation.cylinder(0.0001, segment=segment, sections=5)
            lines.append(line)
        lines = trimesh.util.concatenate(lines)

        xyz = trimesh.Trimesh(vertices=xyz)
        xyz_matches = trimesh.Trimesh(vertices=xyz_matches)
        xyz.visual.vertex_colors = [255, 0, 0, 255]  # red
        xyz_matches.visual.vertex_colors = [0, 255, 0, 255]  # green

        xyz_cat = trimesh.util.concatenate([xyz, xyz_matches])

        xyz_cat.export("%s/%03d-%s.obj" % (self.save_dir, self.current_round, tag))
        lines.export("%s/%03d-%s-lines.obj" % (self.save_dir, self.current_round, tag))

    @staticmethod
    def load_batch(dataset, fids):
        """Load a single batch of reference frames for Tensorboard visualization

        Args:
            dataset (ConcatDataset): Eval dataset for all videos in a sequence
            fids: (nframes,) Frame indices to load
        Returns:
            ref_dict (Dict): Dict with keys "ref_rgb", "ref_mask", "ref_depth",
                "ref_feature", and "ref_flow", each (N,H,W,x)
            batch_aggr (Dict): Batch of input metadata. Keys: "dataid",
                "frameid_sub", "crop2raw", and "feature"
        """
        ref_dict = defaultdict(list)
        batch_aggr = defaultdict(list)
        ref_keys = ["rgb", "mask", "depth", "feature", "vis2d", "normal"]
        batch_keys = ["dataid", "frameid_sub", "crop2raw"]
        for fid in fids:
            batch = dataset[fid]
            for k in ref_keys:
                ref_dict["ref_%s" % k].append(batch[k][:1])
            ref_dict["ref_flow"].append(
                batch["flow"][:1] * (batch["flow_uct"][:1] > 0).astype(float)
            )

            for k in batch_keys:
                batch_aggr[k].append(batch[k])
            batch_aggr["feature"].append(
                batch["feature"].reshape(2, -1, batch["feature"].shape[-1])
            )

        for k, v in ref_dict.items():
            ref_dict[k] = np.concatenate(v, 0)

        for k, v in batch_aggr.items():
            batch_aggr[k] = np.concatenate(v, 0)
        return ref_dict, batch_aggr

    def construct_eval_batch(self, batch):
        """Modify a batch in-place for evaluation

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid",
                "frameid_sub", "crop2raw", and "feature". This function
                modifies it in place to add key "hxy"
        """
        opts = self.opts
        # to tensor
        for k, v in batch.items():
            batch[k] = torch.tensor(v, device=self.device)

        batch["crop2raw"][..., :2] *= opts["train_res"] / opts["eval_res"]

        if not hasattr(self, "hxy"):
            hxy = self.create_xy_grid(opts["eval_res"], self.device)
            self.hxy_cache = hxy[None].expand(len(batch["dataid"]), -1, -1)
        batch["hxy"] = self.hxy_cache

    @staticmethod
    def create_xy_grid(eval_res, device):
        """Create a grid of pixel coordinates on the image plane

        Args:
            eval_res (int): Resolution to evaluate at
            device (torch.device): Target device
        Returns:
            hxy: (eval_res^2, 3) Homogeneous pixel coords on the image plane
        """
        eval_range = torch.arange(eval_res, dtype=torch.float32, device=device)
        hxy = torch.cartesian_prod(eval_range, eval_range)
        hxy = torch.stack([hxy[:, 1], hxy[:, 0], torch.ones_like(hxy[:, 0])], -1)
        return hxy

    def add_image_togrid(self, rendered_seq):
        """Add rendered outputs to Tensorboard visualization grid

        Args:
            rendered_seq (Dict): Dict of volume-rendered outputs. Keys:
                "mask" (M,H,W,1), "vis2d" (M,H,W,1), "depth" (M,H,W,1),
                "flow" (M,H,W,2), "feature" (M,H,W,16), "normal" (M,H,W,3), and
                "eikonal" (M,H,W,1)
        """
        for k, v in rendered_seq.items():
            img_grid = make_image_grid(v)
            self.add_image(self.log, k, img_grid, self.current_round)

    def add_image(self, log, tag, img, step):
        """Convert volume-rendered outputs to RGB and add to Tensorboard

        Args:
            log (SummaryWriter): Tensorboard logger
            tag (str): Image tag
            img: (H_out, W_out, x) Image to show
            step (int): Current step
        """
        if len(img.shape) == 2:
            formats = "HW"
        else:
            formats = "HWC"

        img = img2color(tag, img, pca_fn=self.data_info["apply_pca_fn"])

        log.add_image("img_" + tag, img, step, dataformats=formats)

    @staticmethod
    def add_scalar(log, dict, step):
        """Add a scalar value to Tensorboard log"""
        for k, v in dict.items():
            log.add_scalar(k, v, step)

    @staticmethod
    def construct_test_model(opts, model_class=dvr_model, return_refs=True):
        """Load a model at test time

        Args:
            opts (Dict): Command-line options
        """
        # io
        logname = "%s-%s" % (opts["seqname"], opts["logname"])

        # construct dataset
        eval_dict = Trainer.construct_dataset_opts(opts, is_eval=True)
        evalloader = data_utils.eval_loader(eval_dict)
        data_info, _ = data_utils.get_data_info(evalloader)

        # construct DVR model
        model = model_class(opts, data_info)
        load_path = "%s/%s/ckpt_%s.pth" % (
            opts["logroot"],
            logname,
            opts["load_suffix"],
        )
        _ = Trainer.load_checkpoint(load_path, model)
        model.cuda()
        model.eval()

        if "inst_id" in opts and return_refs:
            # get reference images
            inst_id = opts["inst_id"]
            offset = data_info["frame_info"]["frame_offset"]
            frame_id = np.asarray(
                range(offset[inst_id] - inst_id, offset[inst_id + 1] - inst_id - 1)
            )  # to account for pairs
            # only load a single frame
            if "freeze_id" in opts and opts["freeze_id"] > -1:
                frame_id = frame_id[opts["freeze_id"] : opts["freeze_id"] + 1]
            ref_dict, _ = Trainer.load_batch(evalloader.dataset, frame_id)
        else:
            ref_dict = None

        return model, data_info, ref_dict

    def check_grad(self, thresh=20.0):
        """Check if gradients are above a threshold

        Args:
            thresh (float): Gradient clipping threshold
        """
        # detect large gradients and reload model
        params_list = []
        for param_dict in self.params_ref_list:
            ((name, p),) = param_dict.items()
            if p.requires_grad and p.grad is not None:
                params_list.append(p)
                # if p.grad.isnan().any():
                #     p.grad.zero_()

        # check individual parameters
        grad_norm = torch.nn.utils.clip_grad_norm_(params_list, thresh)
        if grad_norm > thresh or torch.isnan(grad_norm):
            # clear gradients
            self.optimizer.zero_grad()
            if get_local_rank() == 0:
                print("large grad: %.2f, clear gradients" % grad_norm)
            # load cached model from two rounds ago
            if self.model_cache[0] is not None:
                if get_local_rank() == 0:
                    print("fallback to cached model")
                self.model.load_state_dict(self.model_cache[0])
                self.optimizer.load_state_dict(self.optimizer_cache[0])
                self.scheduler.load_state_dict(self.scheduler_cache[0])
            return {}

        # clip individual parameters
        grad_dict = {}
        queue_length = 10
        for param_dict in self.params_ref_list:
            ((name, p),) = param_dict.items()
            if p.requires_grad and p.grad is not None:
                grad = p.grad.reshape(-1).norm(2, -1)
                grad_dict["grad/" + name] = grad
                # maintain a queue of grad norm, and clip outlier grads
                matched_strict, clip_strict = match_param_name(
                    name, self.param_clip_startwith, type="startwith"
                )
                if matched_strict:
                    scale_threshold = clip_strict
                else:
                    continue

                # check the gradient norm
                if name not in self.grad_queue:
                    self.grad_queue[name] = []
                if len(self.grad_queue[name]) > queue_length:
                    med_grad = torch.stack(self.grad_queue[name][:-1]).median()
                    grad_dict["grad_med/" + name] = med_grad
                    if grad > scale_threshold * med_grad:
                        torch.nn.utils.clip_grad_norm_(p, med_grad)
                        # if get_local_rank() == 0:
                        #     print("large grad: %.2f, clear %s" % (grad, name))
                    elif grad > 0:
                        self.grad_queue[name].append(grad)
                        self.grad_queue[name].pop(0)
                    else:
                        pass
                elif grad > 0:
                    self.grad_queue[name].append(grad)
                else:
                    pass

        return grad_dict
