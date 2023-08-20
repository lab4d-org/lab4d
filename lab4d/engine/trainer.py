# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os
import time
from collections import defaultdict
from copy import deepcopy

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
from lab4d.utils.torch_utils import remove_ddp_prefix
from lab4d.utils.vis_utils import img2color, make_image_grid


class Trainer:
    def __init__(self, opts):
        """Train and evaluate a Lab4D model.

        Args:
            opts (Dict): Command-line args from absl (defined in lab4d/config.py)
        """
        # When profiling, use fewer iterations per round so trace files are smaller
        is_resumed = opts["load_path"] != ""
        if opts["profile"]:
            opts["iters_per_round"] = 10

        self.opts = opts

        self.define_dataset()
        self.trainer_init()
        self.define_model()

        # move model to ddp
        self.model = DataParallelPassthrough(
            self.model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=False,
        )

        self.optimizer_init(is_resumed=is_resumed)

        # load model
        if is_resumed:
            self.load_checkpoint_train()

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

        # 0-last image in eval dataset
        self.eval_fid = np.linspace(0, len(self.evalloader) - 1, 9).astype(int)

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

    def get_lr_dict(self):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        opts = self.opts
        lr_base = opts["learning_rate"]
        lr_explicit = lr_base * 10

        param_lr_startwith = {
            "module.fields.field_params": lr_base,
            "module.intrinsics": lr_base,
        }
        param_lr_with = {
            ".logibeta": lr_explicit,
            ".logsigma": lr_explicit,
            ".logscale": lr_explicit,
            ".log_gauss": 0.0,
            ".base_quat": lr_explicit,
            ".base_logfocal": lr_explicit,
            ".base_ppoint": lr_explicit,
            ".shift": lr_explicit,
            ".orient": lr_explicit,
        }
        return param_lr_startwith, param_lr_with

    def optimizer_init(self, is_resumed=False):
        """Set the learning rate for all trainable parameters and initialize
        the optimizer and scheduler.

        Args:
            is_resumed (bool): True if resuming from checkpoint
        """
        opts = self.opts
        self.params_ref_list, params_list, lr_list = self.get_optimizable_param_list()
        self.optimizer = torch.optim.AdamW(
            params_list,
            lr=opts["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        # initial_lr = lr/div_factor
        # min_lr = initial_lr/final_div_factor
        if is_resumed:
            div_factor = 1.0
            final_div_factor = 5.0
            pct_start = 0.0  # cannot be 0
        else:
            div_factor = 25.0
            final_div_factor = 1.0
            pct_start = 2.0 / opts["num_rounds"]  # use 2 epochs to warm up
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            lr_list,
            int(self.total_steps),
            pct_start=pct_start,
            cycle_momentum=False,
            anneal_strategy="linear",
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    def get_optimizable_param_list(self):
        """
        Get the optimizable param list
        Returns:
            params_ref_list (List): List of params
            params_list (List): List of params
            lr_list (List): List of learning rates
        """
        param_lr_startwith, param_lr_with = self.get_lr_dict()
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
                params_list.append({"params": p})
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
        for round_count in range(
            self.current_round, self.current_round + opts["num_rounds"]
        ):
            start_time = time.time()
            with torch_profile(
                self.save_dir, f"{round_count:03d}", enabled=opts["profile"]
            ):
                self.run_one_round(round_count)

            if get_local_rank() == 0:
                print(f"Round {round_count:03d}: time={time.time() - start_time:.3f}s")

    def run_one_round(self, round_count):
        """Evaluation and training for a single round

        Args:
            round_count (int): Current round index
        """
        if get_local_rank() == 0:
            if round_count == 0:
                self.model_eval()

        self.model.update_geometry_aux()
        self.model.export_geometry_aux("%s/%03d" % (self.save_dir, round_count))

        self.model.train()
        self.train_one_round(round_count)
        self.current_round += 1
        self.save_checkpoint(round_count=self.current_round)

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
    def load_checkpoint(load_path, model, optimizer=None):
        """Load a model from checkpoint

        Args:
            load_path (str): Path to checkpoint
            model (dvr_model): Model to update in place
            optimizer (torch.optim.Optimizer or None): If provided, load
                learning rate from checkpoint
        """
        checkpoint = torch.load(load_path)
        model_states = checkpoint["model"]
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_states = remove_ddp_prefix(model_states)
        model.load_state_dict(model_states, strict=False)

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
        # training time
        checkpoint = self.load_checkpoint(
            self.opts["load_path"], self.model, optimizer=self.optimizer
        )
        if not self.opts["reset_steps"]:
            self.current_steps = checkpoint["current_steps"]
            self.current_round = checkpoint["current_round"]

        # reset near_far
        self.model.fields.reset_geometry_aux()

    def train_one_round(self, round_count):
        """Train a single round (going over mini-batches)

        Args:
            round_count (int): round index
        """
        opts = self.opts
        torch.cuda.empty_cache()
        self.model.train()

        self.trainloader.sampler.set_epoch(round_count)  # necessary for shuffling
        for i, batch in enumerate(self.trainloader):
            if i == opts["iters_per_round"]:
                break

            self.model.set_progress(self.current_steps)

            loss_dict = self.model(batch)
            total_loss = torch.sum(torch.stack(list(loss_dict.values())))
            total_loss.mean().backward()
            # print(total_loss)
            # self.print_sum_params()

            self.check_grad()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if get_local_rank() == 0:
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
        opts_dict["dataset_constructor"] = dataset_constructor

        if is_eval:
            opts_dict["multiply"] = False
            opts_dict["pixels_per_image"] = -1
            opts_dict["delta_list"] = []
        else:
            # duplicate dataset to fix number of iterations per round
            opts_dict["multiply"] = True
            opts_dict["pixels_per_image"] = opts["pixels_per_image"]
            opts_dict["delta_list"] = [2, 4, 8]
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
        torch.cuda.empty_cache()
        ref_dict, batch = self.load_batch(self.evalloader.dataset, self.eval_fid)
        self.construct_eval_batch(batch)
        rendered = self.model.evaluate(batch)
        self.add_image_togrid(ref_dict)
        self.add_image_togrid(rendered)
        self.visualize_matches(rendered["xyz"], rendered["xyz_matches"], tag="xyz")
        self.visualize_matches(
            rendered["xyz_cam"], rendered["xyz_reproj"], tag="xyz_cam"
        )

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
        xyz = xyz[0].view(-1, 3).detach().cpu().numpy()
        xyz_matches = xyz_matches[0].view(-1, 3).detach().cpu().numpy()
        xyz = trimesh.Trimesh(vertices=xyz)
        xyz_matches = trimesh.Trimesh(vertices=xyz_matches)

        xyz.visual.vertex_colors = [255, 0, 0, 255]
        xyz_matches.visual.vertex_colors = [0, 255, 0, 255]
        xyz_cat = trimesh.util.concatenate([xyz, xyz_matches])

        xyz_cat.export("%s/%03d-%s.obj" % (self.save_dir, self.current_round, tag))

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
        ref_keys = ["rgb", "mask", "depth", "feature", "vis2d"]
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
    def construct_test_model(opts):
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
        model = dvr_model(opts, data_info)
        load_path = "%s/%s/ckpt_%s.pth" % (
            opts["logroot"],
            logname,
            opts["load_suffix"],
        )
        _ = Trainer.load_checkpoint(load_path, model)
        model.cuda()
        model.eval()

        # get reference images
        inst_id = opts["inst_id"]
        offset = data_info["frame_info"]["frame_offset"]
        frame_id = np.asarray(
            range(offset[inst_id] - inst_id, offset[inst_id + 1] - inst_id - 1)
        )  # to account for pairs
        ref_dict, _ = Trainer.load_batch(evalloader.dataset, frame_id)

        return model, data_info, ref_dict

    def check_grad(self, thresh=5.0):
        """Check if gradients are above a threshold

        Args:
            thresh (float): Gradient clipping threshold
        """
        # parameters that are sensitive to large gradients
        params_list = []
        for param_dict in self.params_ref_list:
            ((name, p),) = param_dict.items()
            if p.requires_grad:
                params_list.append(p)

        grad_norm = torch.nn.utils.clip_grad_norm_(params_list, thresh)
        if grad_norm > thresh:
            # clear gradients
            self.optimizer.zero_grad()
            print("large grad: %.2f, clear gradients" % grad_norm)
            # load cached model from two rounds ago
            if self.model_cache[0] is not None:
                if get_local_rank() == 0:
                    print("fallback to cached model")
                self.model.load_state_dict(self.model_cache[0])
                self.optimizer.load_state_dict(self.optimizer_cache[0])
                self.scheduler.load_state_dict(self.scheduler_cache[0])
