# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os, sys
import pdb
import torch
import numpy as np
import tqdm
from collections import defaultdict
import gc

from lab4d.engine.trainer import Trainer
from lab4d.engine.trainer import get_local_rank, DataParallelPassthrough

from projects.gsplat.gsplat import GSplatModel
from projects.gsplat import config

# from projects.predictor import config


class GSplatTrainer(Trainer):
    def __init__(self, opts):
        """Train and evaluate a Lab4D model.

        Args:
            opts (Dict): Command-line args from absl (defined in lab4d/config.py)
        """
        super().__init__(opts)

    def move_to_ddp(self):
        # move model to ddp
        self.model = DataParallelPassthrough(
            self.model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=True,
        )

    def define_model(self):
        self.device = torch.device("cuda:{}".format(get_local_rank()))
        self.model = GSplatModel(self.opts, self.data_info)

        # ddp
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = self.model.to(self.device)

    def load_checkpoint_train(self):
        if self.opts["load_path"] != "":
            # training time
            checkpoint = self.load_checkpoint(
                self.opts["load_path"], self.model, optimizer=self.optimizer
            )
            if not self.opts["reset_steps"]:
                self.current_steps = checkpoint["current_steps"]
                self.current_round = checkpoint["current_round"]
                self.first_round = self.current_round
                self.first_step = self.current_steps

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
        div_factor = 25.0
        final_div_factor = 1.0
        pct_start = min(1 - 1e-5, 2.0 / opts["num_rounds"])  # use 2 epochs to warm up
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

    def get_lr_dict(self, pose_correction=False):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        opts = self.opts
        lr_base = opts["learning_rate"]

        param_lr_startwith = {
            "module.gaussians._xyz": lr_base,
            "module.gaussians._features_dc": lr_base,
            "module.gaussians._features_rest": lr_base * 0.05,
            "module.gaussians._scaling": lr_base * 0.5,
            "module.gaussians._rotation": lr_base * 0.5,
            "module.gaussians._opacity": lr_base * 5,
            "module.guidance_sd": 0.0,
        }
        param_lr_with = {}

        return param_lr_startwith, param_lr_with

    def train(self):
        super().train()

    def save_checkpoint(self, round_count):
        """Save model checkpoint to disk

        Args:
            round_count (int): Current round index
        """
        opts = self.opts

        if get_local_rank() == 0 and round_count % opts["save_freq"] == 0:
            print("saving round %d" % round_count)
            param_path = "%s/ckpt_%04d.pth" % (self.save_dir, round_count)

            checkpoint = {
                "current_steps": self.current_steps,
                "current_round": self.current_round,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            torch.save(checkpoint, param_path)
            # copy to latest
            latest_path = "%s/ckpt_latest.pth" % (self.save_dir)
            os.system("cp %s %s" % (param_path, latest_path))

    def check_grad(self):
        return {}

    def construct_eval_batch(self, batch):
        pass
