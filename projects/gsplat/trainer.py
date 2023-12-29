# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os, sys
import pdb
import torch
import torch.nn as nn
import numpy as np
import tqdm
from collections import defaultdict
import gc
from bisect import bisect_right

from lab4d.engine.trainer import Trainer
from lab4d.engine.trainer import get_local_rank, DataParallelPassthrough
from lab4d.utils.torch_utils import get_nested_attr, set_nested_attr

from projects.gsplat.gsplat import GSplatModel
from projects.gsplat import config


class CustomSequentialLR(torch.optim.lr_scheduler.SequentialLR):
    def step(self, epoch=None):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        scheduler.step(epoch)
        self._last_lr = scheduler.get_last_lr()


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

    def optimizer_init(self, is_resumed=False, use_warmup_param=False):
        """Set the learning rate for all trainable parameters and initialize
        the optimizer and scheduler.

        Args:
            is_resumed (bool): True if resuming from checkpoint
        """
        opts = self.opts
        param_lr_startwith, param_lr_with = self.get_lr_dict(
            use_warmup_param=use_warmup_param
        )
        self.params_ref_list, params_list, lr_list = self.get_optimizable_param_list(
            param_lr_startwith, param_lr_with
        )
        self.optimizer = torch.optim.AdamW(
            params_list,
            lr=opts["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        if opts["guidance_zero123_wt"] > 0 or opts["guidance_sd_wt"] > 0:
            # new scheduler
            exp_rate = 0.9922
            warmup_iters = 500  # 1=>1/50
            scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, exp_rate
            )
            scheduler_lin = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                exp_rate**warmup_iters,
                1e-3,
                total_iters=self.total_steps,
            )
            milestones = [warmup_iters]
            schedulers = [scheduler_exp, scheduler_lin]
            if opts["inc_warmup_ratio"] > 0:
                const_steps = int(opts["inc_warmup_ratio"] * self.total_steps)
                scheduler_const = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, 1
                )
                schedulers.insert(0, scheduler_const)
                milestones.insert(0, const_steps)
            self.scheduler = CustomSequentialLR(
                self.optimizer, schedulers=schedulers, milestones=milestones
            )
        else:
            # previous scheduler
            if opts["inc_warmup_ratio"] > 0:
                div_factor = 5.0
                final_div_factor = 25.0
                pct_start = opts["inc_warmup_ratio"]
            else:
                div_factor = 25.0
                final_div_factor = 1.0
                pct_start = min(1 - 1e-5, 0.02)  # use 2% to warm up
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

    def get_lr_dict(self, use_warmup_param=False):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        opts = self.opts
        lr_base = opts["learning_rate"]

        if use_warmup_param:
            param_lr_startwith = {
                "module.gaussians._features_dc": lr_base,
                "module.gaussians._features_rest": lr_base * 0.05,
                "module.gaussians._trajectory": lr_base,
                "module.gaussians.camera_mlp": lr_base,
            }
        else:
            param_lr_startwith = {
                "module.gaussians._xyz": lr_base,
                "module.gaussians._features_dc": lr_base,
                "module.gaussians._features_rest": lr_base * 0.05,
                "module.gaussians._scaling": lr_base * 0.5,
                "module.gaussians._rotation": lr_base * 0.5,
                "module.gaussians._opacity": lr_base,
                "module.gaussians._trajectory": lr_base * 0.5,
                "module.gaussians.camera_mlp": lr_base,
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

    @staticmethod
    def construct_test_model(opts, model_class=GSplatModel):
        return Trainer.construct_test_model(opts, model_class=model_class)

    def train_one_round(self):
        """Train a single round (going over mini-batches)"""
        opts = self.opts
        gc.collect()  # need to be used together with empty_cache()
        torch.cuda.empty_cache()
        self.model.train()
        self.optimizer.zero_grad()

        # necessary for shuffling
        self.trainloader.sampler.set_epoch(self.current_round)
        # set max loader length for incremental opt
        if opts["inc_warmup_ratio"] > 0:
            self.set_warmup_hparams()
        for i, batch in tqdm.tqdm(enumerate(self.trainloader)):
            if i == opts["iters_per_round"]:
                break

            progress = (self.current_steps - self.first_step) / self.total_steps
            sub_progress = i / opts["iters_per_round"]
            self.model.set_progress(self.current_steps, progress, sub_progress)

            self.model.convert_img_to_pixel(batch)

            loss_dict = self.model(batch)
            total_loss = torch.sum(torch.stack(list(loss_dict.values())))
            total_loss.mean().backward()
            # print(total_loss)
            # self.print_sum_params()

            grad_dict = self.check_grad()
            self.optimizer.step()
            self.scheduler.step(self.current_steps)
            self.optimizer.zero_grad()

            # keep track of xyz spatial gradients for densification
            self.model.update_densification_stats()

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

    def set_warmup_hparams(self):
        """Set the loader range for incremental optimization"""
        inc_warmup_ratio = self.opts["inc_warmup_ratio"]
        warmup_rounds = inc_warmup_ratio * self.opts["num_rounds"]

        # config optimizer
        if self.current_round == 0:
            self.optimizer_init(use_warmup_param=True)
        elif self.current_round == warmup_rounds:
            self.optimizer_init(use_warmup_param=False)
        self.scheduler.last_epoch = self.current_steps  # specific to onecyclelr

        # config dataloader
        completion_ratio = self.current_round / (warmup_rounds - 1)
        for dataset in self.trainloader.dataset.datasets:
            # per pair opt
            if self.current_round < warmup_rounds:
                min_frameid = int((len(dataset) - 1) * completion_ratio)
                max_frameid = min_frameid + 1
            else:
                min_frameid = 0
                max_frameid = len(dataset)

            # # global opt
            # min_frameid = 0
            # max_frameid = int((len(dataset) - 1) * completion_ratio) + 1

            dataset.set_loader_range(min_frameid=min_frameid, max_frameid=max_frameid)

        # set parameters for incremental opt
        if self.current_round < warmup_rounds:
            self.model.gaussians.set_future_time_params(max_frameid)

    def update_aux_vars(self):
        self.model.update_geometry_aux()
        self.model.export_geometry_aux(
            "%s/%03d-all" % (self.save_dir, self.current_round)
        )

        if self.model.progress > self.opts["inc_warmup_ratio"]:
            self.densify_and_prune()

    def densify_and_prune(self):
        # densify and prune
        if self.opts["guidance_zero123_wt"] > 0:
            max_grad = 1e-3
            min_grad = 0.0
        else:
            max_grad = 1e-4
            min_grad = 1e-6
        clone_mask, prune_mask = self.model.gaussians.densify_and_prune(
            max_grad=max_grad, min_grad=min_grad
        )

        # update stats
        self.model.gaussians.update_point_stats(prune_mask, clone_mask)

        if prune_mask.sum() or clone_mask.sum():
            self.prune_parameters(~prune_mask, clone_mask)

            self.model.update_geometry_aux()
            self.model.export_geometry_aux(
                "%s/%03d-post" % (self.save_dir, self.current_round)
            )
            print("cloned %d/%d" % (clone_mask.sum(), clone_mask.shape[0]))
            print("pruned %d/%d" % (prune_mask.sum(), prune_mask.shape[0]))
            # torch.cuda.empty_cache()
            # self.reset_opacity()

        # update optimizer
        self.optimizer_init()
        self.scheduler.last_epoch = self.current_steps  # specific to onecyclelr
        self.scheduler.step(self.current_steps)

    def prune_parameters(self, valid_mask, clone_mask):
        """
        Remove the optimizer state of the pruned parameters.
        Set the parameters to the remaining ones.
        """
        # first clone, then prune
        dev = self.device
        clone_mask = torch.logical_and(valid_mask, clone_mask)
        valid_mask = torch.cat(
            (valid_mask, torch.ones(clone_mask.sum(), device=dev).bool())
        )

        for param_dict in self.params_ref_list:
            ((name, _),) = param_dict.items()
            if not name.startswith("module.gaussians._"):
                continue
            param = get_nested_attr(self.model, name)
            stored_state = self.optimizer.state.get(param, None)
            if stored_state is not None:
                exp_avg = stored_state["exp_avg"]
                exp_avg_sq = stored_state["exp_avg_sq"]

                exp_avg = torch.cat((exp_avg, exp_avg[clone_mask]))[valid_mask]
                exp_avg_sq = torch.cat((exp_avg_sq, exp_avg_sq[clone_mask]))[valid_mask]

                stored_state["exp_avg"] = exp_avg
                stored_state["exp_avg_sq"] = exp_avg_sq

                del self.optimizer.state[param]
                self.optimizer.state[param] = stored_state

            # set param
            param = torch.cat((param, param[clone_mask]))[valid_mask]
            param = nn.Parameter(param.requires_grad_(True))
            set_nested_attr(self.model, name, param)
