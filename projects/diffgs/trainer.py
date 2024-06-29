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
import open3d.core as o3c

from lab4d.engine.trainer import Trainer
from lab4d.engine.trainer import get_local_rank, DataParallelPassthrough
from lab4d.dataloader import data_utils
from lab4d.utils.torch_utils import get_nested_attr, set_nested_attr

from projects.diffgs.gs_model import GSplatModel
from projects.diffgs import config


class AdaHessian(torch.optim.Optimizer):
    """
    Implements the AdaHessian algorithm from "ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning"

    Arguments:
        params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) -- learning rate (default: 0.1)
        betas ((float, float), optional) -- coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))
        eps (float, optional) -- term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
        hessian_power (float, optional) -- exponent of the hessian trace (default: 1.0)
        update_each (int, optional) -- compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)
        n_samples (int, optional) -- how many times to sample `z` for the approximation of the hessian trace (default: 1)
    """

    def __init__(
        self,
        params,
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        hessian_power=1.0,
        update_each=1,
        n_samples=1,
        average_conv_kernel=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")

        self.n_samples = n_samples
        self.update_each = update_each
        self.average_conv_kernel = average_conv_kernel

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
        )
        super(AdaHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        )

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if (
                not isinstance(p.hess, float)
                and self.state[p]["hessian step"] % self.update_each == 0
            ):
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            print(p.shape)
            if (
                self.state[p]["hessian step"] % self.update_each == 0
            ):  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if (
            self.generator.device != params[0].device
        ):  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        grads = [p.grad for p in params]

        for i in range(self.n_samples):
            zs = [
                torch.randint(0, 2, p.size(), generator=self.generator, device=p.device)
                * 2.0
                - 1.0
                for p in params
            ]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(
                grads,
                params,
                grad_outputs=zs,
                only_inputs=True,
                retain_graph=i < self.n_samples - 1,
            )
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += (
                    h_z * z / self.n_samples
                )  # approximate the expected values of z*(H@z)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = (
                        torch.abs(p.hess)
                        .mean(dim=[2, 3], keepdim=True)
                        .expand_as(p.hess)
                        .clone()
                    )

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group["lr"] * group["weight_decay"])

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p.data
                    )  # Exponential moving average of gradient values
                    state["exp_hessian_diag_sq"] = torch.zeros_like(
                        p.data
                    )  # Exponential moving average of Hessian diagonal square values

                exp_avg, exp_hessian_diag_sq = (
                    state["exp_avg"],
                    state["exp_hessian_diag_sq"],
                )
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(
                    p.hess, p.hess, value=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                k = group["hessian_power"]
                denom = (
                    (exp_hessian_diag_sq / bias_correction2)
                    .pow_(k / 2)
                    .add_(group["eps"])
                )

                # make update
                step_size = group["lr"] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


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
        # if in incremental mode and num-rounds = 0, reset number of rounds
        if opts["inc_warmup_ratio"] > 0 and opts["num_rounds"] == 0:
            eval_dict = self.construct_dataset_opts(opts, is_eval=True)
            evalloader = data_utils.eval_loader(eval_dict)
            warmup_rounds = int(opts["first_fr_steps"] / opts["iters_per_round"])
            inc_rounds = len(evalloader) + warmup_rounds
            opts["num_rounds"] = int(inc_rounds / opts["inc_warmup_ratio"])
            print("# warmup rounds = %d" % warmup_rounds)
            print("# incremental rounds = %d" % inc_rounds)
            print("# total rounds = %d" % opts["num_rounds"])
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

        self.init_model()

        # cache queue of length 2
        self.model_cache = [None, None]
        self.optimizer_cache = [None, None]
        self.scheduler_cache = [None, None]

        self.grad_queue = {}
        self.param_clip_startwith = {
            # "module.gaussians._xyz": 5,
            # "module.gaussians._features_dc": 5,
            # "module.gaussians._features_rest": 5,
            # "module.gaussians._scaling": 5,
            # "module.gaussians._rotation": 5,
            # "module.gaussians._opacity": 5,
            # "module.gaussians._trajectory": 5,
            # "module.gaussians.gs_camera_mlp": 5,
        }

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

        # NOTE: Using DistributedDataParallel with create_graph=True  is not well-supported.
        # The higher-order gradient will  not be synchronized across ranks, and backpropagation  through all_reduce operations will not occur.
        # If you require  DDP to work with higher-order gradients for your use case,  please ping https://github.com/pytorch/pytorch/issues/63929
        # self.optimizer = AdaHessian(
        #     params_list,
        #     lr=opts["learning_rate"],
        #     betas=(0.9, 0.999),
        #     weight_decay=0.0,
        # )

        if opts["inc_warmup_ratio"] > 0:
            div_factor = 1.0
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

        # # cycle lr
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     self.optimizer,
        #     [i * 0.01 for i in lr_list],
        #     lr_list,
        #     step_size_up=10,
        #     step_size_down=1990,
        #     mode="triangular",
        #     gamma=1.0,
        #     scale_mode="cycle",
        #     cycle_momentum=False,
        # )

        # # less aggressive scheduler for diffusion guided optimization
        # # as a way to set group lrs
        # torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     lr_list,
        #     int(self.total_steps),
        #     pct_start=0.0,
        #     div_factor=1,
        #     final_div_factor=1,
        # )
        # exp_rate = 0.9995
        # warmup_iters = 3000  # 1=>1/5
        # scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, exp_rate)
        # scheduler_lin = torch.optim.lr_scheduler.LinearLR(
        #     self.optimizer,
        #     exp_rate**warmup_iters,
        #     0.2,
        #     total_iters=self.total_steps,
        # )
        # milestones = [warmup_iters]
        # schedulers = [scheduler_exp, scheduler_lin]
        # self.scheduler = CustomSequentialLR(
        #     self.optimizer, schedulers=schedulers, milestones=milestones
        # )

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
                "module.bg_color": lr_base * 5,
            }
            param_lr_with = {
                "._features_dc": lr_base,
                "._features_rest": lr_base * 0.05,
                "._trajectory": lr_base,
                ".gs_camera_mlp": lr_base * 2,
            }
        else:
            if opts["extrinsics_type"] == "image":
                camera_lr = lr_base * 0.1
                xyz_lr = lr_base * 0.2
            else:
                camera_lr = lr_base * 2
                xyz_lr = lr_base
            param_lr_startwith = {
                "module.bg_color": lr_base * 5,
                "module.guidance_sd": 0.0,
            }
            param_lr_with = {
                "._xyz": xyz_lr,
                "._features_dc": lr_base,
                "._features_rest": lr_base * 0.05,
                "._scaling": lr_base * 0.5,
                "._rotation": lr_base * 0.5,
                "._opacity": lr_base * 5,
                "._trajectory": lr_base * 0.5,
                ".gs_camera_mlp": camera_lr * 0.1,
                ".lab4d_model": lr_base * 0.1,
                ".shadow_field": lr_base * 0.1,
            }

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
    def construct_test_model(opts, model_class=GSplatModel, return_refs=True):
        return Trainer.construct_test_model(opts, model_class=model_class, return_refs=return_refs)

    def train_one_round(self):
        """Train a single round (going over mini-batches)"""
        opts = self.opts
        gc.collect()  # need to be used together with empty_cache()
        torch.cuda.empty_cache()
        # o3c.cuda.release_cache()
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

            loss_dict = self.model(batch)
            total_loss = torch.sum(torch.stack(list(loss_dict.values())))
            total_loss.mean().backward()

            grad_dict = self.check_grad()
            self.optimizer.step()
            self.scheduler.step(self.current_steps)
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

    def set_warmup_hparams(self):
        """Set the loader range for incremental optimization"""
        inc_warmup_ratio = self.opts["inc_warmup_ratio"]
        warmup_rounds = inc_warmup_ratio * self.opts["num_rounds"]

        # config optimizer
        if self.current_round < warmup_rounds:
            self.optimizer_init(use_warmup_param=True)
        elif self.current_round == warmup_rounds:
            self.optimizer_init(use_warmup_param=False)
        self.scheduler.last_epoch = self.current_steps  # specific to onecyclelr

        # config dataloader
        first_fr_steps = self.opts["first_fr_steps"]  # 1st fr warmup steps
        first_fr_ratio = first_fr_steps / (inc_warmup_ratio * self.total_steps)
        completion_ratio = self.current_round / (warmup_rounds - 1)
        completion_ratio = (completion_ratio - first_fr_ratio) / (1 - first_fr_ratio)
        for dataset in self.trainloader.dataset.datasets:
            # per pair opt
            if self.current_round < int(warmup_rounds * first_fr_ratio):
                min_frameid = 0
                max_frameid = 1
            elif self.current_round < warmup_rounds:
                min_frameid = int((len(dataset) - 1) * completion_ratio)
                max_frameid = min_frameid + 1
            else:
                min_frameid = 0
                max_frameid = len(dataset)

            # # global opt
            # min_frameid = 0
            # max_frameid = int((len(dataset) - 1) * completion_ratio) + 1

            dataset.set_loader_range(min_frameid=min_frameid, max_frameid=max_frameid)
            print("setting loader range to %d-%d" % (min_frameid, max_frameid))

        # set parameters for incremental opt
        if (
            self.current_round >= int(warmup_rounds * first_fr_ratio)
            and self.current_round < warmup_rounds
        ):
            self.model.gaussians.set_future_time_params(min_frameid)

    def update_aux_vars(self):
        self.model.update_geometry_aux()
        self.model.export_geometry_aux(
            "%s/%03d-all" % (self.save_dir, self.current_round)
        )

        # add some noise to improve convergence
        if self.current_round !=0 and self.current_round % 10 == 0:
            self.model.gaussians.reset_gaussian_scale()
        if self.current_round !=0 and self.current_round % 10 == 0:
            self.model.gaussians.randomize_gaussian_center()

    def densify_and_prune(self):
        # densify and prune
        clone_mask, prune_mask = self.model.gaussians.densify_and_prune()

        # update stats
        self.model.gaussians.update_point_stats(prune_mask, clone_mask)

        if prune_mask.sum() or clone_mask.sum():
            self.prune_parameters(~prune_mask, clone_mask)
            vlist = []
            for i in range(len(self.optimizer.param_groups)):
                vlist.append(self.optimizer.state[self.optimizer.param_groups[i]['params'][0]])

            # self.model.update_geometry_aux()
            # self.model.export_geometry_aux(
            #     "%s/%03d-post" % (self.save_dir, self.current_round)
            # )
            print("cloned %d/%d" % (clone_mask.sum(), clone_mask.shape[0]))
            print("pruned %d/%d" % (prune_mask.sum(), prune_mask.shape[0]))
            # torch.cuda.empty_cache()
            # self.reset_opacity()

            # update optimizer
            self.optimizer_init()
            # restore optimizer stats
            for i,v in enumerate(vlist):
                self.optimizer.state[self.optimizer.param_groups[i]['params'][0]] = v
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
            if not "._" in name: # pts related params
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
