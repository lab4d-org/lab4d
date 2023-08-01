# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os, sys
import pdb
import torch
import numpy as np

from lab4d.engine.trainer import Trainer
from lab4d.engine.trainer import get_local_rank
from ppr import config

sys.path.insert(0, "%s/ppr-diffphys" % os.path.join(os.path.dirname(__file__)))
from diffphys.warp_env import phys_model


class PPRTrainer(Trainer):
    def define_model(self):
        super().define_model()

        # define physics model
        opts = self.opts
        opts["phys_vid"] = [int(i) for i in opts["phys_vid"].split(",")]
        model_dict = {}
        model_dict["bg_rts"] = self.model.fields.field_params["bg"].camera_mlp
        model_dict["nerf_root_rts"] = self.model.fields.field_params["fg"].camera_mlp
        model_dict["nerf_body_rts"] = self.model.fields.field_params[
            "fg"
        ].warp.articulation
        model_dict["ks_params"] = self.model.intrinsics
        self.phys_model = phys_model(opts, model_dict, use_dr=True)

        # move model to device
        self.device = torch.device("cuda:{}".format(get_local_rank()))
        self.phys_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.phys_model)
        self.phys_model = self.phys_model.to(self.device)

    def init_model(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics from external priors, if this is the first run"""
        super().init_model()

    def trainer_init(self):
        super().trainer_init()

        opts = self.opts
        self.current_steps_phys = 0  # 0-total_steps
        self.iters_per_phys_cycle = int(
            opts["ratio_phys_cycle"] * opts["iters_per_round"]
        )

    def get_lr_dict(self):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        return super().get_lr_dict()
        # opts = self.opts
        # lr_base = opts["learning_rate"]
        # lr_explicit = lr_base * 10

        # # only update the following parameters
        # param_lr_startwith = {
        #     "module.intrinsics": lr_base,
        #     "module.fields.field_params.fg.camera_mlp.base_quat": lr_explicit,
        # }
        # param_lr_with = {
        #     "inst_embedding": lr_explicit,
        #     "time_embedding.mapping1": lr_base,
        #     ".base_logfocal": lr_explicit,
        #     ".base_ppoint": lr_explicit,
        # }
        # return param_lr_startwith, param_lr_with

    def run_one_round(self, round_count):
        super().run_one_round(round_count)

        # transfer pharameters
        self.run_phys_cycle()
        # transfer pharameters

    def run_phys_cycle(self):
        opts = self.opts
        torch.cuda.empty_cache()

        # eval
        self.phys_model.eval()
        self.phys_model.reinit_envs(1, wdw_length=30, is_eval=True)

        # train
        self.phys_model.train()
        self.phys_model.reinit_envs(
            opts["phys_batch"], wdw_length=opts["phys_wdw_len"], is_eval=False
        )

        for i in range(self.iters_per_phys_cycle):
            self.phys_model.set_progress(self.current_steps_phys)
            self.run_phys_iter()
            self.current_steps_phys += 1
            print(self.current_steps_phys)

    def run_phys_iter(self):
        """Run physics optimization"""
        phys_aux = self.phys_model()
        self.phys_model.backward(phys_aux["total_loss"])
        grad_list = self.phys_model.update()
        phys_aux.update(grad_list)
