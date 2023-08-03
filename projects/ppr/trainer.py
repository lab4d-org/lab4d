# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os, sys
import pdb
import torch
import numpy as np

from lab4d.engine.trainer import Trainer
from lab4d.engine.trainer import get_local_rank
from ppr import config

sys.path.insert(0, "%s/ppr-diffphys" % os.path.join(os.path.dirname(__file__)))
from diffphys.dp_interface import phys_interface
from diffphys.vis import Logger


class PPRTrainer(Trainer):
    def define_model(self):
        super().define_model()

        # define physics model
        opts = self.opts
        opts["phys_vid"] = [int(i) for i in opts["phys_vid"].split(",")]
        model_dict = {}
        model_dict["bg_field"] = self.model.fields.field_params["bg"]
        model_dict["obj_field"] = self.model.fields.field_params["fg"]
        model_dict["intrinsics"] = self.model.intrinsics
        self.phys_model = phys_interface(opts, model_dict)
        self.phys_visualizer = Logger(opts)

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
        print("# iterations per phys cycle: ", self.iters_per_phys_cycle)

    def run_one_round(self, round_count):
        super().run_one_round(round_count)

        # transfer pharameters
        self.run_phys_cycle()
        # transfer pharameters

    def run_phys_cycle(self):
        opts = self.opts
        torch.cuda.empty_cache()

        # re-initialize field2world transforms
        self.model.fields.field_params["bg"].compute_field2world()

        # eval
        self.phys_model.eval()
        self.phys_model.reinit_envs(1, wdw_length=30, is_eval=True)
        for vidid in opts["phys_vid"]:
            frame_start = torch.zeros(1) + self.phys_model.data_offset[vidid]
            _ = self.phys_model(frame_start=frame_start.to(self.device))
            img_size = tuple(self.data_info["raw_size"][vidid][::-1])
            img_size = img_size + (0.5,)  # scale
            data = self.phys_model.query(img_size=img_size)
            self.phys_visualizer.show(
                "%02d-%05d" % (vidid, self.current_steps_phys), data
            )

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
