# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os, sys
import pdb
import torch
import numpy as np
import tqdm

from lab4d.engine.trainer import Trainer
from lab4d.engine.trainer import get_local_rank
from ppr import config

sys.path.insert(0, "%s/ppr-diffphys" % os.path.join(os.path.dirname(__file__)))
from diffphys.dp_interface import phys_interface
from diffphys.vis import Logger


class PPRTrainer(Trainer):
    def define_model(self):
        super().define_model()

        # opts
        opts = self.opts
        opts["phys_vid"] = [int(i) for i in opts["phys_vid"].split(",")]
        opts["urdf_template"] = opts["fg_motion"].split("-")[1].split("_")[0]

        # model
        model_dict = {}
        model_dict["scene_field"] = self.model.fields.field_params["bg"]
        model_dict["object_field"] = self.model.fields.field_params["fg"]
        model_dict["intrinsics"] = self.model.intrinsics

        # define phys model
        self.phys_model = phys_interface(opts, model_dict)
        self.phys_visualizer = Logger(opts)

        # move model to device
        self.device = torch.device("cuda:{}".format(get_local_rank()))
        self.phys_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.phys_model)
        self.phys_model = self.phys_model.to(self.device)

    def get_lr_dict(self):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        param_lr_startwith, param_lr_with = super().get_lr_dict()
        opts = self.opts

        param_lr_with.update(
            {
                "module.fields.field_params.fg.basefield.": 0.0,
                "module.fields.field_params.fg.colorfield.": 0.0,
                "module.fields.field_params.fg.sdf.": 0.0,
                "module.fields.field_params.fg.rgb.": 0.0,
                "module.fields.field_params.bg.basefield.": 0.0,
                "module.fields.field_params.bg.colorfield.": 0.0,
                "module.fields.field_params.bg.sdf.": 0.0,
                "module.fields.field_params.bg.rgb.": 0.0,
            }
        )
        return param_lr_startwith, param_lr_with

    def init_model(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics from external priors, if this is the first run"""
        # super().init_model()
        return

    def trainer_init(self):
        super().trainer_init()

        opts = self.opts
        self.current_steps_phys = 0  # 0-total_steps
        self.iters_per_phys_cycle = int(
            opts["ratio_phys_cycle"] * opts["iters_per_round"]
        )
        print("# iterations per phys cycle:", self.iters_per_phys_cycle)

    def run_one_round(self, round_count):
        # re-initialize field2world transforms
        self.model.fields.field_params["bg"].compute_field2world()
        # transfer pharameters
        self.phys_model.override_states()
        # run physics cycle
        self.run_phys_cycle()
        # transfer pharameters
        self.phys_model.override_states_inv()
        # run dr cycle
        super().run_one_round(round_count)

    def run_phys_cycle(self):
        opts = self.opts
        torch.cuda.empty_cache()

        # eval
        self.phys_model.eval()
        self.phys_model.correct_foot_position()
        self.run_phys_visualization(tag="kinematics")

        # train
        self.phys_model.train()
        self.phys_model.reinit_envs(
            opts["phys_batch"], wdw_length=opts["phys_wdw_len"], is_eval=False
        )
        for i in tqdm.tqdm(range(self.iters_per_phys_cycle)):
            self.phys_model.set_progress(self.current_steps_phys)
            self.run_phys_iter()
            self.current_steps_phys += 1

        # eval again
        self.phys_model.eval()
        self.run_phys_visualization(tag="phys")

    def run_phys_iter(self):
        """Run physics optimization"""
        phys_aux = self.phys_model()
        self.phys_model.backward(phys_aux["total_loss"])
        self.phys_model.update()
        if get_local_rank() == 0:
            del phys_aux["total_loss"]
            self.add_scalar(self.log, phys_aux, self.current_steps_phys)

    def run_phys_visualization(self, tag=""):
        opts = self.opts
        frame_offset_raw = self.phys_model.frame_offset_raw
        vid_frame_max = max(frame_offset_raw[1:] - frame_offset_raw[:-1])
        self.phys_model.reinit_envs(1, wdw_length=vid_frame_max, is_eval=True)
        for vidid in opts["phys_vid"]:
            frame_start = torch.zeros(1) + frame_offset_raw[vidid]
            _ = self.phys_model(frame_start=frame_start.to(self.device))
            img_size = tuple(self.data_info["raw_size"][vidid][::-1])
            img_size = img_size + (0.5,)  # scale
            data = self.phys_model.query(img_size=img_size)
            self.phys_visualizer.show(
                "%s-%02d-%05d" % (tag, vidid, self.current_steps_phys),
                data,
                fps=1.0 / self.phys_model.frame_interval,
            )
