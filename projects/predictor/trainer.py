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
from lab4d.engine.model import dvr_model
from projects.predictor import config
from projects.predictor.predictor import Predictor

from projects.predictor.dataloader.loader import PredictorLoader


class PredTrainer(Trainer):
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
            find_unused_parameters=False,
        )

    def define_dataset(self):
        opts = self.opts
        self.total_steps = opts["num_rounds"] * opts["iters_per_round"]

        dataset_constructor = PredictorLoader(opts)
        self.trainloader = dataset_constructor.get_loader("train")
        self.evalloader = dataset_constructor.get_loader("eval")

        # meta data
        self.eval_fid = np.linspace(0, len(self.evalloader) - 1, 9).astype(int)
        self.data_info = {"apply_pca_fn": None}

    def construct_dataset_opts(self, opts):
        return opts

    def define_model(self):
        self.device = torch.device("cuda:{}".format(get_local_rank()))
        self.model = Predictor(self.opts)

        # ddp
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = self.model.to(self.device)

    def load_checkpoint_train(self):
        pass

    def get_optimizable_param_list(self):
        params_ref_list = []
        params_list = []
        lr_list = []
        for name, p in self.model.named_parameters():
            if name.startswith("module.backbone"):
                continue
            lr = self.opts["learning_rate"]
            params_ref_list.append({name: p})
            params_list.append({"params": p})
            lr_list.append(lr)
            if get_local_rank() == 0:
                print(name, p.shape, lr)
        return params_ref_list, params_list, lr_list

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

    def update_aux_vars(self):
        pass

    def construct_eval_batch(self, batch):
        pass

    def load_batch(self, dataset, fids):
        ref_dict = defaultdict(list)
        batch_aggr = defaultdict(list)

        for fid in fids:
            batch_aggr["index"].append(fid)
            print("loading fid %d" % fid)

        self.model.convert_img_to_pixel(batch_aggr)
        ref_dict["ref_rgb"] = batch_aggr["img"].permute(0, 2, 3, 1).cpu().numpy()

        return ref_dict, batch_aggr
