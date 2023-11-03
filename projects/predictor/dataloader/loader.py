# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import configparser
import glob
import os
import random
import pdb

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from lab4d.engine.trainer import get_local_rank
from lab4d.dataloader.data_utils import duplicate_dataset

from projects.predictor.dataloader.dataset import CustomDataset


class PredictorLoader:
    def __init__(self, opts):
        self.opts_train = self.construct_opts(opts, "train")
        self.opts_eval = self.construct_opts(opts, "eval")

    def construct_opts(self, opts, mode="train"):
        opts_dict = {}
        opts_dict["num_workers"] = opts["num_workers"]
        opts_dict["ngpu"] = opts["ngpu"]
        opts_dict["iters_per_round"] = opts["iters_per_round"]
        opts_dict["imgs_per_gpu"] = opts["imgs_per_gpu"]
        opts_dict["local_rank"] = get_local_rank()

        if mode == "train":
            opts_dict["multiply"] = True
        else:
            opts_dict["multiply"] = False
        return opts_dict

    def get_loader(self, mode="train"):
        if mode == "train":
            return self.train_loader()
        elif mode == "eval":
            return self.eval_loader()
        else:
            raise NotImplementedError

    def train_loader(self):
        """Construct the training dataloader.

        Args:
            opts_dict (Dict): Defined in Trainer::construct_dataset_opts()
        Returns:
            dataloader (:class:`pytorch:torch.utils.data.DataLoader`): Training dataloader
        """
        opts_dict = self.opts_train
        # Set to 0 to debug the data loader
        num_workers = opts_dict["num_workers"]
        # num_workers = min(num_workers, 4)
        # num_workers = 0
        print("# workers: %d" % num_workers)
        print("# iterations per round: %d" % opts_dict["iters_per_round"])
        print(
            "# image samples per iteration: %d"
            % (opts_dict["imgs_per_gpu"] * opts_dict["ngpu"])
        )

        dataset = self.get_dataset(opts_dict)

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=opts_dict["ngpu"],
            rank=opts_dict["local_rank"],
            shuffle=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=opts_dict["imgs_per_gpu"],
            num_workers=num_workers,
            drop_last=True,
            # worker_init_fn=_init_fn,
            pin_memory=True,
            sampler=sampler,
        )
        return dataloader

    def eval_loader(self):
        """Construct the evaluation dataloader.

        Args:
            opts_dict (Dict): Defined in Trainer::construct_dataset_opts()
        Returns:
            dataloader (torch.utils.data.DataLoader): Evaluation dataloader
        """
        num_workers = 0

        dataset = self.get_dataset(self.opts_eval)
        dataset = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=False,
        )
        return dataset

    def get_dataset(self, opts):
        """Construct a PyTorch dataset that includes all videos in a sequence.

        Args:
            opts (Dict): Defined in Trainer::construct_dataset_opts()
            is_eval (bool): Unused
            gpuid (List(int)): Select a subset based on gpuid for npy generation
        Returns:
            dataset (torch.utils.data.Dataset): Concatenation of datasets for each
                video in the sequence `opts["seqname"]`
        """
        dataset = CustomDataset(opts)
        datalist = [dataset]

        if opts["multiply"]:
            datalist = duplicate_dataset(opts, datalist)

        dataset = torch.utils.data.ConcatDataset(datalist)
        return dataset
