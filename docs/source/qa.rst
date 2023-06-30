Q&A
===========================

Installation
---------------------------
- Conda/mamba is not able to resolve conflicts when installing packages.

  - Possible cause: The base conda environment is not clean. See the discussion `in this thread <https://stackoverflow.com/questions/57243296/why-is-it-recommended-to-not-install-additional-packages-in-the-conda-base-envir>`_.
  
  - Fix: Remove packages of the base environment that causes the conflict.

Model training
---------------------------

- Training on >50 videos might cause the following os error::

   [Errno 24] Too many open files

  - To check the current file limit, run::
    
        ulimit -S -n

    To increate open file limit to 4096, run::
      
        ulimit -u -n 4096

- Multi-GPU training hangs but single-GPU training works fine.

  - Run training script with `NCCL_P2P_DISABLE=1 bash scripts/train.sh ...` to disable direct GPU-to-GPU (P2P) communication. See discussion `here <https://github.com/NVIDIA/nccl/issues/631>`_.
  