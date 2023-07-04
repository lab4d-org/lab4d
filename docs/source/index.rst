.. Lab4D documentation master file, created by
   sphinx-quickstart on Fri Jun  2 20:54:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Lab4D's documentation!
=================================

**Lab4D** is a framework for 4D reconstruction from monocular videos. 

Features
-------------------------------
- Representation

  - neural implicit representation

  - deformation fields (neural fields, control-points, skeleton)

  - compositional scene

  - category-level models

- Interface for priors

  - pixelwise priors: depth, flow, DINOv2 features

  - segmentation: track-anything, video instance segmentation

  - camera viewpoint: viewpoint network, manual annotation

- Efficiency

  - multi-gpu training

  - dual-quaternion ops

.. note::

  This is an alpha release and the APIs are subject to change as we continuously improve and refine it. 
  We encourage users to provide feedback and report bugs via `github issues <https://github.com/lab4d-org/lab4d/issues/new/choose>`_. 
  Thank you for your support. 
