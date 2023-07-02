.. Lab4D documentation master file, created by
   sphinx-quickstart on Fri Jun  2 20:54:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tutorials
=================================

Overview
---------------------------------
Inferring 4D representations given 2D observations is challenging due to its under-constrained nature. 
With recent advances in differentiable rendering, visual correspondence and segmentation, we built an optimization framework that 
reconstructs dense 4D structures with test-time optimization, by minimizing the different between the rendered 2D images and the input observations.

The tutorials introduce a complete workflow of Lab4D. We'll use the method and dataset from the following papers:

- `BANMo: Building Animatable 3D Neural Models from Many Casual Videos <https://banmo-www.github.io/>`_, CVPR 2022.
- `RAC: Reconstructing Animatable Categories from Videos <https://gengshan-y.github.io/rac-www/>`_, CVPR 2023.
- `Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis <https://andrewsonga.github.io/totalrecon/>`_, Arxiv 2023.

`The tutorials assumes a basic familiarity with Python and Differentiable Rendering concepts.`

Each of the tutorial can be executed in a couple of ways:

- **Customized videos**: This option allows you to train a model on your own videos.
- **Preprocessed data**: This option skips the preprocessing step and train models on the `preprocessed data </lab4d/data_models.html>`_ we provide.
- **Render-only**: This option skips model training and allows you to render the `pre-trained model weights </lab4d/data_models.html>`_ we provide.


Content
---------------------------------
.. toctree::
   :maxdepth: 1

   arbitrary_video
   single_video_cat
   multi_video_cat
   category_model
   preprocessing

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
