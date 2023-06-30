Reconstruct a category from videos
=======================================

In this tutorial, we build a shape and pose model of a category using ~48 videos of different human, similar to the setup of `RAC <https://gengshan-y.github.io/rac-www/>`_.

.. raw:: html

  <div style="display: flex; justify-content: center;">
   <video width="50%" src="/lab4d/_static/media/human-48.mp4" controls autoplay muted>
     Your browser does not support the video tag.
   </video>
  </div>

Get pre-processed data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, download pre-processeed data (20G)::

  bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/c6lrg2aaabat4gu57avbq/human-48.zip?dl=0&rlkey=ezpc3k13qgm1yqzm4v897whcj"

.. note::
  
  The command to pre-process the `human-48`` dataset is::

    # Args: video name, quadruped category, on gpu 0-6
    python scripts/run_preprocess.py human-48 human "0,1,2,3,4,5,6"

  To use customized videos, see the `preprocessing tutorial </lab4d/tutorials/preprocessing.html>`_.

.. note::

  Besides `human-48`, you man download the pre-processed data of `cat-85`` and `dog-98` with::

    bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/xfaot22qbzz0o0ncl5bna/cat-85.zip?dl=0&rlkey=wcer6lf0u4en7tjzaonj5v96q"
    bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/h2m7f3jqzm4a2u3lpxhki/dog-98.zip?dl=0&rlkey=x4fy74mbk7qrhc5ovmt4lwpkg"

Training
-----------

To train the dynamic neural fields ::

  # Args: training script, gpu id, input args
  bash scripts/train.sh lab4d/train.py 0,1,2,3,4,5,6 --seqname human-48 --logname skel-soft --fg_motion comp_skel-human_dense --nosingle_inst --num_batches 120

.. note::

  In this setup, we follow RAC and `HumanNeRF <https://grail.cs.washington.edu/projects/humannerf/>`_ 
  to use a hybrid deformation model. The hybrid model contains both a skeleton and soft deformation fields
  (`--fg_motion comp_skel-human_dense`). The skeleton explains the rigid motion and the soft deformation fields explain the remaining motion.

  Skeleton specifies the 3D rest joint locations and a tree topology (in `lab4d/nnutils/pose.py`).
  We provide a human skeleton (modified from the mojuco human format) and a quadruped skeleton (modified from `Mode-Adaptive Neural Networks for Quadruped Motion Control <https://github.com/sebastianstarke/AI4Animation>`_).

  We also add `--nosingle_inst` to enable instance-specific morphology code, which represents the between-instance
  shape and bone length variations.


Visualization during training
--------------------------------
Here we show the final bone locations (1st), camera transformations and geometry (2nd).

.. raw:: html

  <style>
    model-viewer {
      width: 100%;
      height: 400px;
      
    }
  </style>

  <div style="display: flex; justify-content: center;">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/human-48-bone.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>

    <div style="display: flex; justify-content: center;">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/human-48-proxy.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>

The camera transformations are sub-sampled to 200 frames to speed up the visualization.

Rendering after training
----------------------------
To render reference view of a video (e.g., video 0), run::

  # reference view
  python lab4d/render.py --flagfile=logdir/$logname/opts.log --load_suffix latest --inst_id 0 --render_res 256

.. note::

  Some of the frames with small motion are not rendered (determined by preprocessing). 

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media/human-48-0_ref.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media/human-48-0_ref-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>


To render novel views, run::

  # turntable views, --viewpoint rot-elevation-angles
  python lab4d/render.py --flagfile=logdir/$logname/opts.log --load_suffix latest  --inst_id 0 --viewpoint rot-0-360 --render_res 256


.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media/human-48-0_turntable-120.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media/human-48-0_turntable-120-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>


Exporting meshes and motion parameters after training
--------------------------------------------------------

To export meshes and motion parameters of video 0, run::

    python lab4d/export.py --flagfile=logdir/$logname/opts.log --load_suffix latest --inst_id 0

.. raw:: html

  <style>
    model-viewer {
      width: 100%;
      height: 400px;
      
    }
  </style>

  <div style="display: flex; justify-content: center;">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/human-48-0-mesh.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>


Re-animation
----------------------------
RAC disentangles the space of morphology and motion, which enables motion transfer between instances.

We show the re-animation results of re-animating the motion of video 0 while keeping the instance detail of video 8.
To render the re-animated video, run::

  # reanimation in the reference view
  python lab4d/reanimate.py --flagfile=logdir/$logname/opts.log --load_suffix latest --motion_id 0 --inst_id 8 --render_res 256

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media/human-48-reanimate-8.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media/human-48-reanimate-8-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>

Visit other `tutorials </lab4d/tutorials/#content>`_.