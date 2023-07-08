2. Reconstruct a cat from a single video
==========================================

Previously, we've reconstructed a rigid body (a car). In this example, we show how to reconstruct a deformable object (a cat!).

.. raw:: html

  <div style="display: flex; justify-content: center;">
   <video width="100%" src="/lab4d/_static/media_resized/cat-pikachu-0.mp4" controls autoplay muted>
     Your browser does not support the video tag.
   </video>
  </div>

Get pre-processed data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, download and extract pre-processeed data::

  bash scripts/download_unzip.sh "https://www.dropbox.com/s/mb7zgk73oomix4s/cat-pikachu-0.zip"

To use custom videos, see the `preprocessing tutorial </lab4d/tutorials/preprocessing.html>`_.

Training
^^^^^^^^^^^

To optimize the dynamic neural fields::

  # Args: training script, gpu id, input args
  bash scripts/train.sh lab4d/train.py 0 --seqname cat-pikachu-0 --logname fg-skel --fg_motion skel-quad 

The difference from the previous example is that we model the object motion with a skeleton-based 
deformation field, instead of treating it as a rigid body.

You may choose `fg_motion` from one of the following motion fields: 
  - rigid: rigid motion field (i.e., root body motion only, no deformation)
  - dense: dense motion field (similar to `D-NeRF <https://www.albertpumarola.com/research/D-NeRF/index.html>`_)
  - bob: bag-of-bones motion field (neural blend skinning in `BANMo <https://banmo-www.github.io/>`_)
  - skel-human/quad: human or quadruped skeleton motion field (in `RAC <https://gengshan-y.github.io/rac-www/>`_)
  - comp_skel-human/quad_dense: composed motion field (with skeleton-based deformation and soft deformation in `RAC <https://gengshan-y.github.io/rac-www/>`_)

.. note::

  The optimization uses 13G GPU memory and takes around 21 minutes on a 3090 GPU. You may find the list of flags at `lab4d/config.py <https://github.com/lab4d-org/lab4d/blob/main/lab4d/config.py>`_.
  To get higher quality, train for more iterations by adding `--num_batches 120`. 

  To run on a machine with less GPU memory, you may reduce the `--minibatch_size`.


Visualization during training
------------------------------------------
Please use tensorboard to monitor losses and intermediate renderings.

Here we show the final bone locations (1st), camera transformations and geometry (2nd).

.. raw:: html

  <style>
    model-viewer {
      width: 100%;
      height: 400px;
      
    }
  </style>

  <div style="display: flex; justify-content: center;">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/cat-pikachu-0-bone.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>

    <div style="display: flex; justify-content: center;">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/cat-pikachu-0-proxy.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>


Rendering after training
----------------------------
After training, we can check the reconstruction quality by rendering the reference view and novel views. 
Pre-trained checkpoints are provided `here </lab4d/data_models.html#checkpoints>`_.

To render reference views of the input video, run::

  # reference view
  python lab4d/render.py --flagfile=logdir/$logname/opts.log --load_suffix latest --render_res 256

.. note::

  Some of the frames are skipped during preprocessing (according to static-frame filtering) 
  Those filtered frames are not used for training, and not rendered here.

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-0_ref.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-0_ref-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>


To render novel views, run::

  # turntable views, --viewpoint rot-elevation-angles --freeze_id frame-id-to-freeze
  python lab4d/render.py --flagfile=logdir/$logname/opts.log --load_suffix latest --viewpoint rot-0-360 --render_res 256 --freeze_id 50


.. note::
  
    The `freeze_id` is set to 50 to freeze the time at the 50-th frame while rotating the camera around the object.

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-0_turntable.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-0_turntable-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>

To render a video of the proxy geometry and cameras over training iterations, run::

  python scripts/render_intermediate.py --testdir logdir/$logname/

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-0-proxy.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>

Exporting meshes and motion parameters after training
--------------------------------------------------------

To export meshes and motion parameters, run::

    python lab4d/export.py --flagfile=logdir/$logname/opts.log --load_suffix latest

.. raw:: html

  <style>
    model-viewer {
      width: 100%;
      height: 400px;
      
    }
  </style>

  <div style="display: flex; justify-content: center;">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/cat-pikachu-0-mesh.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>


Reconstruct the total scene
------------------------------------------------------------

Now we have reconstructed the cat, can we put the cat in the scene? To do so, we train compositional neural fields with a foreground and a background component.
Run the following to load the pre-trained foreground field and train the composed fields::

    # Args: training script, gpu id, input args
    bash scripts/train.sh lab4d/train.py 0 --seqname cat-pikachu-0 --logname comp-comp-s2 --field_type comp --fg_motion comp_skel-quad_dense --data_prefix full --num_batches 120 --load_path logdir/cat-pikachu-0-fg-skel/ckpt_latest.pth
    
.. note::

    The `file_type` is changed `comp` to compose the background field with the foreground field during 
    differentiable rendering.

    The `fg_motion` is changed to `comp_skel-quad_dense` to use the composed warping field (with skeleton-based deformation and soft deformation) for the foreground object.

    To reconstruct the background, the `data_prefix` is changed to `full` to load the full frames instead of frames cropped around the object.

.. note::

    We load the pretrained foreground model `logdir/cat-pikachu-0-fg-skel/ckpt_latest.pth` to initialize the optimization.
  
    The optimization of 120 batches (24k minibatches/iterations) takes around 3.5 hours on a 3090 GPU. 


To render videos from the bird's eye view::

  # bird's eye view, elevation angle=20 degree
  python lab4d/render.py --flagfile=logdir/cat-pikachu-0-comp-comp-s2/opts.log --load_suffix latest --render_res 256 --viewpoint bev-20

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-0-comp_bev.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-0-comp_bev-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>


Visit other `tutorials </lab4d/tutorials/#content>`_.