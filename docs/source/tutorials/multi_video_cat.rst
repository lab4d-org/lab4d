3. Reconstruct a cat from multiple videos
==========================================

In the previous tutorial, we reconstructed a cat with a single video. 
In this example, we improve the reconstruction by feeding more videos of the same cat to the pipeline, similar to the setup of `BANMo <https://banmo-www.github.io/>`_.

Get pre-processed data
-------------------------

First, download pre-processeed data::

  bash scripts/download_unzip.sh "https://www.dropbox.com/s/3w0vhh05olzwwn4/cat-pikachu.zip"


To use custom videos, see the `preprocessing tutorial </lab4d/tutorials/preprocessing.html>`_.


Training
-----------

To train the dynamic neural fields::

  # Args: training script, gpu id, input args
  bash scripts/train.sh lab4d/train.py 0,1 --seqname cat-pikachu --logname fg-bob --fg_motion bob --reg_gauss_skin_wt 0.01


.. note::

  In this setup, we follow BANMo to use neural blend skinning with 25 bones (`--fg_motion bob`). 
  We also use a larger weight for the gaussian bone regularization (`--reg_gauss_skin_wt 0.01`) to encourage the bones to be inside the object.

.. note::

  Since there are more video frames than the previous example, we want to get more samples of rays in each (mini)batch.
  This can be achieved by specifying a larger per-gpu batch size (e.g., `--imgs_per_gpu 224`) or using more gpus.

  The number of rays per (mini)batch is computed as `number of gpus` x `imgs_per_gpu` x `pixels_per_image`.

.. note::
  The training takes around 20 minutes on two 3090 GPUs.
  You may find the list of flags at `lab4d/config.py <https://github.com/lab4d-org/lab4d/blob/main/lab4d/config.py>`_.
  The rendering results in this page assumes 120 rounds.

Visualization during training
---------------------------------
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
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/cat-pikachu-bone.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>

    <div style="display: flex; justify-content: center;">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/cat-pikachu-proxy.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>

The camera transformations are sub-sampled to 200 frames to speed up the visualization.

Rendering after training
----------------------------
After training, we can check the reconstruction quality by rendering the reference view and novel views. 
Pre-trained checkpoints are provided `here </lab4d/data_models.html#checkpoints>`_.

To render reference view of a video (e.g., video 8), run::

  # reference view
  python lab4d/render.py --flagfile=logdir/$logname/opts.log --load_suffix latest --inst_id 8 --render_res 256

.. note::

  Some of the frames with small motion are not rendered (determined by preprocessing). 

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-8_ref.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-8_ref-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>


To render novel views, run::

  # turntable views, --viewpoint rot-elevation-angles
  python lab4d/render.py --flagfile=logdir/$logname/opts.log --load_suffix latest  --inst_id 8 --viewpoint rot-0-360 --render_res 256


.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-8_turntable-120.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media_resized/cat-pikachu-8_turntable-120-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>


Exporting meshes and motion parameters after training
------------------------------------------------------------

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
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/cat-pikachu-mesh.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>

.. note::

  The default setting may produce broken meshes. To get better one as shown above, train for more iterations by adding `--num_rounds 120`. Also see `this <https://github.com/lab4d-org/lab4d/issues/46#issuecomment-2206518886>`_ for an explanation.

Visit other `tutorials </lab4d/tutorials/#content>`_.