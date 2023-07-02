3. Reconstruct a cat from multiple videos
==========================================

In the previous tutorial, we reconstructed a cat with a single video. 
In this example, we improve the reconstruction by feeding more videos of the same cat to the pipeline, similar to the setup of `BANMo <https://banmo-www.github.io/>`_.

Get pre-processed data
-------------------------

First, download pre-processeed data::

  bash scripts/download_unzip.sh "https://www.dropbox.com/s/3w0vhh05olzwwn4/cat-pikachu.zip"


To use customized videos, see the `preprocessing tutorial </lab4d/tutorials/preprocessing.html>`_.

.. note::

  Preprocessing takes about 90 minutes for 11 videos on a single device. To speed up the pre-processing, 
  we can parallelize tasks over multiple gpus with the following::
  
    # Args: video name, quadruped category, on four gpus 0,1,2,3
    python scripts/run_preprocess.py cat-pikachu quad "0,1,2,3"


Training
-----------

To train the dynamic neural fields::

  # Args: training script, gpu id, input args
  bash scripts/train.sh lab4d/train.py 0,1 --seqname cat-pikachu --logname fg-bob --fg_motion bob --reg_gauss_mask_wt 0.1


.. note::

  In this setup, we follow BANMo to use neural blend skinning with 25 bones (`--fg_motion bob`). 
  We also use a larger weight on the gaussian bone rendering loss (`--reg_gauss_mask_wt 0.1`) to encourage the bones to lie within the object surface.

.. note::

  Since there are more video frames than the previous example, we want to set a larger number of ray samples per-batch.
  This can be achieved by specifying a larger per-gpu batch size (e.g., `--minibatch_size 224`) 
  or using more gpus.

  The number of rays per-batch is computed as `number of gpus` x `minibatch_size`.

.. note::
  The training takes around 20 minutes on two 3090 GPUs.
  You may find the list of flags at `lab4d/config.py <https://github.com/lab4d-org/lab4d/blob/main/lab4d/config.py>`_.
  The rendering results in this page assumes 120 batches.

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
    <video width="50%" src="/lab4d/_static/media/cat-pikachu-8_ref.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media/cat-pikachu-8_ref-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>


To render novel views, run::

  # turntable views, --viewpoint rot-elevation-angles
  python lab4d/render.py --flagfile=logdir/$logname/opts.log --load_suffix latest  --inst_id 8 --viewpoint rot-0-360 --render_res 256


.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media/cat-pikachu-8_turntable-120.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media/cat-pikachu-8_turntable-120-xyz.mp4" controls autoplay muted loop>
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

Visit other `tutorials </lab4d/tutorials/#content>`_.