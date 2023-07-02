Reconstruct an arbitrary instance
========================================

In the first tutorial, we show how to reconstruct an arbitrary instance from a single video, 
taking `car-turnaround-2` as an example. 

.. raw:: html

  <div style="display: flex; justify-content: center;">
   <video width="100%" src="/lab4d/_static/media/car-turnaround-2.mp4" controls autoplay muted>
     Your browser does not support the video tag.
   </video>
  </div>

.. note:: 
  To reconstruct a complete shape, the video should contain sufficiently diverse viewpoint of the object. 

Download preprocessed data
---------------------------------------

To download preprocessed data, run::

  bash scripts/download_unzip.sh "https://www.dropbox.com/s/cv90orj961fibxt/car-turnaround-2-0000.zip"

This will download and unzip the preprocessed data to `database/processed/$type-of-processed-data/Full-Resolution/car-turnaround-2-0000/`.

To use customized videos, see the `preprocessing tutorial </lab4d/tutorials/preprocessing.html>`_.

.. note:: 

  The preprocessed data is stored with the following structure under `database/processed/`:

  - JPEGImages/Full-Resolution/$seqname/%05d.jpg
  
    - stores the raw rgb images (after flow filtering that removes static frames)

  - Annotations/Full-Resolution/$seqname/%05d.{npy,jpg}

    - .npy stores an image array with instance ids. We store the array as np.int8. 

      - If there is no detection, set all pixels values to -1

      - value 0: background

      - 1...127: instance ids. Currently only support 1 instance per video (id=1).

    - .jpg is for visualization purpose

  - Features/Full-Resolution/$seqname/{densepose-%02d.npy, dinov2-%02d.npy}
  
    - stores pixel features of segmented objects, either from `DensePose-CSE <https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_CSE.md>`_ (for either human or quadruped animals) or `DINOv2 <https://ai.facebook.com/blog/dino-v2-computer-vision-self-supervised-learning/>`_ (for generic objects).

  - Flow{FW,BW}_%02d/Full-Resolution/$seqname/%05d.npy
  
    - stores forward / backward optical flow and their uncertainty from `VCNPlus <https://github.com/gengshan-y/rigidmask>`_.

  - Depth/Full-Resolution/$seqname/%05d.npy
    
    - stores depth maps estimated by `ZoeDepth <https://github.com/isl-org/ZoeDepth>`_

  - Cameras/Full-Resolution/$seqname/%02d.npy

    - world-to-camera transformations (00.npy) and object-to-camera transformations (01.npy).

    - We use the opencv coordinate system convention defined as follows:

      - x: right

      - y: down

      - z: forward

  The metadata file is stored at `database/configs/...`, which is used to load the dataset.


Visualize preprocessed data
---------------------------------------
Before training, let's check the accuracy of the pesudo ground-truth. 

Instance segmentation and tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualization of instance segmentation and tracking can be found at `Annotations/Full-Resolution/$seqname/vis.mp4`:

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="100%" src="/lab4d/_static/media/car-turnaround-2-anno.mp4" controls autoplay muted>
      Your browser does not support the video tag.
    </video>
  </div>


Optical flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualization of Optical flow can be found at `Flow{FW,BW}_%02d/Full-Resolution/$seqname/visflow-%05d.jpg`. Color indicates flow direction 
and length indicates flow magnitude. The empty region is where the flow is uncertain.

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <image width="100%" src="/lab4d/_static/images/visflo-00081.jpg"> </image>
  </div>


World / object to camera transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Visualizations of world / object to camera transformations can be found at `Cameras/Full-Resolution/$seqname/*.obj`. To visualize .obj files, 
use the vscode-3d-preview extension of VS Code, or download to local and open with meshlab.

Below we show sparsely-annotated transformations (1st, `...canonical-prealign.obj`) 
and full transformations of all frames (2nd, `...canonical.obj`):

.. raw:: html

  <style>
    .model-container {
      width: 100%;
    }

    @media (min-width: 768px) {
      .model-container {
        width: 50%;
      }
    }

    model-viewer {
      width: 100%;
      height: 400px;
    }
  </style>

  <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
    <div class="model-container">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/car-turnaround-2-canonical-prealign.glb" auto-rotate camera-controls>
      </model-viewer>
    </div>
    <div class="model-container">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/car-turnaround-2-canonical.glb" auto-rotate camera-controls>
      </model-viewer>
    </div>
  </div>

.. note::

  We assume opencv coordinate convention in the above visualizations. Each camera is represented by three axes: x (red, right), y (green, down), z (blue, forward).
  The object-to-camera transformations are roughly annotated in 12 frames and refined and propogated to all 120 frames using flow and monocular depth. 


Model Training
---------------------------------------

In this stage, we use the pseudo ground-truth from the previous steps to train dynamic neural fields. 
The camera transformations are used to initialize the model. 
The other data including rgb, segmentation, flow, and depth are used to supervise the model.

Run::

  # Args: training script, gpu id, args for training script
  bash scripts/train.sh lab4d/train.py 0 --seqname car-turnaround-2 --logname fg-rigid --fg_motion rigid

.. note::
  The optimization takes around 14 minutes on a 3090. 
  You may find the list of flags at `lab4d/engine/config.py`.

  By default we use 20 batches (each batch contains 200 iterations), 
  which leads to a good reconstruction quality and is used for developement purpose.
  To get higher quality, train for more iterations by adding `--num_batches 120`. The rendering results in this page assumes 120 batches, which takes 1.5 hours.
  

Visualization during training
---------------------------------------
- We use tensorboard to monitor losses and visualize intermediate renderings. Tensorboard logs are saved at `logdir/$logname`. To use tensorboard in VS Code, hold `shift+cmd+p` and select launch tensorboard.
- Camera transformations and a low-res proxy geometry are saved at `logdir/$logname/...proxy.obj`. We show the final proxy geometry and cameras below:

.. raw:: html

  <style>
    model-viewer {
      width: 100%;
      height: 400px;
      
    }
  </style>

  <div style="display: flex; justify-content: center;">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/car-turnaround-2-proxy.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>

- To render a video of the proxy geometry and cameras over training iterations, run::

    python scripts/render_intermediate.py --testdir logdir/$logname/

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media/car-turnaround-2-proxy.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>

We provide the a checkpoint trained with `--num_batches 120` (equivalent to 24k iterations). Download and unzip to `logdir/car-turnaround-2-fg-rigid-b120` by running::

  bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/tyutfhzhm4h3gpxq3lser/log-car-turnaround-2-fg-rigid-b120.zip?dl=0&rlkey=uic2ea0hm0nts30tnyac1dt82"

Rendering after training
---------------------------------------
To render the reference view, run::

  # reference view
  python lab4d/render.py --flagfile=logdir/$logname/opts.log --load_suffix latest --render_res 256

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="50%" src="/lab4d/_static/media/car-turnaround_ref.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="50%" src="/lab4d/_static/media/car-turnaround_ref-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>

On the left we show the rgb rendering and on the right we show the dense corresonpdence (same color indicates the same canonical surface point).



To render novel views, run::

  # turntable views, --viewpoint rot-elevation-angles
  python lab4d/render.py --flagfile=logdir/$logname/opts.log --load_suffix latest --viewpoint rot-0-360 --render_res 256

  # birds-eye-views, --viewpoint bev-elevation
  python lab4d/render.py --flagfile=logdir/$logname/opts.log --load_suffix latest --viewpoint bev-90 --render_res 256

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <video width="25%" src="/lab4d/_static/media/car-turnaround_turntable-120.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="25%" src="/lab4d/_static/media/car-turnaround_turntable-120-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="25%" src="/lab4d/_static/media/car-turnaround_bev-120.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
    <video width="25%" src="/lab4d/_static/media/car-turnaround_bev-120-xyz.mp4" controls autoplay muted loop>
      Your browser does not support the video tag.
    </video>
  </div>


.. note:: 

  Rendering the above video at 256x256 takes ~40s on a 3090 (~0.4s/frame).
  The default rendering resolution is set to 128x128 for fast rendering.

Exporting meshes and motion parameters after training
-----------------------------------------------------------

To export meshes and motion parameters, run::

    python lab4d/export.py --flagfile=logdir/$logname/opts.log --load_suffix latest --level 0.005

.. note:: 

  The `--level`` parameter is the contour value that marching cubes use to search for isosurfaces.
  The default value 0.0 should work in most cases. In this example, we use 0.005 to obtain a more complete surface.

.. raw:: html

  <style>
    model-viewer {
      width: 100%;
      height: 400px;
      
    }
  </style>

  <div style="display: flex; justify-content: center;">
      <model-viewer autoplay ar shadow-intensity="1"  src="/lab4d/_static/meshes/car-turnaround-2-mesh.glb" auto-rotate camera-controls>
      </model-viewer>
  </div>

Visit other `tutorials </lab4d/tutorials/#content>`_.