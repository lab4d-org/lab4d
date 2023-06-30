Pre-process customized videos
========================================

In this tutorial, we show how to preprocess customized videos that can be Later used for training, taking `car-turnaround-2` as an example. 
To preprocess a custommized video (or a folder of videos), run::

  # Args: folder name, category from {human, quad, other}, on gpu 0
  python scripts/run_preprocess.py car-turnaround-2 other "0"
  
This will produce the results we saw in the first tutorial.

.. note::
    The processing of `human` and `quad` (quadruped) categories are fully automatic.
    When processing the `other` cateogry, the user will be prompted in terminal to open a link to a gradio page.
    Follow the link to finish tracking and camera rotation annotations.

`Next, we will get into the details of segmentation and camera transformations.`

Frame filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To remove static frames which does not provide extra signal for reconstruction (i.e., frames without motion or with small motion), 
we run optical flow over consecutive frames and skip a frame if the median flow magnitude is smaller than a threshold.

.. note::
    There is a flag in `scripts/run_preprocess.py`` that you can set to False to turn off frame filtering.

Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For human and quadruped animals, we use `MinVIS <https://github.com/NVlabs/MinVIS>`_ to get segmentations.
MinVIS is automatic but only works for a fixed vocabulary.
For the other categories, we use `Track-Anything <https://github.com/gaomingqi/Track-Anything>`_, which asks the user to specify a point on the object of interest.
  
.. note::

  There is a flag in `scripts/run_preprocess.py`` that switches the segmentation method.


Object-to-camera transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use BANMo-viewpoint network to estimate to viewpoint / rotation for human and quadruped animals.

For the other categories, we ask the user to annotate the camera rotations for a few frames (by aligning the orientation of a reference 3D model to the input image) as shown below.

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <image width="100%" src="/lab4d/_static/images/camera_annot.png"> </image>
  </div>

Later, we run a script (`preprocess/scripts/{canonical/camera_registration.py}`) that uses optical flow and depth to propogate and smooth the rotation annotations.

The translations are approximated with the 2D object center and size (from segmentation) 
using a orthographic camera model. 

.. note::

  To align the 3D model with the input image, use the sidebar to mark the roll, elevation and azimuth angles of the camera. Remember to click save after finishing a frame, and click exit after finishing all the videos.
  We recommend aligning  the camera when object undergoes every 90 degree of rotation (e.g., from front-facing to facing left).
  
  There is a flag in `scripts/run_preprocess.py`` that switches the camera estimation method.

  

Visit other `tutorials </lab4d/tutorials/#content>`_.