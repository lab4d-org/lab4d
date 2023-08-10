5. Pre-process custom videos
========================================

In this tutorial, we show how to preprocess custom videos that can be later used for training. We provide some 
`raw videos </lab4d/data_models.html#raw-videos>`_ for you to try out. 
The download links are provided as `database/vid_data/$seqname`, where `$seqname`` is the name of the sequence.

Taking `cat-pikachu-0` in the `second tutorial </lab4d/tutorials/single_video_cat.html>`_ for example, 
run the following to download and process the data::

  # Args: sequence name, text prompt (segmentation), category from {human, quad, other} (camera viewpoint), gpu id
  python scripts/run_preprocess.py cat-pikachu-0 cat quad "0"

.. note::
    To preprocess other videos, create a folder named `database/raw/$seqname`, move the videos into it, and run the above command.

`Next, we will get into the details of processing.`

Frame filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, we first remove near-static frames (i.e., frames without motion or with small motion) since they do not provide useful extra signal for reconstruction.
To do so, we run optical flow over consecutive frames and skip a frame if the median flow magnitude is smaller than a threshold.

.. note::
    There is a flag in `scripts/run_preprocess.py`` that you can set to False to turn on/off frame filtering.

Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide a web GUI and a command line interface for object segmentation. 

**Interactive segmentation**: `Track-Anything <https://github.com/gaomingqi/Track-Anything>`_ will be used given text prompt "other", e.g.,::
  
    python scripts/run_preprocess.py cat-pikachu-0 other quad "0"

It creates a web interfaces and asks the user to specify point prompts on the object of interest.


**Automatic segmentation**: `Grounding-DINO <https://github.com/IDEA-Research/GroundingDINO>`_ will be used to determin which object to track 
in the first frame given a valid text prompt e.g., ::
    
    python scripts/run_preprocess.py cat-pikachu-0 cat quad "0"

  
.. note::

  There is a flag in `scripts/run_preprocess.py`` that switches the segmentation method.


Object-to-camera transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For human and quadruped animals, we use a viewpoint network (presented in BANMo) to estimate the camera viewpoint / rotation with regard to a canonical 3D coordinate.

For other categories, user will be asked to annotate camera viewpoints (by aligning the orientation of a reference 3D model to the input image)  for a few frames as shown below.

.. raw:: html

  <div style="display: flex; justify-content: center;">
    <image width="100%" src="/lab4d/_static/images/camera_annot.png"> </image>
  </div>

.. note::

  To align the 3D model with the provided image, utilize the sidebar to specify the camera's roll, elevation, and azimuth angles. After adjusting each frame, ensure you click 'save.' Once you've completed adjustments for all the videos, click 'exit.'
  We suggest making an annotation every time the object turns 90 degrees, such as when it changes from a front-facing position to facing left.
  
  In the `scripts/run_preprocess.py` file, there's a flag that allows you to change the method used for camera estimation."
  
After getting the sparse annotations, we run camera registration that propogates the rotation annotations using optical flow and monocular depth.
Camera translations are approximated with 2D object center and size (from segmentation) assuming a orthographic camera model. 


Parallelizing the pre-processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Preprocessing 10 videos takes about 90 minutes on a single device. To speed up the pre-processing, 
we can parallelize tasks over multiple gpus with the following::

  # Args: sequence name, text prompt for segmentation, category from {human, quad, other} for camera viewpoint, gpu id
  python scripts/run_preprocess.py cat-pikachu animal quad "0,1,2,3"


Visit other `tutorials </lab4d/tutorials/#content>`_.