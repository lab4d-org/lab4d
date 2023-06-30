Get Started
===================
 
Requirements
-------------------------

- **Linux** machine with at least 1 GPU (we tested on 3090s)
- **Conda**

  - Follow `this link <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#installing-on-linux>`_ to install conda.

  - Recommended: use mamba for package management (more efficient than conda). Install mamba with::
    
      conda install -c conda-forge mamba -y

- For developers: use `VS Code <https://code.visualstudio.com/>`_ with Black Formatter.

Set up the environment
-------------------------

Clone the repository and create a conda environment with the required packages::

    git clone git@github.com:gengshan-y/lab4d.git --recursive

    cd lab4d

    mamba env create -f environment.yml

    conda activate lab4d

    bash scripts/install-deps.sh


Running the Tutorial Code
---------------------------------------------
See the `Tutorials page </lab4d/tutorials>`_.


.. .. Lab4D documentation master file, created by
..    sphinx-quickstart on Fri Jun  2 20:54:08 2023.
..    You can adapt this file completely to your liking, but it should at least
..    contain the root `toctree` directive.

.. Welcome to Lab4D's DOCUMENTATION!
.. =================================

.. .. toctree::
..    :maxdepth: 2

..    get_started

.. .. Indices and tables
.. .. ==================

.. .. * :ref:`genindex`
.. .. * :ref:`modindex`
.. .. * :ref:`search`
