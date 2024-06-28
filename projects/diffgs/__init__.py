import sys, os

sys.path.insert(0, os.getcwd())
from projects.diffgs import gs_model 

from lab4d.utils.profile_utils import decorate_module
decorate_module(gs_model)