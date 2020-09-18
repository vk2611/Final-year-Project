import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf

from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

import cv2
cap = cv2.VideoCapture(0)

# ## Env setup

from utils import label_map_util

from utils import visualization_utils as vis_util


