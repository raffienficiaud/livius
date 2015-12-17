# Import Basic modules

import numpy as np
import os 

# Import everything needed to edit video clips
from moviepy.editor import *
from moviepy.Clip import *
#from moviepy.video.tools.cuts import FramesMatches
#from moviepy.video.fx.crop import crop
from moviepy.video.VideoClip import *
from moviepy.config import get_setting # ffmpeg, ffmpeg.exe, etc...

#Importing ploting libraries for shoeing purposes
import matplotlib.pyplot as plt
from pylab import *

# Importing Opencv
import cv2