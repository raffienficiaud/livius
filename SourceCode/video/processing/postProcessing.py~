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



def perspective_transformation2D(img, coordinates, desiredScreenLayout=(1280,960)):

    slideShow = np.array([[0,0],[desiredScreenLayout[0]-1,0],[desiredScreenLayout[0]-1,desiredScreenLayout[1]-1],\
                        [0,desiredScreenLayout[1]-1]],np.float32)
    retval = cv2.getPerspectiveTransform(coordinates,slideShow)
    warp = cv2.warpPerspective(img,retval,desiredScreenLayout)
    return warp


def transformation3D(clip, coordinates, desiredScreenLayout=(1280,960)):
    def new_tranformation(frame):
        return perspective_transformation2D(frame, coordinates, desiredScreenLayout)
    return clip.fl_image(new_tranformation)
