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


def get_histogram_min_max(hist):
    """Gets the 1- and 99-percentile 
    """    


def get_histograms(clip):
    hist_dir = os.path.join(os.getcwd(), 'histograms')

    if not os.path.exists(hist_dir):    
        os.makedirs(hist_dir)

    fps = clip.fps
    frame_count = 0
    hist_count = 0

    for frame in clip.iter_frames():
        if frame_count % fps == 0:
            frame_count = 0

            min_val = 50.0
            max_val = 200.0



            framecorrected = 255.0 * (np.maximum(frame.astype(float32) - min_val, np.zeros(frame.shape))) / (max_val - min_val)

            framecorrected = framecorrected.astype(uint8)

            # This shows the current picture in a window
            fig = figure()
            fig.add_subplot(2,1,0)            
            plt.imshow(frame)

            fig.add_subplot(2,1,1)
            plt.imshow(framecorrected)

            plt.show()            
            # cv2.destroyAllWindows()

            # Histogram for each color channel
            hist_blue = cv2.calcHist([frame],[0],None,[256],[0,256])
            hist_green = cv2.calcHist([frame],[1],None,[256],[0,256])
            hist_red = cv2.calcHist([frame],[2],None,[256],[0,256])

            np.save('histograms/histogram_blue' + str(hist_count), hist_blue)
            np.save('histograms/histogram_green' + str(hist_count), hist_green)
            np.save('histograms/histogram_red' + str(hist_count), hist_red)


            # Plot and save as png

            # plt.plot(hist_blue, color='b')
            # plt.plot(hist_green, color='g')
            # plt.plot(hist_red, color='r')
            # plt.xlim([0,256])

            # plt.savefig('histograms/histogram' + str(hist_count) + '.png')
            # plt.clf()

            # # Histogram for grayscale picture
            # grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY);
            # hist_gray = cv2.calcHist([grayscale], [0], None, [256], [0,256])

            # plt.plot(hist_gray)
            # plt.xlim([0,256])
            # plt.savefig('histograms/histogram_gray' + str(hist_count) + '.png')
            # plt.clf()

            hist_count += 1

        frame_count += 1

