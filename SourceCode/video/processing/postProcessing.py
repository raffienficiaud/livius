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


def get_min_max_boundaries_for_normalized_histogram(hist):
    """Gets the 1- and 99-percentile as an approximation of the boundaries
       of the histogram.

       Note: 
        The Histogram is expected to be normalized

       Returns both the min and the max value for the histogram
    """    
    t_min = 0
    t_max = 255

    min_mass = 0
    max_mass = 0

    # Integrate until we reach 1% of the mass from each direction
    while min_mass < 0.001:
        min_mass = min_mass + hist[t_min]
        t_min = t_min + 1

    while max_mass < 0.001:
        max_mass = max_mass + hist[t_max]
        t_max = t_max - 1

    return t_min, t_max

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

            # Get Histograms of the frame for each color channel
            hist_blue = cv2.calcHist([frame],[0],None,[256],[0,256])
            hist_green = cv2.calcHist([frame],[1],None,[256],[0,256])
            hist_red = cv2.calcHist([frame],[2],None,[256],[0,256])

            # Histogram for grayscale picture
            grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY);
            hist_gray = cv2.calcHist([grayscale], [0], None, [256], [0,256])

            # Normalize
            hist_blue = cv2.normalize(hist_blue)
            hist_green = cv2.normalize(hist_green)
            hist_red = cv2.normalize(hist_red)
            hist_gray = cv2.normalize(hist_gray)

            # Obtain the estimated boundaries for each color channel
            min_blue, max_blue = get_min_max_boundaries_for_normalized_histogram(hist_blue)
            min_green, max_green = get_min_max_boundaries_for_normalized_histogram(hist_green)
            min_red, max_red = get_min_max_boundaries_for_normalized_histogram(hist_red)

            min_val = min(min_blue, min_green, min_red)
            max_val = max(max_blue, max_green, max_red) 

            print "Min Frame :", frame[:,:,0].min()
            print "Min Frame :", frame[:,:,1].min()
            print "Min Frame :", frame[:,:,2].min()

            print "min and max of each channel boundary:", min_val, max_val

            framecorrected = 255.0 * (np.maximum(frame.astype(float32) - min_val, 0))
            # framecorrected = 255.0 * (frame.astype(float32) - min_val)
            framecorrected = framecorrected / (max_val - min_val)

            framecorrected = np.minimum(framecorrected, 255.0)

            framecorrected = framecorrected.astype(uint8)


            # This tries out using the grayscale image

            min_val, max_val = get_min_max_boundaries_for_normalized_histogram(hist_gray)


            print "Min Frame :", grayscale.min()
            print "grayscale", min_val, max_val

            compare = 255.0 * (np.maximum(frame.astype(float32) - min_val, np.zeros(frame.shape)))
            compare = compare / (max_val - min_val)
            compare = np.minimum(compare, 255.0)
            compare = compare.astype(uint8)


            # This shows the current picture and the color correction in a window
            fig = figure()

            fig.add_subplot(3,1,1)
            plt.imshow(frame)     
            plt.title("Original Frame")       

            fig.add_subplot(3,1,2)            
            plt.imshow(compare)
            plt.title("Color correction using the 1 and 99 percentile of the grayscale histogram")

            fig.add_subplot(3,1,3)
            plt.imshow(framecorrected)
            plt.title("Color correction using min and max of all color channel percentile boundaries")


            plt.show()            
            # cv2.destroyAllWindows()

            # framecorrected = 255.0 * (frame.astype(float32) - min_val) / (max_val - min_val)
            # framecorrected = np.empty(frame.shape, dtype=float32)
            # framecorrected[:,:,red] = 255.0 * (np.maximum(frame[:,:,red].astype(float32) - min_red, np.zeros(frame.shape[:2]))) / (max_red - min_red)   
            # framecorrected[:,:,green] = 255.0 * (np.maximum(frame[:,:,green].astype(float32) - min_green, np.zeros(frame.shape[:2]))) / (max_green - min_green)
            # framecorrected[:,:,blue] = 255.0 * (np.maximum(frame[:,:,blue].astype(float32) - min_blue, np.zeros(frame.shape[:2]))) / (max_blue - min_blue)
            


            # Save Histogram for each color channel
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

            # plt.plot(hist_gray)
            # plt.xlim([0,256])
            # plt.savefig('histograms/histogram_gray' + str(hist_count) + '.png')
            # plt.clf()

            hist_count += 1

        frame_count += 1

