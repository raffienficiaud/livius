# %debug

"""
This module is a collection of small Python functions and classes which make common patterns shorter and easier.
It is by no means a complete collection but you can keep extending it.
If you need any other convenient utilities, you would add them inside this module.
"""

import numpy as np
import os
import sys
import json
from moviepy.editor import *
from moviepy.Clip import *
from moviepy.video.VideoClip import *
import matplotlib.pyplot as plt
from pylab import *
import cv2
from functools import wraps


def linear_interpolation(x, x0, x1, y0, y1):
    """Linearly interpolates x between the two points (x0, y0) and (x1, y1)"""
    return y0 + float(y1 - y0) * (float(x - x0) / float(x1 - x0))


def get_transformation_points_from_normalized_rect(rect, image):
    """
    Returns a 4x2 Numpy Array containg the points in the order:
    [TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT ]

    :param rect: A normalized rectangle specified by [x,y,width,height]
    :param image: The image (specifies the scaling for the transformation points)
    """
    x,y,width,height = rect
    image_height, image_width = image.shape[:2]

    minx = x * image_width
    miny = y * image_height
    maxx = (x + width) * image_width
    maxy = (y + height) * image_height

    return np.array([[minx, maxy],
                     [maxx, maxy],
                     [maxx, miny],
                     [minx, miny]],
                    dtype=np.float32)


def get_polygon_outer_bounding_box(polygon):
    """Get the outer bounding box of a polygon defined as a sequence of
    points (2-uples) in the 2D plane.

    :param polygon: (x1,y1), ... (xn,yn)
    :return: (x,y,width,height)
    """

    xs = [i[0] for i in polygon]  # may be improved if using numpy methods directly
    ys = [i[1] for i in polygon]

    mx = min(xs)
    my = min(ys)
    return mx, my, max(xs) - mx, max(ys) - my


def crop_image_from_normalized_coordinates(im, rect):
    """
    Returns a cropped image from `rect` where the coordinates are normalized.
    The form of `rect` is similar to :py:`get_polygon_outer_bounding_box`.
    """
    import math

    minx, miny, rect_width, rect_height = rect
    im_height, im_width = im.shape[:2]

    x1, x2 = int(math.floor(minx * im_width)), int(math.ceil((minx + rect_width) * im_width))
    y1, y2 = int(math.floor(miny * im_height)), int(math.ceil((miny + rect_height) * im_height))

    if len(im.shape) == 2:
        return im[y1:y2, x1:x2]
    elif len(im.shape) == 3:
        return im[y1:y2, x1:x2, :]
    else:
        raise RuntimeError('Image does not have grayscale or color format')


def prompt_yes_no_terminal(question, default="yes"):

    """
    It asks a yes/no question via raw_input() and return user answer.
    Inputs:

        question: is a string that is presented to the user.
        default: is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    outputs:
        The "answer" return value is True for "yes" or False for "no".
    """

    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("[tools] Invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def rectify_coordinates(oldCoordinates):

    """
    This function serves to rectify the coordinate of the detected slide area
    to the order like: [TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT ]

    Inputs:

        oldCoordinates: numpy array with 8 elements, normally (4,2)

    outputs:

        newCoordinates: numpy array with below order
        [TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT ]
    """

    if (oldCoordinates.size == 8):
        oldCoordinates = oldCoordinates.reshape((4, 2))
        newCoordinates = np.zeros((4, 2), dtype=np.float32)

        add = oldCoordinates.sum(1)
        newCoordinates[0] = oldCoordinates[np.argmin(add)]
        newCoordinates[2] = oldCoordinates[np.argmax(add)]

        diff = np.diff(oldCoordinates, axis=1)
        newCoordinates[1] = oldCoordinates[np.argmin(diff)]
        newCoordinates[3] = oldCoordinates[np.argmax(diff)]
        return newCoordinates
    else:
        sys.exit("[tools] Error: You have selected less than 4 points. You must choose four coordinates.")


def video_duration_shrink (fullVideoPath, tStart, tEnd, writeFlie=False):

    """
    This function serves to shrink the video duration and
    cuts the clip between two times.
    Inputs:

        fullVideoPath: The corresponding path of the desired video
        tStart: The starting time to cut the video from that (default = 0)
        tEnd: The ending time to finish the video untill that (default = end of video)

        Hint:
        Times can be represented either in seconds (tStart=230.54),
        as a couple (minutes, seconds) (tStart=(3,50.54)), as a triplet (hour, min, sec)
        (tStart=(0,3,50.54)) or as a string (tStart='00:03:50.54')).

    Outputs:
        shrinkedVideo: The shrinked version of the input video between tStart and tEnd.

    Options:
        writeFlie: A flag to set whether write the new video or not (default = False)
        you may simply change the codecs, fps and etc inside the function.
        default codec: libx264
        default container: mp4

    Example:
        video_duration_shrink (targetVideo, tStart=(40,50.0), tEnd=(45,0.0), writeFlie = True)

    """

    # Read the main video
    mainVideo = VideoFileClip(fullVideoPath)
    newVideo = mainVideo.subclip(tStart, tEnd)

    if writeFlie:
        newVideo.write_videofile("shrinkedVideo.mp4", fps=mainVideo.fps, codec='libx264')




def sum_all_differences_frames(fullPathToVideoFile, marginToReadFrames, flagShow):


    """
    This function can be served for getting the sum of the difference of the pixel value of selected frames.
    After creating an instance of the class and call the corresponding method,
    a window will be open to get the user for points and then return the points back.
    Inputs:
        fullPathToVideoFile: The corresponding path to the video file
        marginToReadFrames: The amount of seconds from the beginning and end of the video
                                to read the frames and calculate the sum of them.
        flagShow: If true, the final summed frame has been shown.

    Outputs:
        finalDiffSumNorm: The final 2D-image is the sum of difference of all frames between
                         (video.duration)-marginToReadFrames - marginToReadFrames).

    Example:
        sum_all_differences_frames(inputVideo, marginToReadFrames=20, flagShow=True)

    """

    # Reading the video file
    video = VideoFileClip(fullPathToVideoFile, audio=False)
    W, H = video.size

    finalDiffSum = np.zeros((H, W), dtype=float)
    counter = 0
    for t in frange(marginToReadFrames, int(video.duration) - marginToReadFrames, (1 / video.fps)):

        # Getting the frame
        firstSlide = cv2.cvtColor(video.get_frame(t), cv2.COLOR_RGB2GRAY)
        firstSlide = firstSlide.astype('int16')

        # Getting the frame
        secondSlide = cv2.cvtColor(video.get_frame(t + (1 / video.fps)), cv2.COLOR_RGB2GRAY)
        secondSlide = secondSlide.astype('int16')


        dif = secondSlide - firstSlide
        finalDiffSum = finalDiffSum + dif
        counter = counter + 1

    finalDiffSumNorm = finalDiffSum / counter

    if flagShow:
        plt.imshow(finalDiffSumNorm, cmap=cm.Greys_r)
        plt.show()

    return finalDiffSumNorm





def max_all_differences_frames(fullPathToVideoFile, marginToReadFrames, flagShow):

    """
    This function can be served for getting the max of the difference of the pixel value of selected frames.
    After creating an instance of the class and call the corresponding method,
    a window will be open to get the user for points and then return the points back.

    Inputs:
        fullPathToVideoFile: The corresponding path to the video file
        marginToReadFrames: The amount of seconds from the beginning and end of the video
                                to read the frames and calculate the sum of them.
        flagShow: If true, the final summed frame has been shown.

    Outputs:
        finalDiffMaxNorm: The final 2D-image is the max of the difference of all frames between
                         (video.duration)-marginToReadFrames - marginToReadFrames).

    Example:
        max_all_differences_frames(inputVideo, marginToReadFrames=20, flagShow=True)
    """

    # Reading the video file
    video = VideoFileClip(fullPathToVideoFile, audio=False)
    W, H = video.size

    finalDiffMax = np.zeros((H, W), dtype=float)
    counter = 0
    for t in frange(marginToReadFrames, int(video.duration) - marginToReadFrames, (1 / video.fps)):

        # Getting the frame
        firstSlide = cv2.cvtColor(video.get_frame(t), cv2.COLOR_RGB2GRAY)
        firstSlide = firstSlide.astype('int16')

        # Getting the frame
        secondSlide = cv2.cvtColor(video.get_frame(t + (1 / video.fps)), cv2.COLOR_RGB2GRAY)
        secondSlide = secondSlide.astype('int16')


        dif = secondSlide - firstSlide
        finalDiffMax = np.maximum(finalDiffMax , dif)
        counter = counter + 1

    finalDiffMaxNorm = finalDiffMax / counter

    if flagShow:
        plt.imshow(finalDiffMaxNorm, cmap=cm.Greys_r)
        plt.show()

    return finalDiffMaxNorm





def max_all_frames(fullPathToVideoFile, marginToReadFrames, flagShow):

    """ This function can be served for getting the max pixel value of selected frames.
    After creating an instance of the class and call the corresponding method,
    a window will be open to get the user for points and then return the points back.

    Inputs:
        fullPathToVideoFile: The corresponding path to the video file
        marginToReadFrames: The amount of seconds from the beginning and end of the video
                                to read the frames and calculate the sum of them.
        flagShow: If true, the final summed frame has been shown.

    Outputs:
        finalMaxNorm: The final 2D-image is the max of all frames between
                         (video.duration)-marginToReadFrames - marginToReadFrames).

    Example:
        max_all_frames(inputVideo, marginToReadFrames=20, flagShow=True)

    """

    # Reading the video file
    video = VideoFileClip(fullPathToVideoFile, audio=False)
    W, H = video.size

    finalMax = np.zeros((H, W), dtype=float)
    finalMax1 = np.zeros((H, W), dtype=float)
    counter = 0
    for t in frange(marginToReadFrames, int(video.duration) - marginToReadFrames, (1 / video.fps)):

        # Getting the frame
        firstSlide = cv2.cvtColor(video.get_frame(t), cv2.COLOR_RGB2GRAY)
        firstSlide = firstSlide.astype('int16')

        # Getting the frame
        secondSlide = cv2.cvtColor(video.get_frame(t + (1 / video.fps)), cv2.COLOR_RGB2GRAY)
        secondSlide = secondSlide.astype('int16')


        finalMax1 = np.maximum(secondSlide , firstSlide)
        finalMax = np.maximum(finalMax1 , finalMax)
        counter = counter + 1

    finalMaxNorm = finalMax / counter

    if flagShow:
        plt.imshow(finalMaxNorm, cmap=cm.Greys_r)
        plt.show()

    return finalMaxNorm

def read_histogram_correlations_and_boundaries_from_json_file(filepath, slide_stripe=0):
    """Reads a json file containing the information about the histogram correlations and the
       histogram boundaries. The information is computed for several stripes.

       slide_stripe is the stripe we assume the slides of the talk to be in.

       Returns all histogram correlations and boundaries for the desired stripe

       Note:
       The json file/object must have the following structure:

       @todo(Stephan): update the description of the json file
       {
        "frame_id": {
            "dist_stripes": {"0": , ..., "N_stripes - 1": },
            "boundaries": {"0": {"min": , "max":}, ..., "N_stripes - 1": {"min": , "max": }}
        }
       }
    """
    with open(filepath) as f:
        distances_histogram = json.load(f)

    frame_indices = [(i, int(i)) for i in distances_histogram.keys()]
    frame_indices.sort(key=lambda x: x[1])

    histogram_correlations_dict = {}
    histogram_boundaries_dict = {}
    for count, count_integer in frame_indices:
        current_correlations = distances_histogram[count]['dist_stripes']
        current_boundaries = distances_histogram[count]['boundaries']
        for i in current_correlations.keys():

            if not histogram_correlations_dict.has_key(int(i)):
                histogram_correlations_dict[int(i)] = []

            if not histogram_boundaries_dict.has_key(int(i)):
                histogram_boundaries_dict[int(i)] = []

            histogram_correlations_dict[int(i)].append(float(current_correlations[i]))

            # @todo(Stephan): We should probably only need int here instead of float
            current_min_max_boundaries = (float(current_boundaries[i]['min']), float(current_boundaries[i]['max']))

            histogram_boundaries_dict[int(i)].append(current_min_max_boundaries)

    # N_stripes = max(histogram_correlations_dict.keys())

    return histogram_correlations_dict[slide_stripe], histogram_boundaries_dict[slide_stripe], frame_indices


class CallbacksPoints:
    """
    This class implemented for call back points and event-process.
    It is called in slideDetection module for getting user selected points via mouse.
    """


    def __init__(self, image_size):
        self.start_point = []
        self.end_point = []
        self.image_size = image_size

        self.points = []

        self.next_point = 0
        self.const_lines = []

        self.h = 0
        self.w = 0
        self.deg = 0

        self.lx = []
        self.ly = []

    def callback_press(self, event):
        if self.next_point > 3:
            return

        self.next_point += 1
        self.points.append([event.xdata, event.ydata])

        if self.next_point > 1:
            self.draw_lines(event, redraw=True)

    def draw_lines(self, event, redraw=False):
        assert(len(self.points) > 1)

        ax = event.inaxes

        for x in range(len(self.points) - 1):
            # draw line from self.points[x] to self.points[x+1]
            p0 = self.points[x]
            p1 = self.points[x + 1]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r', linewidth=1)

        if redraw:
            ax.set_xlim(0, self.image_size[1])
            ax.set_ylim(self.image_size[0], 0)
            ax.figure.canvas.draw()


    def callback_motion(self, event):

        if not event.inaxes: return

        x1 = event.xdata
        y1 = event.ydata

        ax = event.inaxes
        ax.lines = []


        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        if (len(self.points) > 0) and (len(self.points) <= 3):

            x0 = self.points[-1][0]
            y0 = self.points[-1][1]

            ax.plot([x0, x1], [y0, y1], 'r', linewidth=1)

            ax.set_xlim(0, self.image_size[1])
            ax.set_ylim(self.image_size[0], 0)

        if len(self.points) > 1:
            self.draw_lines(event)

        self.lx.set_ydata(y1)
        self.ly.set_xdata(x1)

        ax.figure.canvas.draw()


    def connect(self, fig):
        fig.canvas.mpl_connect('button_press_event', self.callback_press)
        fig.canvas.mpl_connect('motion_notify_event', self.callback_motion)

    def get_x0(self):
        return self.points[0][0]
    def get_y0(self):
        return self.points[0][1]
    def get_rect(self):
        return (self.points[0][0],  # x0
                self.points[0][1],  # y0
                self.w,
                self.h,
                self.deg_grad)

    def get_points(self):
        if self.next_point > 3:
            return self.points
        else:
            return []



if __name__ == '__main__':


    targetVideo = "/media/pbahar/Data Raid/Videos/18.03.2015/video2.mp4"
    t = "/media/pbahar/Data Raid/Videos/18.05.2015/ProfSeidel.mov"
    video_duration_shrink (t, tStart=(30, 0.0), tEnd=(32, 0.0), writeFlie=True)
