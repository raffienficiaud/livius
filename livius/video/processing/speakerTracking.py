

"""
This module implements various algorithms based on CMT for speaker tracking.
The implementations are based on http://www.gnebehay.com/cmt/ and for more details,
the readers will be reffered to the following paper.
http://www.gnebehay.com/publications/wacv_2014/wacv_2014.pdf

This algorithm has been combined with Kalman filter, and
different approaches have been used to ease the computation time.

For every algorithm, one class has been defined in which there is a method named "speakerTracker" to call.
Different classes have different attributes, but in order to call the corresponding
function "speakerTracker", all classes are the same.




classes:
- CMT_algorithm
    --> to run the CMT only
- CMT_algorithm_kalman_filter
    --> to run the CMT, predict, update and smooth the results by kalman filter
- CMT_algorithm_kalman_filter_stripe
    --> to run the CMT, predict, update and smooth the results by kalman filter (in a stripe line including the speaker)
- CMT_algorithm_kalman_filter_downsample
    --> to run the CMT, predict, update and smooth the results by kalman filter (downsampling the frames to ease the computational time)
- CMT_algorithm_kalman_filter_vertical_mean
    --> to run the CMT, predict, update and smooth the results by kalman filter (no vertical coordinate estimation --
    we assume, the speaker wont jump during the lecture)
- CMT_algorithm_kalman_filter_neighboring
    --> to run the CMT, predict, update and smooth the results by kalman filter (keypoint calculation is only for neighboring window)
- FOV_specification
    --> to get a certain amount of field of view including the speaker
"""

import os
import cv2
import exceptions
import json
import glob
import time
import logging
import traceback, sys, code

from numpy import empty, nan
import sys
import math
import CMT.CMT
import CMT.util as cmtutil
import numpy as np
from moviepy.editor import *
from pykalman import KalmanFilter
from functools import wraps
from moviepy.video.fx.crop import crop as moviepycrop
import matplotlib.pyplot as plt
from pylab import *



from util.histogram import get_histogram_min_max_with_percentile


# logging facility
FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# temporary path
_tmp_path = os.path.join('temp_path')
if not os.path.exists(_tmp_path):
    os.makedirs(_tmp_path)

debug = True


def counterFunction(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        return func(*args, **kwargs)
    tmp.count = 0
    return tmp


class CMT_algorithm():



    def __init__(self, inputPath, bBox=None , skip=None):
        self.inputPath = inputPath  # 'The input path.'
        self.bBox = bBox  # 'Specify initial bounding box.'
        self.skip = skip  # 'Skip the first n frames.'
        self.CMT = CMT.CMT.CMT()


    def speakerTracker(self):


        # Clean up
        cv2.destroyAllWindows()

        if self.inputPath is not None:

            # If a path to a file was given, assume it is a single video file
            if os.path.isfile(self.inputPath):
                cap = cv2.VideoCapture(self.inputPath)
                clip = VideoFileClip(self.inputPath, audio=False)
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTrackering] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_' + 'speakerCoordinates.txt'


                # Skip first frames if required
                if self.skip is not None:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip)

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

                # Skip first frames if required
                if self.skip is not None:
                    cap.frame = 1 + self.skip

        else:
            # If no input path was specified, open camera device
            sys.exit("[speakerTrackering] Error: no input path was specified")

        # Read first frame
        status, im0 = cap.read()
        imGray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        imDraw = np.copy(im0)

        if self.bBox is not None:
            # Try to disassemble user specified bounding box
            values = self.bBox.split(',')
            try:
                values = [int(v) for v in values]
            except:
                raise Exception('[speakerTrackering] Unable to parse bounding box')
            if len(values) != 4:
                raise Exception('[speakerTrackering] Bounding box must have exactly 4 elements')
            bbox = np.array(values)

            # Convert to point representation, adding singleton dimension
            bbox = cmtutil.bb2pts(bbox[None, :])

            # Squeeze
            bbox = bbox[0, :]

            tl = bbox[:2]
            br = bbox[2:4]
        else:
            # Get rectangle input from user
            (tl, br) = cmtutil.get_rect(imDraw)

        print '[speakerTrackering] Using', tl, br, 'as initial bounding box for the speaker'

        self.CMT.initialise(imGray0, tl, br)

        newClip = clip.fl_image(self.crop)

        return newClip


    def crop (self, frame):

        windowSize = (2 * 640, 2 * 360)
        newFrames = np.zeros((windowSize[0], windowSize[1], 3))
        imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.CMT.process_frame(imGray)

        if not (math.isnan(self.CMT.center[0]) or math.isnan(self.CMT.center[1])
                or (self.CMT.center[0] <= 0) or (self.CMT.center[1] <= 0)):

            x1 = np.floor(self.CMT.center[1] - windowSize[1] / 2)
            y1 = np.floor(self.CMT.center[0] - windowSize[0] / 2)
            x2 = np.floor(x1 + windowSize[1])
            y2 = np.floor(y1 + windowSize[0])

            # Corner correction (Height)
            if (x1 <= 0):
                x1 = 0
                x2 = np.floor(x1 + windowSize[1])
            if (x2 >= imGray.shape[0]):
                x2 = np.floor(imGray.shape[0])
                x1 = np.floor(x2 - windowSize[1])
            # Corner correction (Width)
            if (y1 <= 0):
                y1 = 0
                y2 = np.floor(y1 + windowSize[0])
            if (y2 >= imGray.shape[1]):
                y2 = np.floor(imGray.shape[1])
                y1 = np.floor(y2 - windowSize[0])
            newFrames = frame[x1:x2, y1:y2, :]

        # print 'Center: {0:.2f},{1:.2f}'.format(CMT.center[0], CMT.center[1])
        return newFrames



class CMT_algorithm_kalman_filter():



    def __init__(self, inputPath, skip=None):
        self.inputPath = inputPath  # 'The input path.'
        self.skip = skip  # 'Skip the first n frames.'
        self.frameCounter = 0
        self.numFrames = None
        self.CMT = CMT.CMT.CMT()


    def speakerTracker(self):


        # Clean up
        cv2.destroyAllWindows()

        if self.inputPath is not None:

            # If a path to a file was given, assume it is a single video file
            if os.path.isfile(self.inputPath):
                cap = cv2.VideoCapture(self.inputPath)
                clip = VideoFileClip(self.inputPath, audio=False)
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTracker] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_'



                # Skip first frames if required
                if self.skip is not None:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip)

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

                # Skip first frames if required
                if self.skip is not None:
                    cap.frame = 1 + self.skip

        else:
            # If no input path was specified, open camera device
            sys.exit("[speakerTracker] Error: no input path was specified")

        # Read first frame
        status, im0 = cap.read()
        imGray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)

        if debug:
            # speaker bounding box used for debugging
            (tl, br) = (1848, 840), (2136, 1116)
        else:
            imDraw = np.copy(im0)
            (tl, br) = cmtutil.get_rect(imDraw)

        print '[speakerTrackering] Using', tl, br, 'as initial bounding box for the speaker'

        self.CMT.initialise(imGray0, tl, br)

        measuredTrack = np.zeros((self.numFrames + 10, 2)) - 1

        count = 0

        while count <= self.numFrames:

            status, im = cap.read()
            if not status:
                break
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            logging.debug('[tracker] processing frame %d', count)

            self.CMT.process_frame(im_gray)

            # debug
            if debug:
                im_debug = np.copy(im)
                cmtutil.draw_keypoints(self.CMT.active_keypoints, im_debug, (0, 0, 255))
                cmtutil.draw_keypoints(self.CMT.tracked_keypoints, im_debug, (0, 255, 0))



            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , count)
            if not (math.isnan(self.CMT.center[0])
                    or math.isnan(self.CMT.center[1])
                    or (self.CMT.center[0] <= 0)
                    or (self.CMT.center[1] <= 0)):
                measuredTrack[count, 0] = self.CMT.center[0]
                measuredTrack[count, 1] = self.CMT.center[1]
            else:
                # take the previous estimate if none is found in the current frame
                measuredTrack[count, 0] = measuredTrack[count - 1, 0]
                measuredTrack[count, 1] = measuredTrack[count - 1, 1]

            if debug:
                cmtutil.draw_bounding_box((int(measuredTrack[count, 0] - 50), int(measuredTrack[count, 1] - 50)),
                                          (int(measuredTrack[count, 0] + 50), int(measuredTrack[count, 1] + 50)),
                                          im_debug)


                cv2.imwrite(os.path.join(_tmp_path, 'debug_file_%.6d.png' % count), im_debug)

                im_debug = np.copy(im)
                cmtutil.draw_keypoints([kp.pt for kp in self.CMT.keypoints_cv], im_debug, (0, 0, 255))
                cv2.imwrite(os.path.join(_tmp_path, 'all_keypoints_%.6d.png' % count), im_debug)

            count += 1

        numMeas = measuredTrack.shape[0]
        markedMeasure = np.ma.masked_less(measuredTrack, 0)

        # Kalman Filter Parameters
        deltaT = 1.0 / clip.fps
        transitionMatrix = [[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]]  # A
        observationMatrix = [[1, 0, 0, 0], [0, 1, 0, 0]]  # C

        xinit = markedMeasure[0, 0]
        yinit = markedMeasure[0, 1]
        vxinit = markedMeasure[1, 0] - markedMeasure[0, 0]
        vyinit = markedMeasure[1, 1] - markedMeasure[0, 1]
        initState = [xinit, yinit, vxinit, vyinit]  # mu0
        initCovariance = 1.0e-3 * np.eye(4)  # sigma0
        transistionCov = 1.0e-4 * np.eye(4)  # Q
        observationCov = 1.0e-1 * np.eye(2)  # R
        kf = KalmanFilter(transition_matrices=transitionMatrix,
            observation_matrices=observationMatrix,
            initial_state_mean=initState,
            initial_state_covariance=initCovariance,
            transition_covariance=transistionCov,
            observation_covariance=observationCov)

        self.measuredTrack = measuredTrack
        (self.filteredStateMeans, self.filteredStateCovariances) = kf.filter(markedMeasure)
        (self.filterStateMeanSmooth, self.filterStateCovariancesSmooth) = kf.smooth(markedMeasure)

        # np.savetxt((baseName + 'speakerCoordinates_CMT_Kalman.txt'),
                   # np.hstack((np.asarray(self.filteredStateMeans), np.asarray(self.filteredStateCovariances) ,
                   # np.asarray(self.filterStateMeanSmooth), np.asarray(self.filterStateCovariancesSmooth) )))
        # np.savetxt((baseName + 'speakerCoordinates_CMT.txt'), np.asarray(measuredTrack))

        newClip = clip.fl_image(self.crop)
        return newClip



    @counterFunction
    def crop (self, frame):

        self.frameCounter = self.crop.count
        # print self.frameCounter
        windowSize = (2 * 640, 2 * 360)
        newFrames = np.zeros((windowSize[0], windowSize[1], 3))

        if self.frameCounter <= self.numFrames:

            imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            x1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter) - 1][1] - windowSize[1] / 2)
            y1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter) - 1][0] - windowSize[0] / 2)
            x2 = np.floor(x1 + windowSize[1])
            y2 = np.floor(y1 + windowSize[0])

            # Corner correction (Height)
            if (x1 <= 0):
                x1 = 0
                x2 = np.floor(x1 + windowSize[1])
            if (x2 >= imGray.shape[0]):
                x2 = np.floor(imGray.shape[0])
                x1 = np.floor(x2 - windowSize[1])
            # Corner correction (Width)
            if (y1 <= 0):
                y1 = 0
                y2 = np.floor(y1 + windowSize[0])
            if (y2 >= imGray.shape[1]):
                y2 = np.floor(imGray.shape[1])
                y1 = np.floor(y2 - windowSize[0])
            # print x1, y1 , x2, y2
            newFrames = frame[x1:x2, y1:y2, :]

        return newFrames


class FOV_specification():



    def __init__(self, inputPath, skip=None):
        self.inputPath = inputPath  # 'The input path.'
        self.skip = skip  # 'Skip the first n frames.'
        self.CMT = CMT.CMT.CMT()


    def speakerTracker(self):


        # Clean up
        cv2.destroyAllWindows()

        if self.inputPath is not None:

            # If a path to a file was given, assume it is a single video file
            if os.path.isfile(self.inputPath):
                cap = cv2.VideoCapture(self.inputPath)
                clip = VideoFileClip(self.inputPath, audio=False)
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTrackering] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_' + 'speakerCoordinates.txt'


                # Skip first frames if required
                if self.skip is not None:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip)

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

                # Skip first frames if required
                if self.skip is not None:
                    cap.frame = 1 + self.skip

        else:
            # If no input path was specified, open camera device
            sys.exit("[speakerTrackering] Error: no input path was specified")

        # Read first frame
        status, im0 = cap.read()
        imGray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        imDraw = np.copy(im0)

        (tl, br) = cmtutil.get_rect(imDraw)

        print '[speakerTrackering] Using', tl, br, 'as initial bounding box for the speaker'

        # Get the points to crop the video with 4:3 aspect ratio
        # If the selected points are near to the border of the image, It will automaticaly croped till borders.
        x1 = np.floor(tl[0])
        y1 = np.floor(tl[1])
        x2 = np.floor(br[0])
        y2 = np.floor(tl[1] + np.abs((br[0] - tl[0]) * (3.0 / 4.0)))

        print x1, x2, y1, y2
        croppedClip = moviepycrop(clip, x1, y1, x2, y2)

        return croppedClip


class CMT_algorithm_kalman_filter_stripe():



    def __init__(self, inputPath, skip=None):
        self.inputPath = inputPath  # 'The input path.'
        self.frameCounter = 0
        self.numFrames = None
        self.CMT = CMT.CMT.CMT()


    def speakerTracker(self):


        # Clean up
        cv2.destroyAllWindows()

        if self.inputPath is not None:

            # If a path to a file was given, assume it is a single video file
            if os.path.isfile(self.inputPath):
                cap = cv2.VideoCapture(self.inputPath)
                clip = VideoFileClip(self.inputPath, audio=False)
                W, H = clip.size
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTracker] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_'

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

        else:
            # If no input path was specified, open camera device
            sys.exit("[speakerTracker] Error: no input path was specified")

        # Read first frame
        status, im0 = cap.read()
        imDraw = np.copy(im0)

        (tl, br) = cmtutil.get_rect(imDraw)

        print '[speakerTrackering] Using', tl, br, 'as initial bounding box for the speaker'
        x1 = 1
        x2 = W
        y1 = tl[1] - 100
        y2 = br[1] + 100

        imGray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        croppedGray0 = imGray0[y1:y2, x1:x2]
        self.CMT.initialise(croppedGray0, tl, br)

        measuredTrack = np.zeros((self.numFrames + 10, 2)) - 1


        count = 0
        for frame in clip.iter_frames():
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            im_gray = grayFrame[y1:y2, x1:x2]

            plt.imshow(im_gray, cmap=cm.Greys_r)
            plt.show()

            self.CMT.process_frame(im_gray)

            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , count)
            if not (math.isnan(self.CMT.center[0]) or math.isnan(self.CMT.center[1])
                or (self.CMT.center[0] <= 0) or (self.CMT.center[1] <= 0)):
                measuredTrack[count, 0] = self.CMT.center[0]
                measuredTrack[count, 1] = self.CMT.center[1]
            count += 1


        numMeas = measuredTrack.shape[0]
        markedMeasure = np.ma.masked_less(measuredTrack, 0)

        # Kalman Filter Parameters
        deltaT = 1.0 / clip.fps
        transitionMatrix = [[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]]  # A
        observationMatrix = [[1, 0, 0, 0], [0, 1, 0, 0]]  # C

        xinit = markedMeasure[0, 0]
        yinit = markedMeasure[0, 1]
        vxinit = markedMeasure[1, 0] - markedMeasure[0, 0]
        vyinit = markedMeasure[1, 1] - markedMeasure[0, 1]
        initState = [xinit, yinit, vxinit, vyinit]  # mu0
        initCovariance = 1.0e-3 * np.eye(4)  # sigma0
        transistionCov = 1.0e-4 * np.eye(4)  # Q
        observationCov = 1.0e-1 * np.eye(2)  # R
        kf = KalmanFilter(transition_matrices=transitionMatrix,
            observation_matrices=observationMatrix,
            initial_state_mean=initState,
            initial_state_covariance=initCovariance,
            transition_covariance=transistionCov,
            observation_covariance=observationCov)

        self.measuredTrack = measuredTrack
        (self.filteredStateMeans, self.filteredStateCovariances) = kf.filter(markedMeasure)
        (self.filterStateMeanSmooth, self.filterStateCovariancesSmooth) = kf.smooth(markedMeasure)


        newClip = clip.fl_image(self.crop)
        return newClip



    @counterFunction
    def crop (self, frame):

        self.frameCounter = self.crop.count
        # print self.frameCounter
        windowSize = (2 * 640, 2 * 360)
        newFrames = np.zeros((windowSize[0], windowSize[1], 3))

        if self.frameCounter <= self.numFrames:

            imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            x1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter) - 1][1] - windowSize[1] / 2)
            y1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter) - 1][0] - windowSize[0] / 2)
            x2 = np.floor(x1 + windowSize[1])
            y2 = np.floor(y1 + windowSize[0])

            # Corner correction (Height)
            if (x1 <= 0):
                x1 = 0
                x2 = np.floor(x1 + windowSize[1])
            if (x2 >= imGray.shape[0]):
                x2 = np.floor(imGray.shape[0])
                x1 = np.floor(x2 - windowSize[1])
            # Corner correction (Width)
            if (y1 <= 0):
                y1 = 0
                y2 = np.floor(y1 + windowSize[0])
            if (y2 >= imGray.shape[1]):
                y2 = np.floor(imGray.shape[1])
                y1 = np.floor(y2 - windowSize[0])
            # print x1, y1 , x2, y2
            newFrames = frame[x1:x2, y1:y2, :]

        return newFrames



class CMT_algorithm_kalman_filter_downsample():



    def __init__(self, inputPath, resizeFactor=0.5, skip=None):
        self.inputPath = inputPath  # 'The input path.'
        self.frameCounter = 0
        self.numFrames = None
        self.CMT = CMT.CMT.CMT()
        self.resizeFactor = resizeFactor


    def speakerTracker(self):


        # Clean up
        cv2.destroyAllWindows()

        if self.inputPath is not None:

            # If a path to a file was given, assume it is a single video file
            if os.path.isfile(self.inputPath):
                cap = cv2.VideoCapture(self.inputPath)
                clip = VideoFileClip(self.inputPath, audio=False)
                W, H = clip.size
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTracker] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_'

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

        else:
            # If no input path was specified, open camera device
            sys.exit("[speakerTracker] Error: no input path was specified")

        # Read first frame
        status, im0 = cap.read()
        imGray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        imResized = cv2.resize(imGray0, (0, 0), fx=self.resizeFactor, fy=self.resizeFactor)
        imDraw = np.copy(imResized)

        (tl, br) = cmtutil.get_rect(imDraw)

        print '[speakerTrackering] Using', tl, br, 'as initial bounding box for the speaker'

        self.CMT.initialise(imResized, tl, br)

        measuredTrack = np.zeros((self.numFrames + 10, 2)) - 1

        count = 0

        while count <= self.numFrames:

            status, im = cap.read()
            if not status:
                break
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_resized = cv2.resize(im_gray, (0, 0), fx=self.resizeFactor, fy=self.resizeFactor)

            tic = time.time()
            self.CMT.process_frame(im_resized)
            toc = time.time()
            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , count)
            # print 1000*(toc-tic)
            if not (math.isnan(self.CMT.center[0]) or math.isnan(self.CMT.center[1])
                or (self.CMT.center[0] <= 0) or (self.CMT.center[1] <= 0)):
                measuredTrack[count, 0] = self.CMT.center[0]
                measuredTrack[count, 1] = self.CMT.center[1]
            count += 1


        numMeas = measuredTrack.shape[0]
        markedMeasure = np.ma.masked_less(measuredTrack, 0)

        # Kalman Filter Parameters
        deltaT = 1.0 / clip.fps
        transitionMatrix = [[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]]  # A
        observationMatrix = [[1, 0, 0, 0], [0, 1, 0, 0]]  # C

        xinit = markedMeasure[0, 0]
        yinit = markedMeasure[0, 1]
        vxinit = markedMeasure[1, 0] - markedMeasure[0, 0]
        vyinit = markedMeasure[1, 1] - markedMeasure[0, 1]
        initState = [xinit, yinit, vxinit, vyinit]  # mu0
        initCovariance = 1.0e-3 * np.eye(4)  # sigma0
        transistionCov = 1.0e-4 * np.eye(4)  # Q
        observationCov = 1.0e-1 * np.eye(2)  # R
        kf = KalmanFilter(transition_matrices=transitionMatrix,
            observation_matrices=observationMatrix,
            initial_state_mean=initState,
            initial_state_covariance=initCovariance,
            transition_covariance=transistionCov,
            observation_covariance=observationCov)

        self.measuredTrack = measuredTrack
        (self.filteredStateMeans, self.filteredStateCovariances) = kf.filter(markedMeasure)
        (self.filterStateMeanSmooth, self.filterStateCovariancesSmooth) = kf.smooth(markedMeasure)

        # np.savetxt((baseName + 'speakerCoordinates_CMT_Kalman.txt'),
                   # np.hstack((np.asarray(self.filteredStateMeans), np.asarray(self.filteredStateCovariances) ,
                   # np.asarray(self.filterStateMeanSmooth), np.asarray(self.filterStateCovariancesSmooth) )))
        # np.savetxt((baseName + 'speakerCoordinates_CMT.txt'), np.asarray(measuredTrack))

        newClip = clip.fl_image(self.crop)
        return newClip



    @counterFunction
    def crop (self, frame):

        self.frameCounter = self.crop.count
        # print self.frameCounter
        windowSize = (2 * 640, 2 * 360)
        newFrames = np.zeros((windowSize[0], windowSize[1], 3))

        if self.frameCounter <= self.numFrames:

            imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            x1 = np.floor((self.filterStateMeanSmooth[(self.frameCounter) - 1][1]) * (1.0 / self.resizeFactor) - windowSize[1] / 2)
            y1 = np.floor((self.filterStateMeanSmooth[(self.frameCounter) - 1][0]) * (1.0 / self.resizeFactor) - windowSize[0] / 2)
            x2 = np.floor(x1 + windowSize[1])
            y2 = np.floor(y1 + windowSize[0])

            # Corner correction (Height)
            if (x1 <= 0):
                x1 = 0
                x2 = np.floor(x1 + windowSize[1])
            if (x2 >= imGray.shape[0]):
                x2 = np.floor(imGray.shape[0])
                x1 = np.floor(x2 - windowSize[1])
            # Corner correction (Width)
            if (y1 <= 0):
                y1 = 0
                y2 = np.floor(y1 + windowSize[0])
            if (y2 >= imGray.shape[1]):
                y2 = np.floor(imGray.shape[1])
                y1 = np.floor(y2 - windowSize[0])
            # print x1, y1 , x2, y2
            newFrames = frame[x1:x2, y1:y2, :]

        return newFrames


class CMT_algorithm_kalman_filter_vertical_mean():



    def __init__(self, inputPath, skip=None):
        self.inputPath = inputPath  # 'The input path.'
        self.frameCounter = 0
        self.numFrames = None
        self.CMT = CMT.CMT.CMT()


    def speakerTracker(self):


        # Clean up
        cv2.destroyAllWindows()

        if self.inputPath is not None:

            # If a path to a file was given, assume it is a single video file
            if os.path.isfile(self.inputPath):
                cap = cv2.VideoCapture(self.inputPath)
                clip = VideoFileClip(self.inputPath, audio=False)
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTracker] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_'

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

        else:
            # If no input path was specified, open camera device
            sys.exit("[speakerTracker] Error: no input path was specified")

        # Read first frame
        status, im0 = cap.read()
        imGray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        imDraw = np.copy(im0)

        (tl, br) = cmtutil.get_rect(imDraw)

        print '[speakerTrackering] Using', tl, br, 'as initial bounding box for the speaker'

        self.CMT.initialise(imGray0, tl, br)
        # self.inity = tl[1] - self.CMT.center_to_tl[1]
        measuredTrack = np.zeros((self.numFrames + 10, 2)) - 1


        count = 0
        for frame in clip.iter_frames():
            im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.CMT.process_frame(im_gray)

            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , count)
            # print self.inity
            if not (math.isnan(self.CMT.center[0]) or (self.CMT.center[0] <= 0)):
                measuredTrack[count, 0] = self.CMT.center[0]
                measuredTrack[count, 1] = self.CMT.center[1]
            count += 1


        numMeas = measuredTrack.shape[0]
        markedMeasure = np.ma.masked_less(measuredTrack, 0)

        # Kalman Filter Parameters
        deltaT = 1.0 / clip.fps
        transitionMatrix = [[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]]  # A
        observationMatrix = [[1, 0, 0, 0], [0, 1, 0, 0]]  # C

        xinit = markedMeasure[0, 0]
        yinit = markedMeasure[0, 1]
        vxinit = markedMeasure[1, 0] - markedMeasure[0, 0]
        vyinit = markedMeasure[1, 1] - markedMeasure[0, 1]
        initState = [xinit, yinit, vxinit, vyinit]  # mu0
        initCovariance = 1.0e-3 * np.eye(4)  # sigma0
        transistionCov = 1.0e-4 * np.eye(4)  # Q
        observationCov = 1.0e-1 * np.eye(2)  # R
        kf = KalmanFilter(transition_matrices=transitionMatrix,
            observation_matrices=observationMatrix,
            initial_state_mean=initState,
            initial_state_covariance=initCovariance,
            transition_covariance=transistionCov,
            observation_covariance=observationCov)

        self.measuredTrack = measuredTrack
        (self.filteredStateMeans, self.filteredStateCovariances) = kf.filter(markedMeasure)
        (self.filterStateMeanSmooth, self.filterStateCovariancesSmooth) = kf.smooth(markedMeasure)
        self.inity = np.mean(self.filterStateMeanSmooth[:][1], axis=0)
        newClip = clip.fl_image(self.crop)
        return newClip



    @counterFunction
    def crop (self, frame):

        self.frameCounter = self.crop.count
        # print self.frameCounter
        windowSize = (2 * 640, 2 * 360)
        newFrames = np.zeros((windowSize[0], windowSize[1], 3))

        if self.frameCounter <= self.numFrames:

            imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            y1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter) - 1][0] - windowSize[1] / 2)
            x1 = np.floor(self.inity - windowSize[0] / 2)
            x2 = np.floor(x1 + windowSize[1])
            y2 = np.floor(y1 + windowSize[0])

            # Corner correction (Height)
            if (x1 <= 0):
                x1 = 0
                x2 = np.floor(x1 + windowSize[1])
            if (x2 >= imGray.shape[0]):
                x2 = np.floor(imGray.shape[0])
                x1 = np.floor(x2 - windowSize[1])
            # Corner correction (Width)
            if (y1 <= 0):
                y1 = 0
                y2 = np.floor(y1 + windowSize[0])
            if (y2 >= imGray.shape[1]):
                y2 = np.floor(imGray.shape[1])
                y1 = np.floor(y2 - windowSize[0])
            # print x1, y1 , x2, y2
            newFrames = frame[x1:x2, y1:y2, :]

        return newFrames



class CMT_algorithm_kalman_filter_neighboring():



    def __init__(self, inputPath, skip=None):
        self.inputPath = inputPath  # 'The input path.'
        self.skip = skip  # 'Skip the first n frames.'
        self.frameCounter = 0
        self.numFrames = None
        self.CMT = CMT.CMT.CMT()


    def speakerTracker(self):


        # Clean up
        cv2.destroyAllWindows()

        if self.inputPath is not None:

            # If a path to a file was given, assume it is a single video file
            if os.path.isfile(self.inputPath):
                cap = cv2.VideoCapture(self.inputPath)
                clip = VideoFileClip(self.inputPath, audio=False)
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTracker] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_'



                # Skip first frames if required
                if self.skip is not None:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip)

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

                # Skip first frames if required
                if self.skip is not None:
                    cap.frame = 1 + self.skip

        else:
            # If no input path was specified, open camera device
            sys.exit("[speakerTracker] Error: no input path was specified")

        # The number of pixels from the center of object till the border of cropped image
        marginPixels = 300

        # Read first frame
        status, im0 = cap.read()
        imGray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        imDraw = np.copy(im0)

        (tl0, br0) = cmtutil.get_rect(imDraw)

        print '[speakerTracker] Using', tl0, br0, 'as initial bounding box for the speaker'

        # First initialization to get the center
        self.CMT.initialise(imGray0, tl0, br0)

        # The first x and y coordinates of the object of interest
        self.inity = tl0[1] - self.CMT.center_to_tl[1]
        self.initx = tl0[0] - self.CMT.center_to_tl[0]

        # Crop the first frame
        imGray0_initial = imGray0[self.inity - marginPixels : self.inity + marginPixels,
                                  self.initx - marginPixels : self.initx + marginPixels]


        # Calculate the translation vector from main image to the cropped frame
        self.originFromMainImageY = self.inity - marginPixels
        self.originFromMainImageX = self.initx - marginPixels

        # Calculate the position of the selected rectangle in the cropped frame
        tl = (tl0[0] - self.originFromMainImageX , tl0[1] - self.originFromMainImageY)
        br = (br0[0] - self.originFromMainImageX , br0[1] - self.originFromMainImageY)
        # print '[speakerTracker] Using', tl, br, 'as initial bounding box for the speaker'

        # initialization and keypoint calculation
        self.CMT.initialise(imGray0_initial, tl, br)

        # Center of object in cropped frame
        self.currentY = tl[1] - self.CMT.center_to_tl[1]
        self.currentX = tl[0] - self.CMT.center_to_tl[0]

        # Center of object in main frame
        self.currentYMainImage = self.currentY + self.originFromMainImageY
        self.currentXMainImage = self.currentX + self.originFromMainImageX


        measuredTrack = np.zeros((self.numFrames + 10, 2)) - 1
        count = 0


        # loop to read all frames,
        # crop them with the center of last frame,
        # calculate keypoints and center of the object


        for frame in clip.iter_frames():

            # Read the frame and convert it to gray scale
            im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Corner correction (Height)
            if (self.currentYMainImage + marginPixels >= im_gray.shape[0]):
                self.currentYMainImage = im_gray.shape[0] - marginPixels - 1
            else:
                self.currentYMainImage = self.currentYMainImage

            if (self.currentXMainImage + marginPixels >= im_gray.shape[1]):
                self.currentXMainImage = im_gray.shape[1] - marginPixels - 1
            else:
                self.currentXMainImage = self.currentXMainImage

            if (self.currentYMainImage - marginPixels <= 0):
                self.currentYMainImage = 0 + marginPixels + 1
            else:
                self.currentYMainImage = self.currentYMainImage

            if (self.currentXMainImage - marginPixels <= 0):
                self.currentXMainImage = 0 + marginPixels + 1
            else:
                self.currentXMainImage = self.currentXMainImage


            # Crop it by previous coordinates
            im_gray_crop = im_gray[self.currentYMainImage - marginPixels : self.currentYMainImage + marginPixels,
                                   self.currentXMainImage - marginPixels : self.currentXMainImage + marginPixels]

            # plt.imshow(im_gray_crop, cmap = cm.Greys_r)
            # plt.show()

            # print "self.currentYMainImage:", self.currentYMainImage
            # print "self.currentXMainImage:", self.currentXMainImage
            # print im_gray_crop.shape

            # Compute all keypoints in the cropped frame
            self.CMT.process_frame(im_gray_crop)
            # print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , count)


            if not (math.isnan(self.CMT.center[0]) or math.isnan(self.CMT.center[1])
                or (self.CMT.center[0] <= 0) or (self.CMT.center[1] <= 0)):

                # Compute the center of the object with respect to the main image
                self.diffY = self.CMT.center[0] - self.currentY
                self.diffX = self.CMT.center[1] - self.currentX

                self.currentYMainImage = self.diffY + self.currentYMainImage
                self.currentXMainImage = self.diffX + self.currentXMainImage

                self.currentY = self.CMT.center[0]
                self.currentX = self.CMT.center[1]
                # Save the center of frames in an array for further process
                measuredTrack[count, 0] = self.currentYMainImage
                measuredTrack[count, 1] = self.currentXMainImage

            else:
                self.currentYMainImage = self.currentYMainImage
                self.currentXMainImage = self.currentXMainImage



            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.currentYMainImage, self.currentXMainImage , count)
            count += 1

        numMeas = measuredTrack.shape[0]
        markedMeasure = np.ma.masked_less(measuredTrack, 0)

        # Kalman Filter Parameters
        deltaT = 1.0 / clip.fps
        transitionMatrix = [[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]]  # A
        observationMatrix = [[1, 0, 0, 0], [0, 1, 0, 0]]  # C

        xinit = markedMeasure[0, 0]
        yinit = markedMeasure[0, 1]
        vxinit = markedMeasure[1, 0] - markedMeasure[0, 0]
        vyinit = markedMeasure[1, 1] - markedMeasure[0, 1]
        initState = [xinit, yinit, vxinit, vyinit]  # mu0
        initCovariance = 1.0e-3 * np.eye(4)  # sigma0
        transistionCov = 1.0e-4 * np.eye(4)  # Q
        observationCov = 1.0e-1 * np.eye(2)  # R

        # Kalman Filter bias
        kf = KalmanFilter(transition_matrices=transitionMatrix,
            observation_matrices=observationMatrix,
            initial_state_mean=initState,
            initial_state_covariance=initCovariance,
            transition_covariance=transistionCov,
            observation_covariance=observationCov)

        self.measuredTrack = measuredTrack
        # Kalman Filter
        (self.filteredStateMeans, self.filteredStateCovariances) = kf.filter(markedMeasure)
        # Kalman Smoother
        (self.filterStateMeanSmooth, self.filterStateCovariancesSmooth) = kf.smooth(markedMeasure)
        newClip = clip.fl_image(self.crop)
        return newClip



    @counterFunction
    def crop (self, frame):

        self.frameCounter = self.crop.count
        # print self.frameCounter
        windowSize = (2 * 640, 2 * 360)
        newFrames = np.zeros((windowSize[0], windowSize[1], 3))

        if self.frameCounter <= self.numFrames:

            imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Use Kalman Filter Smoother results to crop the frames with corresponding window size
            x1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter) - 1][0] - windowSize[1] / 2)
            y1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter) - 1][1] - windowSize[0] / 2)
            x2 = np.floor(x1 + windowSize[1])
            y2 = np.floor(y1 + windowSize[0])

            # Corner correction (Height)
            if (x1 <= 0):
                x1 = 0
                x2 = np.floor(x1 + windowSize[1])
            if (x2 >= imGray.shape[0]):
                x2 = np.floor(imGray.shape[0])
                x1 = np.floor(x2 - windowSize[1])
            # Corner correction (Width)
            if (y1 <= 0):
                y1 = 0
                y2 = np.floor(y1 + windowSize[0])
            if (y2 >= imGray.shape[1]):
                y2 = np.floor(imGray.shape[1])
                y1 = np.floor(y2 - windowSize[0])
            # print x1, y1 , x2, y2
            newFrames = frame[x1:x2, y1:y2, :]

        return newFrames



class BBoxTracker(object):

    def __init__(self):
        self.bboxes = []

    def set_size(self, width, height):

        self.width = width
        self.height = height

    def add_bounding_box(self, timestamp, center, width):
        # TODO resize according to the original size
        self.bboxes.append((timestamp, center, width))

class DummyTracker(object):
    """A simple implementation of the speacker tracker"""

    def __init__(self,
                 inputPath,
                 slide_coordinates,
                 resize_max=None,
                 fps=None,
                 skip=None,
                 speaker_bb_height_location=None):
        """
        :param inputPath: input video file or path containing images
        :param slide_coordinates: the coordinates where the slides are located (in 0-1 space)
        :param resize_max: max size
        :param fps: frame per second in use if the video is a sequence of image files
        :param speaker_bb_height_location: if given, this will be used as the possible heights at which the speaker should be tracked.
        """

        if inputPath is None:
            raise exceptions.RuntimeError("no input specified")

        self.inputPath = inputPath  # 'The input path.'
        self.skip = skip  # 'Skip the first n frames.'

        self.slide_crop_coordinates = self._inner_rectangle(slide_coordinates)
        print self.slide_crop_coordinates
        self.resize_max = resize_max
        self.tracker = BBoxTracker()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.fgbg = cv2.BackgroundSubtractorMOG()

        # TODO this location should be in the full frame, or indicated in the range [0,1]
        self.speaker_bb_height_location = speaker_bb_height_location


    def _inner_rectangle(self, coordinates):
        """Get the inner rectangle of the slide coordinates for cropping the image.

           Returns a 4x1 numpy array in the order:
           [min_y, max_y, min_x, max_x]
        """

        # This is specified by the rectify_coordinates() function in slideDetection.py
        top_left = 0
        top_right = 1
        bottom_right = 2
        bottom_left = 3

        x = 0
        y = 1

        min_x = max(coordinates[top_left, x], coordinates[bottom_left, x])
        max_x = min(coordinates[top_right, x], coordinates[bottom_right, x])

        # y is flipped, so top and bottom are as well
        min_y = max(coordinates[top_left, y], coordinates[top_right, y])
        max_y = min(coordinates[bottom_left, y], coordinates[bottom_right, y])

        return np.array([min_y, max_y, min_x, max_x])

    def _resize(self, im):
        """Resizes the input image according to the initial parameters"""
        if self.resize_max is None:
            return im

        # assuming landscape orientation
        dest_size = self.resize_max, int(im.shape[0] * (float(self.resize_max) / im.shape[1]))
        return cv2.resize(im, dest_size)


    def speakerTracker(self):


        # Clean up
        cv2.destroyAllWindows()


        # TODO move this in a function
        # If a path to a file was given, assume it is a single video file
        if os.path.isfile(self.inputPath):
            cap = cv2.VideoCapture(self.inputPath)
            clip = VideoFileClip(self.inputPath, audio=False)
            self.fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

            self.width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
            self.height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

            self.tracker.set_size(self.width, self.height)

            logger.info("[VIDEO] %d frames @ %d fps, frames size (%d x %d)",
                    self.numFrames, self.fps, self.width, self.height)

            pathBase = os.path.basename(self.inputPath)
            pathDirectory = os.path.dirname(self.inputPath)
            baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_' + 'speakerCoordinates.txt'


            # Skip first frames if required
            if self.skip is not None:
                cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip)

        # Otherwise assume it is a format string for reading images
        else:
            cap = cmtutil.FileVideoCapture(self.inputPath)

            # Skip first frames if required
            if self.skip is not None:
                cap.frame = 1 + self.skip


        # Read first frame
        status, im0_not_resized = cap.read()

        im0 = self._resize(im0_not_resized)
        im0_lab = cv2.cvtColor(im0, cv2.COLOR_BGR2LAB)
        im0_gray = cv2.cvtColor(im0_not_resized, cv2.COLOR_BGR2GRAY)

        if debug:
            # speaker bounding box used for debugging
            (tl, br) = (2052, 948), (2376, 1608)
        else:
            imDraw = np.copy(im0)
            (tl, br) = cmtutil.get_rect(imDraw)


        logger.info('[TRACKER] Using %s, %s as initial bounding box for the speaker', tl, br)
        measuredTrack = np.zeros((self.numFrames + 10, 2)) - 1

        frame_count = -1
        # previous histogram
        previous_hist_plane = None
        previous_hist_vertical_stripes = None  # previous histograms computed vertically for "activity" recognition on the area where the speaker is
        distances_histogram = {}

        while frame_count <= self.numFrames:

            status = cap.grab()
            if not status:
                break

            frame_count += 1
            time = float(frame_count) / float(self.fps)
            current_time_stamp = datetime.timedelta(seconds=int(time))

            if (self.fps is not None) and (frame_count % self.fps) != 0:
                continue

            logging.info('[VIDEO] processing frame %.6d / %d - time %s / %s - %3.3f %%',
                         frame_count,
                         self.numFrames,
                         current_time_stamp,
                         datetime.timedelta(seconds=self.numFrames / self.fps),
                         100 * float(frame_count) / self.numFrames)
            status, im = cap.retrieve()

            if not status:
                logger.error('[VIDEO] error reading frame %d', frame_count)

            # resize and color conversion
            im = self._resize(im)
            im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # color diff
            im_diff = (im_lab - im0_lab) ** 2
            im_diff_lab = np.sqrt(np.sum(im_diff, axis=2))

            # background
            fgmask = self.fgbg.apply(im0)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel)

            # threshold the diff
            # histogram
            hist = []
            for i in range(im_diff.shape[2]):
                hist.append(cv2.calcHist([im_diff], [i], None, [256], [0, 256]))

            hist_plane = []
            slide_hist_plane = []

            # Compute the histogram for the slide image
            resized_x = im_diff_lab.shape[1]
            resized_y = im_diff_lab.shape[0]

            min_y = self.slide_crop_coordinates[0] * resized_y
            max_y = self.slide_crop_coordinates[1] * resized_y
            min_x = self.slide_crop_coordinates[2] * resized_x
            max_x = self.slide_crop_coordinates[3] * resized_x
            slide = im_gray[min_y : max_y, min_x : max_x]
            slidehist = cv2.calcHist([slide], [0], None, [256], [0, 256])

            plt.subplot(2, 1, 1)
            plt.imshow(slide, cmap=cm.Greys_r)
            plt.subplot(2, 1, 2)
            plt.plot(slidehist)
            plt.xlim([0, 256])

            histogram_boundaries = get_histogram_min_max_with_percentile(slidehist, False)

            # this is part of a pre-processing
            # dividing the plane vertically by N=3 and computing histograms on that. The purpose of this is to detect the environment changes
            N_stripes = 3
            for i in range(N_stripes):
                location = int(i * im_diff_lab.shape[0] / float(N_stripes)), min(im_diff_lab.shape[0], int((i + 1) * im_diff_lab.shape[0] / float(N_stripes)))
                current_plane = im_diff_lab[location[0]:location[1], :]


                # print current_plane.min(), current_plane.max()
                hist_plane.append(cv2.calcHist([current_plane.astype(np.uint8)], [0], None, [256], [0, 256]))
                # slide_hist_plane.append(cv2.calcHist(current_slide_plane))

            # dividing the location of the speaker by N=10 vertical stripes. The purpose of this is to detect the x location of activity/motion
            hist_vertical_stripes = []
            energy_vertical_stripes = []
            N_vertical_stripes = 10
            if self.speaker_bb_height_location is not None:

                for i in range(N_vertical_stripes):
                    location = int(i * im_diff_lab.shape[1] / float(N_vertical_stripes)), min(im_diff_lab.shape[1], int((i + 1) * im_diff_lab.shape[1] / float(N_vertical_stripes)))
                    current_vertical_stripe = im_diff_lab[self.speaker_bb_height_location[0]:self.speaker_bb_height_location[1], location[0]:location[1]]
                    hist_vertical_stripes.append(cv2.calcHist([current_vertical_stripe.astype(np.uint8)], [0], None, [256], [0, 256]))
                    energy_vertical_stripes.append(current_vertical_stripe.sum())
                    pass
                pass

            # histogram distance

            # location of all the connected components

            if previous_hist_plane is not None:
                distances_histogram[frame_count] = {}
                element = distances_histogram[frame_count]
                # element['timestamp'] = current_time_stamp
                element['dist_stripes'] = {}
                for e, h1, h2 in zip(range(N_stripes), previous_hist_plane, hist_plane):
                    element['dist_stripes'][e] = cv2.compareHist(h1, h2, cv2.cv.CV_COMP_CORREL)


            if previous_hist_vertical_stripes is not None:
                element = distances_histogram.get(frame_count, {})
                distances_histogram[frame_count] = element
                element['vert_stripes'] = {}
                for e, h1, h2 in zip(range(N_vertical_stripes), previous_hist_vertical_stripes, hist_vertical_stripes):
                    element['vert_stripes'][e] = cv2.compareHist(h1, h2, cv2.cv.CV_COMP_CORREL)

            # "activity" which is the enery in each stripe
            element = distances_histogram.get(frame_count, {})
            distances_histogram[frame_count] = element
            element['energy_stripes'] = {}
            element['peak_stripes'] = {}
            element['histogram_boundaries'] = {}
            for e, energy, h1 in zip(range(N_vertical_stripes), energy_vertical_stripes, hist_vertical_stripes):
                element['energy_stripes'][e] = int(energy)
                element['peak_stripes'][e] = max([i for i, j in enumerate(h1) if j > 0])

            # Store histogram boundaries
            element['histogram_boundaries']['min'] = histogram_boundaries[0]
            element['histogram_boundaries']['max'] = histogram_boundaries[1]


            # debug
            if debug:
                cv2.imwrite(os.path.join(_tmp_path, 'background_%.6d.png' % frame_count), fgmask)
                cv2.imwrite(os.path.join(_tmp_path, 'diff_%.6d.png' % frame_count), im_diff)
                cv2.imwrite(os.path.join(_tmp_path, 'diff_lab_%.6d.png' % frame_count), im_diff_lab)

                with open(os.path.join(_tmp_path, 'info_%.6d.json' % frame_count), 'w') as f:
                    f.write(json.dumps(distances_histogram))
                # cv2.imwrite(os.path.join(_tmp_path, 'diff_thres_%.6d.png' % frame_count), color_mask)


            im0 = im
            im0_lab = im_lab
            im0_gray = im_gray

            previous_hist_plane = hist_plane
            previous_hist_vertical_stripes = hist_vertical_stripes

            continue


            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , frame_count)
            if not (math.isnan(self.CMT.center[0])
                    or math.isnan(self.CMT.center[1])
                    or (self.CMT.center[0] <= 0)
                    or (self.CMT.center[1] <= 0)):
                measuredTrack[frame_count, 0] = self.CMT.center[0]
                measuredTrack[frame_count, 1] = self.CMT.center[1]
            else:
                # take the previous estimate if none is found in the current frame
                measuredTrack[frame_count, 0] = measuredTrack[frame_count - 1, 0]
                measuredTrack[frame_count, 1] = measuredTrack[frame_count - 1, 1]

            if debug:
                cmtutil.draw_bounding_box((int(measuredTrack[frame_count, 0] - 50), int(measuredTrack[frame_count, 1] - 50)),
                                          (int(measuredTrack[frame_count, 0] + 50), int(measuredTrack[frame_count, 1] + 50)),
                                          im_debug)


                cv2.imwrite(os.path.join(_tmp_path, 'debug_file_%.6d.png' % frame_count), im_debug)

                im_debug = np.copy(im)
                cmtutil.draw_keypoints([kp.pt for kp in self.CMT.keypoints_cv], im_debug, (0, 0, 255))
                cv2.imwrite(os.path.join(_tmp_path, 'all_keypoints_%.6d.png' % frame_count), im_debug)



        return


    def crop (self, frame):

        windowSize = (2 * 640, 2 * 360)
        newFrames = np.zeros((windowSize[0], windowSize[1], 3))
        imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.CMT.process_frame(imGray)

        if not (math.isnan(self.CMT.center[0]) or math.isnan(self.CMT.center[1])
                or (self.CMT.center[0] <= 0) or (self.CMT.center[1] <= 0)):

            x1 = np.floor(self.CMT.center[1] - windowSize[1] / 2)
            y1 = np.floor(self.CMT.center[0] - windowSize[0] / 2)
            x2 = np.floor(x1 + windowSize[1])
            y2 = np.floor(y1 + windowSize[0])

            # Corner correction (Height)
            if (x1 <= 0):
                x1 = 0
                x2 = np.floor(x1 + windowSize[1])
            if (x2 >= imGray.shape[0]):
                x2 = np.floor(imGray.shape[0])
                x1 = np.floor(x2 - windowSize[1])
            # Corner correction (Width)
            if (y1 <= 0):
                y1 = 0
                y2 = np.floor(y1 + windowSize[0])
            if (y2 >= imGray.shape[1]):
                y2 = np.floor(imGray.shape[1])
                y1 = np.floor(y2 - windowSize[0])
            newFrames = frame[x1:x2, y1:y2, :]

        # print 'Center: {0:.2f},{1:.2f}'.format(CMT.center[0], CMT.center[1])
        return newFrames


def plot_histogram_distances():
    """Reads back the sequence of histograms and plots the distance between two consecutive histograms over time"""

    list_files = glob.glob(os.path.join(_tmp_path, 'info_*.json'))
    list_files.sort()
    # the last one contains all the necessary information, -2 is used for tests while the script is running
    last_file = list_files[-2]


    with open(last_file) as f:
        distances_histogram = json.load(f)

    frame_indices = [(i, int(i)) for i in distances_histogram.keys()]
    frame_indices.sort(key=lambda x: x[1])

    plots_dict = {}
    plots_dict_index = []
    for count, count_integer in frame_indices:
        if 'dist_stripes' not in distances_histogram[count]:
            continue
        plots_dict_index.append(count_integer)
        current_sample = distances_histogram[count]['dist_stripes']
        for i in current_sample.keys():

            if int(i) not in plots_dict:
                plots_dict[int(i)] = []

            plots_dict[int(i)].append(float(current_sample[i]))

    N_stripes = max(plots_dict.keys())

    # vertical stripes are the location of the speaker
    plots_dict_vert_stripes = {}
    plots_dict_vert_stripes_index = []
    for count, count_integer in frame_indices:
        if 'vert_stripes' not in distances_histogram[count]:
            continue

        plots_dict_vert_stripes_index.append(count_integer)
        current_sample = distances_histogram[count]['vert_stripes']
        for i in current_sample.keys():
            if int(i) not in plots_dict_vert_stripes:
                plots_dict_vert_stripes[int(i)] = []

            plots_dict_vert_stripes[int(i)].append(float(current_sample[i]))

    plots_dict_vert_stripes_energy = {}
    plots_dict_vert_stripes_energy_index = []
    for count, count_integer in frame_indices:
        current_sample = distances_histogram[count]['energy_stripes']
        plots_dict_vert_stripes_energy_index.append(count_integer)
        for i in current_sample.keys():
            if int(i) not in plots_dict_vert_stripes_energy:
                plots_dict_vert_stripes_energy[int(i)] = []

            plots_dict_vert_stripes_energy[int(i)].append(float(current_sample[i]))

    N_stripes_vert = max(plots_dict_vert_stripes.keys())




    from matplotlib import pyplot as plt




    for i in sorted(plots_dict.keys()):
        plt.subplot(N_stripes + 1, 1, i + 1)  # , sharex=True)
        plt.plot(plots_dict_index, plots_dict[i], aa=False, linewidth=1)
        if i == 0:
            plt.title('Histogram distance for each stripe')

        # lines.set_linewidth(1)
        plt.ylabel('Stripe %d' % i)

    plt.xlabel('frame #')

    plt.savefig(os.path.join(_tmp_path, 'histogram_distance.png'))


    # plotting the vertical stripes content
    for i in sorted(plots_dict_vert_stripes.keys()):

        plt.subplot(N_stripes_vert + 1, 1, i + 1)  # , sharex=True)
        plt.plot(plots_dict_vert_stripes_index, plots_dict_vert_stripes[i], aa=False, linewidth=1)

        if i == 0:
            plt.title('Histogram distance for each vertical stripe')

        # lines.set_linewidth(1)
        plt.ylabel('%d' % i)

        plt.tick_params(axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom='off',  # ticks along the bottom edge are off
                        top='off',  # ticks along the top edge are off
                        labelbottom='off')  # labels along the bottom edge are off

        plt.tick_params(axis='y',
                        which='both',
                        left='off',  # ticks along the bottom edge are off
                        right='off',  # ticks along the top edge are off
                        top='off',
                        bottom='off',
                        labelleft='on')


    plt.xlabel('frame #')
    plt.tick_params(axis='x',
                    which='both',
                    bottom='on',
                    top='off',
                    labelbottom='on')

    plt.savefig(os.path.join(_tmp_path, 'histogram_vert_distance.png'), dpi=(200))


    # plotting the vertical stripes content: energy
    for i in sorted(plots_dict_vert_stripes_energy.keys()):

        plt.subplot(N_stripes_vert + 1, 1, i + 1)  # , sharex=True)
        plt.plot(plots_dict_vert_stripes_energy_index, plots_dict_vert_stripes_energy[i], aa=False, linewidth=1)

        if i == 0:
            plt.title('Energy for each vertical stripe')


        plt.tick_params(axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom='off',  # ticks along the bottom edge are off
                        top='off',  # ticks along the top edge are off
                        labelbottom='off')  # labels along the bottom edge are off

        plt.tick_params(axis='y',
                        which='both',
                        left='off',  # ticks along the bottom edge are off
                        right='off',  # ticks along the top edge are off
                        top='off',
                        bottom='off',
                        labelleft='on')

        # lines.set_linewidth(1)
        plt.ylabel('%d' % i, fontsize=3)
        # plt.axis([0, max(plots_dict_vert_stripes_energy[i])])


    plt.xlabel('frame #')
    plt.tick_params(axis='x',
                    which='both',
                    bottom='on',
                    top='off',
                    labelbottom='on')

    plt.savefig(os.path.join(_tmp_path, 'histogram_vert_energy.png'), dpi=(200))


class SimpleTracker(object):
    """An even simpler implementation by Edgar, based on Raffi's code and ideas"""

    def __init__(self,
                 inputPath,
                 slide_coordinates,
                 resize_max=None,
                 fps=None,
                 skip=None,
                 speaker_bb_height_location=None):
        """
        :param inputPath: input video file or path containing images
        :param slide_coordinates: the coordinates where the slides are located (in 0-1 space)
        :param resize_max: max size
        :param fps: frame per second in use if the video is a sequence of image files
        :param speaker_bb_height_location: if given, this will be used as the possible heights at which the speaker should be tracked.
        """

        if inputPath is None:
            raise exceptions.RuntimeError("no input specified")

        self.inputPath = inputPath  # 'The input path.'
        self.skip = skip  # 'Skip the first n frames.'
        self.fps = fps
        self.source = None

        self.y_location = 200 # 200 for video_7, 270 for the other

        self.min_y_detect = self.y_location - 20
        self.max_y_detect = self.y_location + 20
        self.difference_threshold = 20.0

        self.slide_crop_coordinates = self._inner_rectangle(slide_coordinates)
        logging.info('[SLIDES] Slide coordinates: %s', self.slide_crop_coordinates)
        self.resize_max = resize_max
        #self.tracker = BBoxTracker()
        #self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #self.fgbg = cv2.BackgroundSubtractorMOG()

        # TODO this location should be in the full frame, or indicated in the range [0,1]
        #self.speaker_bb_height_location = speaker_bb_height_location


    def _inner_rectangle(self, coordinates):
        """Get the inner rectangle of the slide coordinates for cropping the image.

           Returns a 4x1 numpy array in the order:
           [min_y, max_y, min_x, max_x]
        """

        # This is specified by the rectify_coordinates() function in slideDetection.py
        top_left = 0
        top_right = 1
        bottom_right = 2
        bottom_left = 3

        x = 0
        y = 1

        min_x = max(coordinates[top_left, x], coordinates[bottom_left, x])
        max_x = min(coordinates[top_right, x], coordinates[bottom_right, x])

        # y is flipped, so top and bottom are as well
        min_y = max(coordinates[top_left, y], coordinates[top_right, y])
        max_y = min(coordinates[bottom_left, y], coordinates[bottom_right, y])

        return np.array([min_y, max_y, min_x, max_x])

    def _resize(self, im):
        """Resizes the input image according to the initial parameters"""
        if self.resize_max is None:
            return im

        # assuming landscape orientation
        dest_size = self.resize_max, int(im.shape[0] * (float(self.resize_max) / im.shape[1]))
        return cv2.resize(im, dest_size)


    def speakerTracker(self):

        # Clean up
        cv2.destroyAllWindows()

        # TODO move this in a function
        # If a path to a file was given, assume it is a single video file
        if os.path.isfile(self.inputPath):
            cap = cv2.VideoCapture(self.inputPath)
            #clip = VideoFileClip(self.inputPath, audio=False)
            self.fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

            self.width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
            self.height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

            logger.info("[VIDEO] %d frames @ %d fps, frames size (%d x %d)",
                    self.numFrames, self.fps, self.width, self.height)
            self.source = 'video'

            # Skip first frames if required
            if self.skip is not None:
                cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip)

        # Otherwise assume it is a format string for reading images
        else:
            cap = cmtutil.FileVideoCapture(self.inputPath)
            self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

            logger.info("[VIDEO] %d frames in path %s", self.numFrames, self.inputPath)
            self.source = 'path'

            # Skip first frames if required
            if self.skip is not None:
                cap.frame = 1 + self.skip


        # Read first frame
        status, im0_not_resized = cap.read()

        # initialize the old images
        im0 = self._resize(im0_not_resized)
        im0_lab = cv2.cvtColor(im0, cv2.COLOR_BGR2LAB)
        im0_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)

        frame_count = 0

        # initialize Kalman filter
        m = np.matrix('0; 0')
        m[0,0] = self.resize_max/2
        P = np.matrix('100000 0; 0 100000')
        A = np.matrix('1 0.01; 0 1')
        C = np.matrix('1 0')
        Q = np.matrix('50 0; 0 1')
        R = np.matrix('100')

        # plotting
        mass = np.zeros(shape=(self.numFrames,1))

        data = {}

        while frame_count < self.numFrames - 1: # -1

            status = cap.grab()
            if not status:
                break

            frame_count += 1
            if self.source=='video':
                time = float(frame_count) / float(self.fps)
            else:
                time = float(frame_count)
            current_time_stamp = datetime.timedelta(seconds=int(time))

            if self.source=='video' and (self.fps is not None) and (frame_count % self.fps) != 0:
                continue

            #logging.info('[VIDEO] processing frame %.6d / %d - time %s / %s - %3.3f %%',
                         #frame_count,
                         #self.numFrames,
                         #current_time_stamp,
                         #datetime.timedelta(seconds=self.numFrames / self.fps),
                         #100 * float(frame_count) / self.numFrames)
            status, im = cap.retrieve()

            #if not status:
                #logger.error('[VIDEO] error reading frame %d', frame_count)

            # resize and color conversion
            im = self._resize(im)
            im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # color diff
            im_diff = (im_lab - im0_lab) ** 2
            im_diff_lab = np.sum(im_diff, axis=2)/3

            # debug
            if debug:
                cv2.imwrite(os.path.join(_tmp_path, 'diff_%.6d.png' % frame_count), im_diff)
                cv2.imwrite(os.path.join(_tmp_path, 'diff_lab_%.6d.png' % frame_count), im_diff_lab)

            ''' simple centroid "measurement" '''

            gauss_filter = np.tile(np.exp(- (np.power(np.array(range(0,640))-m.item(0),2) / (2*25.0**2))), [im.shape[0],1])
            im_diff_lab_u8 = cv2.convertScaleAbs(im_diff_lab) # convert to 8 bit image
            im_diff_lab_u8[im_diff_lab_u8 < self.difference_threshold] = 0 # threshold to remove artifacts
            im_diff_lab_u8[:self.min_y_detect, : ] = 0 # zero out the top
            im_diff_lab_u8[self.max_y_detect:, : ] = 0 # zero out the bottom
            im_diff_lab_u8 = im_diff_lab_u8*gauss_filter

            moments = cv2.moments(im_diff_lab_u8) # moments gives us the centroid
            cx = moments['m10']/moments['m00'] # weighted centroid in x coordinates
            cy = moments['m01']/moments['m00'] # weighted centroid in y coordinates
            cy = self.y_location

            data[frame_count] = {}
            data[frame_count]['x'] = cx

            # Kalman filter
            #   prediction
            m_ = A*m
            P_ = C*P*C.transpose() + Q

            nll = ((cx - C*m_)**2 / (2*P.item(0)))

            mass[frame_count,0] = nll

            if nll < 400:
                #   update
                r = cx - C*m_
                S = C*P_*C.transpose() + R
                K = P_*C.transpose()*np.linalg.inv(S)
                m = m_ + K*r
                P = P_ - K*S*K.transpose()
            else:
                P = P_

            data[frame_count]['m'] = m.item(0)

            # store the calculated position
            with open(os.path.join(_tmp_path, 'simpleTracker_%.6d.json' % frame_count), 'w') as f:
                    f.write(json.dumps(data))

            #mass[frame_count,0] = moments['m00']

            kx = m.item(0)
            if kx < 120:
                kx = 120
            speaker = im_gray[cy-50:cy+50, int(kx)-120:int(kx)+120]

            # plot for debugging
            if debug:
                plt.clf()
                plt.ion()

                plt.subplot(2, 1, 1)
                plt.imshow(im_gray, cmap=cm.Greys_r)
                plt.plot([cx], [cy], 'r+', markersize=50.0)
                plt.plot([m.item(0)], [cy], 'g+', markersize=50.0)
                plt.autoscale(tight=True)

                plt.subplot(2, 1, 2)
                plt.imshow(speaker, cmap=cm.Greys_r)
                #plt.plot(mass)

                plt.draw()
                plt.show()

            # prepare for the next frame
            im0 = im
            im0_lab = im_lab
            im0_gray = im_gray

        return

def plot_simpleTracker_result():
    list_files = glob.glob(os.path.join(_tmp_path, 'simpleTracker_*.json'))
    list_files.sort()
    last_file = list_files[-1]
    with open(last_file) as f:
        data = json.load(f)
    frame_indices = [(i, int(i)) for i in data.keys()]
    frame_indices.sort(key=lambda x: x[1])

    X = np.zeros((len(frame_indices),1))
    M = np.zeros((len(frame_indices),1))
    T = np.zeros((len(frame_indices),1))
    for i in range(len(frame_indices)):
        X[i] = data[frame_indices[i][0]]['x']
        M[i] = data[frame_indices[i][0]]['m']
        T[i] = frame_indices[i][1]

    plt.plot(T,X,'b')
    plt.plot(T,M,'r')
    plt.show()

if __name__ == '__main__':

    storage = '../Videos'
    filename = 'video_7.mp4'
    #storage = '/is/ei/eklenske/Videos/dummy/thumbnails/'
    #filename = ''

    # plot_histogram_distances()
    # sys.exit(0)

    try:

        if not True:
            obj = SimpleTracker(os.path.join(storage, filename),
                              slide_coordinates=np.array([[ 0.36004776, 0.01330207],
                                                          [ 0.68053395, 0.03251761],
                                                          [ 0.67519468, 0.42169076],
                                                          [ 0.3592881, 0.41536275]]),
                              resize_max=640,
                              speaker_bb_height_location=(155, 260),
                              fps=30)
            new_clip = obj.speakerTracker()
            #plot_histogram_distances()
        else:
            plot_simpleTracker_result()

        # plot_histogram_distances()
        # sys.exit(0)
        #new_clip.write_videofile("video_CMT_algorithm_kalman_filter.mp4")

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
        frame = last_frame().tb_frame
        ns = dict(frame.f_globals)
        ns.update(frame.f_locals)
        code.interact(local=ns)
