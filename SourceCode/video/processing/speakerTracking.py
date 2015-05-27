

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
- CMT_algorithm_kalman_filter_neighbering
    --> to run the CMT, predict, update and smooth the results by kalman filter (keypoint calculation is only for neighboring window)
- FOV_specification 
    --> to get a certain amount of field of view including the speaker
"""

import cv2
from numpy import empty, nan
import os
import sys
import time
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


def counterFunction(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        return func(*args, **kwargs)
    tmp.count = 0
    return tmp


class CMT_algorithm():



    def __init__(self,inputPath, bBox = None , skip = None):
        self.inputPath = inputPath     # 'The input path.'
        self.bBox = bBox               # 'Specify initial bounding box.'
        self.skip = skip               # 'Skip the first n frames.'
        self.CMT = CMT.CMT.CMT()

       
    def speakerTracker(self):
        

        # Clean up
        cv2.destroyAllWindows()

        if self.inputPath is not None:

            # If a path to a file was given, assume it is a single video file
            if os.path.isfile(self.inputPath):
                cap = cv2.VideoCapture(self.inputPath)
                clip  = VideoFileClip(self.inputPath,audio=False)
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTrackering] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_' + 'speakerCoordinates.txt'
                

                #Skip first frames if required
                if self.skip is not None:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip)

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

                #Skip first frames if required
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
        
        newClip = clip.fl_image( self.crop )
                   
        return newClip 
    
    
    def crop (self,frame):
    
        windowSize = (2*640,2*360)
        newFrames = np.zeros((windowSize[0],windowSize[1],3))
        imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.CMT.process_frame(imGray)

        if not (math.isnan(self.CMT.center[0]) or math.isnan(self.CMT.center[1])
                or (self.CMT.center[0] <= 0) or (self.CMT.center[1] <= 0)): 
            
            x1 = np.floor(self.CMT.center[1] - windowSize[1]/2)
            y1 = np.floor(self.CMT.center[0] - windowSize[0]/2)
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
            newFrames = frame[x1:x2,y1:y2,:]
            
        #print 'Center: {0:.2f},{1:.2f}'.format(CMT.center[0], CMT.center[1])
        return newFrames



class CMT_algorithm_kalman_filter():



    def __init__(self,inputPath, skip = None):
        self.inputPath = inputPath     # 'The input path.'
        self.skip = skip            # 'Skip the first n frames.'
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
                clip  = VideoFileClip(self.inputPath,audio=False)
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTracker] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_' 
                
                
                
                #Skip first frames if required
                if self.skip is not None:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip)

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

                #Skip first frames if required
                if self.skip is not None:
                    cap.frame = 1 + self.skip

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

        measuredTrack=np.zeros((self.numFrames+10,2))-1
           
        count=0
        
        while count <= self.numFrames:
            
            status, im = cap.read()
            if not status:
                break
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            
            self.CMT.process_frame(im_gray)

            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , count)
            if not (math.isnan(self.CMT.center[0]) or math.isnan(self.CMT.center[1])
                or (self.CMT.center[0] <= 0) or (self.CMT.center[1] <= 0)):
                measuredTrack[count,0] = self.CMT.center[0]
                measuredTrack[count,1] = self.CMT.center[1]               
            count += 1                
        
        numMeas=measuredTrack.shape[0]
        markedMeasure=np.ma.masked_less(measuredTrack,0)
        
        # Kalman Filter Parameters
        deltaT = 1.0/clip.fps
        transitionMatrix=[[1,0,deltaT,0],[0,1,0,deltaT],[0,0,1,0],[0,0,0,1]]   #A
        observationMatrix=[[1,0,0,0],[0,1,0,0]]   #C

        xinit = markedMeasure[0,0]
        yinit = markedMeasure[0,1]
        vxinit = markedMeasure[1,0]-markedMeasure[0,0]
        vyinit = markedMeasure[1,1]-markedMeasure[0,1]
        initState = [xinit,yinit,vxinit,vyinit]    #mu0
        initCovariance = 1.0e-3*np.eye(4)          #sigma0
        transistionCov = 1.0e-4*np.eye(4)          #Q
        observationCov = 1.0e-1*np.eye(2)          #R
        kf = KalmanFilter(transition_matrices = transitionMatrix,
            observation_matrices = observationMatrix,
            initial_state_mean = initState,
            initial_state_covariance = initCovariance,
            transition_covariance = transistionCov,
            observation_covariance = observationCov)
        
        self.measuredTrack = measuredTrack
        (self.filteredStateMeans, self.filteredStateCovariances) = kf.filter(markedMeasure)
        (self.filterStateMeanSmooth, self.filterStateCovariancesSmooth) = kf.smooth(markedMeasure)
        
        #np.savetxt((baseName + 'speakerCoordinates_CMT_Kalman.txt'), 
                   #np.hstack((np.asarray(self.filteredStateMeans), np.asarray(self.filteredStateCovariances) ,
                   #np.asarray(self.filterStateMeanSmooth), np.asarray(self.filterStateCovariancesSmooth) )))
        #np.savetxt((baseName + 'speakerCoordinates_CMT.txt'), np.asarray(measuredTrack))
        
        newClip = clip.fl_image( self.crop ) 
        return newClip 
    
    
    
    @counterFunction
    def crop (self,frame):

        self.frameCounter = self.crop.count
        #print self.frameCounter
        windowSize=(2*640,2*360)        
        newFrames = np.zeros((windowSize[0],windowSize[1],3))
        
        if self.frameCounter <= self.numFrames:
            
            imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            x1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter)-1][1] - windowSize[1]/2)
            y1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter)-1][0] - windowSize[0]/2)
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
            #print x1, y1 , x2, y2
            newFrames = frame[x1:x2,y1:y2,:]    
            
        return newFrames
    

class FOV_specification():



    def __init__(self,inputPath, skip = None):
        self.inputPath = inputPath     # 'The input path.'
        self.skip = skip                # 'Skip the first n frames.'
        self.CMT = CMT.CMT.CMT()


    def speakerTracker(self):


        # Clean up
        cv2.destroyAllWindows()

        if self.inputPath is not None:

            # If a path to a file was given, assume it is a single video file
            if os.path.isfile(self.inputPath):
                cap = cv2.VideoCapture(self.inputPath)
                clip  = VideoFileClip(self.inputPath,audio=False)
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTrackering] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_' + 'speakerCoordinates.txt'


                #Skip first frames if required
                if self.skip is not None:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip)

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

                #Skip first frames if required
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
        y2 = np.floor(tl[1] + np.abs((br[0] - tl[0])*(3.0/4.0))) 

        print x1, x2, y1, y2
        croppedClip = moviepycrop(clip, x1, y1, x2, y2)

        return croppedClip
       

class CMT_algorithm_kalman_filter_stripe():



    def __init__(self,inputPath, skip = None):
        self.inputPath = inputPath     # 'The input path.'
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
                clip  = VideoFileClip(self.inputPath,audio=False)
                W,H = clip.size
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
        y1 = tl[1]-100
        y2 = br[1]+100       
 
        imGray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        croppedGray0 = imGray0[y1:y2, x1:x2]
        self.CMT.initialise(croppedGray0, tl, br)

        measuredTrack=np.zeros((self.numFrames+10,2))-1
        
        
        count =0
        for frame in clip.iter_frames():
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            im_gray = grayFrame[y1:y2, x1:x2]
            
                       
            plt.imshow(im_gray, cmap = cm.Greys_r)
            plt.show()
            
            self.CMT.process_frame(im_gray)

            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , count)
            if not (math.isnan(self.CMT.center[0]) or math.isnan(self.CMT.center[1])
                or (self.CMT.center[0] <= 0) or (self.CMT.center[1] <= 0)):
                measuredTrack[count,0] = self.CMT.center[0]
                measuredTrack[count,1] = self.CMT.center[1]               
            count += 1 

    
        numMeas=measuredTrack.shape[0]
        markedMeasure=np.ma.masked_less(measuredTrack,0)
        
        # Kalman Filter Parameters
        deltaT = 1.0/clip.fps
        transitionMatrix=[[1,0,deltaT,0],[0,1,0,deltaT],[0,0,1,0],[0,0,0,1]]   #A
        observationMatrix=[[1,0,0,0],[0,1,0,0]]   #C

        xinit = markedMeasure[0,0]
        yinit = markedMeasure[0,1]
        vxinit = markedMeasure[1,0]-markedMeasure[0,0]
        vyinit = markedMeasure[1,1]-markedMeasure[0,1]
        initState = [xinit,yinit,vxinit,vyinit]    #mu0
        initCovariance = 1.0e-3*np.eye(4)          #sigma0
        transistionCov = 1.0e-4*np.eye(4)          #Q
        observationCov = 1.0e-1*np.eye(2)          #R
        kf = KalmanFilter(transition_matrices = transitionMatrix,
            observation_matrices = observationMatrix,
            initial_state_mean = initState,
            initial_state_covariance = initCovariance,
            transition_covariance = transistionCov,
            observation_covariance = observationCov)
        
        self.measuredTrack = measuredTrack
        (self.filteredStateMeans, self.filteredStateCovariances) = kf.filter(markedMeasure)
        (self.filterStateMeanSmooth, self.filterStateCovariancesSmooth) = kf.smooth(markedMeasure)
        
        
        newClip = clip.fl_image( self.crop ) 
        return newClip 
    
    
    
    @counterFunction
    def crop (self,frame):

        self.frameCounter = self.crop.count
        #print self.frameCounter
        windowSize=(2*640,2*360)        
        newFrames = np.zeros((windowSize[0],windowSize[1],3))
        
        if self.frameCounter <= self.numFrames:
            
            imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            x1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter)-1][1] - windowSize[1]/2)
            y1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter)-1][0] - windowSize[0]/2)
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
            #print x1, y1 , x2, y2
            newFrames = frame[x1:x2,y1:y2,:]    
            
        return newFrames
    
   
    
class CMT_algorithm_kalman_filter_downsample():



    def __init__(self,inputPath, resizeFactor = 0.5, skip = None):
        self.inputPath = inputPath     # 'The input path.'
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
                clip  = VideoFileClip(self.inputPath,audio=False)
                W,H = clip.size
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
        imResized = cv2.resize(imGray0, (0,0), fx=self.resizeFactor, fy=self.resizeFactor)
        imDraw = np.copy(imResized)

        (tl, br) = cmtutil.get_rect(imDraw)

        print '[speakerTrackering] Using', tl, br, 'as initial bounding box for the speaker'
               
        self.CMT.initialise(imResized, tl, br)

        measuredTrack=np.zeros((self.numFrames+10,2))-1
           
        count=0
        
        while count <= self.numFrames:
            
            status, im = cap.read()
            if not status:
                break
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_resized = cv2.resize(im_gray, (0,0), fx=self.resizeFactor, fy=self.resizeFactor)
            
            tic = time.time()
            self.CMT.process_frame(im_resized)
            toc = time.time()
            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , count)
            #print 1000*(toc-tic)
            if not (math.isnan(self.CMT.center[0]) or math.isnan(self.CMT.center[1])
                or (self.CMT.center[0] <= 0) or (self.CMT.center[1] <= 0)):
                measuredTrack[count,0] = self.CMT.center[0]
                measuredTrack[count,1] = self.CMT.center[1]               
            count += 1 

    
        numMeas=measuredTrack.shape[0]
        markedMeasure=np.ma.masked_less(measuredTrack,0)
        
        # Kalman Filter Parameters
        deltaT = 1.0/clip.fps
        transitionMatrix=[[1,0,deltaT,0],[0,1,0,deltaT],[0,0,1,0],[0,0,0,1]]   #A
        observationMatrix=[[1,0,0,0],[0,1,0,0]]   #C

        xinit = markedMeasure[0,0]
        yinit = markedMeasure[0,1]
        vxinit = markedMeasure[1,0]-markedMeasure[0,0]
        vyinit = markedMeasure[1,1]-markedMeasure[0,1]
        initState = [xinit,yinit,vxinit,vyinit]    #mu0
        initCovariance = 1.0e-3*np.eye(4)          #sigma0
        transistionCov = 1.0e-4*np.eye(4)          #Q
        observationCov = 1.0e-1*np.eye(2)          #R
        kf = KalmanFilter(transition_matrices = transitionMatrix,
            observation_matrices = observationMatrix,
            initial_state_mean = initState,
            initial_state_covariance = initCovariance,
            transition_covariance = transistionCov,
            observation_covariance = observationCov)
        
        self.measuredTrack = measuredTrack
        (self.filteredStateMeans, self.filteredStateCovariances) = kf.filter(markedMeasure)
        (self.filterStateMeanSmooth, self.filterStateCovariancesSmooth) = kf.smooth(markedMeasure)
        
        #np.savetxt((baseName + 'speakerCoordinates_CMT_Kalman.txt'), 
                   #np.hstack((np.asarray(self.filteredStateMeans), np.asarray(self.filteredStateCovariances) ,
                   #np.asarray(self.filterStateMeanSmooth), np.asarray(self.filterStateCovariancesSmooth) )))
        #np.savetxt((baseName + 'speakerCoordinates_CMT.txt'), np.asarray(measuredTrack))
        
        newClip = clip.fl_image( self.crop ) 
        return newClip 
    
    
    
    @counterFunction
    def crop (self,frame):

        self.frameCounter = self.crop.count
        #print self.frameCounter
        windowSize=(2*640,2*360)        
        newFrames = np.zeros((windowSize[0],windowSize[1],3))
        
        if self.frameCounter <= self.numFrames:
            
            imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            x1 = np.floor( (self.filterStateMeanSmooth[(self.frameCounter)-1][1]) * (1.0/self.resizeFactor) - windowSize[1]/2 )
            y1 = np.floor( (self.filterStateMeanSmooth[(self.frameCounter)-1][0]) * (1.0/self.resizeFactor) - windowSize[0]/2 )
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
            #print x1, y1 , x2, y2
            newFrames = frame[x1:x2,y1:y2,:]    
            
        return newFrames
    

class CMT_algorithm_kalman_filter_vertical_mean():



    def __init__(self,inputPath, skip = None):
        self.inputPath = inputPath     # 'The input path.'
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
                clip  = VideoFileClip(self.inputPath,audio=False)
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
        #self.inity = tl[1] - self.CMT.center_to_tl[1]
        measuredTrack=np.zeros((self.numFrames+10,2))-1
        
        
        count =0
        for frame in clip.iter_frames():
            im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            self.CMT.process_frame(im_gray)

            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , count)
            #print self.inity
            if not (math.isnan(self.CMT.center[0]) or (self.CMT.center[0] <= 0)):
                measuredTrack[count,0] = self.CMT.center[0]
                measuredTrack[count,1] = self.CMT.center[1]               
            count += 1 

    
        numMeas=measuredTrack.shape[0]
        markedMeasure=np.ma.masked_less(measuredTrack,0)
        
        # Kalman Filter Parameters
        deltaT = 1.0/clip.fps
        transitionMatrix=[[1,0,deltaT,0],[0,1,0,deltaT],[0,0,1,0],[0,0,0,1]]   #A
        observationMatrix=[[1,0,0,0],[0,1,0,0]]   #C

        xinit = markedMeasure[0,0]
        yinit = markedMeasure[0,1]
        vxinit = markedMeasure[1,0]-markedMeasure[0,0]
        vyinit = markedMeasure[1,1]-markedMeasure[0,1]
        initState = [xinit,yinit,vxinit,vyinit]    #mu0
        initCovariance = 1.0e-3*np.eye(4)          #sigma0
        transistionCov = 1.0e-4*np.eye(4)          #Q
        observationCov = 1.0e-1*np.eye(2)          #R
        kf = KalmanFilter(transition_matrices = transitionMatrix,
            observation_matrices = observationMatrix,
            initial_state_mean = initState,
            initial_state_covariance = initCovariance,
            transition_covariance = transistionCov,
            observation_covariance = observationCov)
        
        self.measuredTrack = measuredTrack
        (self.filteredStateMeans, self.filteredStateCovariances) = kf.filter(markedMeasure)
        (self.filterStateMeanSmooth, self.filterStateCovariancesSmooth) = kf.smooth(markedMeasure)
        self.inity = np.mean(self.filterStateMeanSmooth[:][1], axis=0)
        newClip = clip.fl_image( self.crop ) 
        return newClip 
    
    
    
    @counterFunction
    def crop (self,frame):

        self.frameCounter = self.crop.count
        #print self.frameCounter
        windowSize=(2*640,2*360)        
        newFrames = np.zeros((windowSize[0],windowSize[1],3))
        
        if self.frameCounter <= self.numFrames:
            
            imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            y1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter)-1][0] - windowSize[1]/2)
            x1 = np.floor(self.inity - windowSize[0]/2)
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
            #print x1, y1 , x2, y2
            newFrames = frame[x1:x2,y1:y2,:]    
            
        return newFrames    

    

class CMT_algorithm_kalman_filter_neighbering():



    def __init__(self,inputPath, skip = None):
        self.inputPath = inputPath     # 'The input path.'
        self.skip = skip            # 'Skip the first n frames.'
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
                clip  = VideoFileClip(self.inputPath,audio=False)
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
                self.numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                print "[speakerTracker] Number of frames" , self.numFrames
                pathBase = os.path.basename(self.inputPath)
                pathDirectory = os.path.dirname(self.inputPath)
                baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_' 
                
                
                
                #Skip first frames if required
                if self.skip is not None:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip)

            # Otherwise assume it is a format string for reading images
            else:
                cap = cmtutil.FileVideoCapture(self.inputPath)

                #Skip first frames if required
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
        tl =  (tl0[0] - self.originFromMainImageX , tl0[1] - self.originFromMainImageY)
        br =  (br0[0] - self.originFromMainImageX , br0[1] - self.originFromMainImageY)
        #print '[speakerTracker] Using', tl, br, 'as initial bounding box for the speaker'
        
        # initialization and keypoint calculation
        self.CMT.initialise(imGray0_initial, tl, br)
        
        # Center of object in cropped frame
        self.currentY = tl[1] - self.CMT.center_to_tl[1]
        self.currentX = tl[0] - self.CMT.center_to_tl[0]
        
        # Center of object in main frame
        self.currentYMainImage = self.currentY + self.originFromMainImageY
        self.currentXMainImage = self.currentX + self.originFromMainImageX
        

        measuredTrack=np.zeros((self.numFrames+10,2))-1    
        count =0
        
        
        # loop to read all frames, 
        # crop them with the center of last frame, 
        # calculate keypoints and center of the object
        
        
        for frame in clip.iter_frames():
            
            # Read the frame and convert it to gray scale
            im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Corner correction (Height)
            if (self.currentYMainImage + marginPixels >= im_gray.shape[0]):                 
                self.currentYMainImage = im_gray.shape[0] - marginPixels -1
            else:
                self.currentYMainImage = self.currentYMainImage
                                    
            if (self.currentXMainImage + marginPixels >= im_gray.shape[1]):                 
                self.currentXMainImage = im_gray.shape[1] - marginPixels -1   
            else:
                self.currentXMainImage = self.currentXMainImage
                
            if (self.currentYMainImage - marginPixels <= 0):                 
                self.currentYMainImage = 0 + marginPixels +1
            else:
                self.currentYMainImage = self.currentYMainImage
                
            if (self.currentXMainImage - marginPixels <= 0):                 
                self.currentXMainImage = 0 + marginPixels +1   
            else:
                self.currentXMainImage = self.currentXMainImage
            
            
            # Crop it by previous coordinates      
            im_gray_crop = im_gray[self.currentYMainImage - marginPixels : self.currentYMainImage + marginPixels,
                                   self.currentXMainImage - marginPixels : self.currentXMainImage + marginPixels]
            
            #plt.imshow(im_gray_crop, cmap = cm.Greys_r)
            #plt.show() 
            
            #print "self.currentYMainImage:", self.currentYMainImage
            #print "self.currentXMainImage:", self.currentXMainImage
            #print im_gray_crop.shape
            
            # Compute all keypoints in the cropped frame
            self.CMT.process_frame(im_gray_crop) 
            #print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.CMT.center[0], self.CMT.center[1] , count)
            
            
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
                measuredTrack[count,0] = self.currentYMainImage
                measuredTrack[count,1] = self.currentXMainImage
                
            else:
                self.currentYMainImage = self.currentYMainImage
                self.currentXMainImage = self.currentXMainImage
            

            
            print 'frame: {2:4d}, Center: {0:.2f},{1:.2f}'.format(self.currentYMainImage, self.currentXMainImage , count)
            count += 1 
 
        numMeas=measuredTrack.shape[0]
        markedMeasure=np.ma.masked_less(measuredTrack,0)
        
        # Kalman Filter Parameters
        deltaT = 1.0/clip.fps
        transitionMatrix=[[1,0,deltaT,0],[0,1,0,deltaT],[0,0,1,0],[0,0,0,1]]   #A
        observationMatrix=[[1,0,0,0],[0,1,0,0]]   #C

        xinit = markedMeasure[0,0]
        yinit = markedMeasure[0,1]
        vxinit = markedMeasure[1,0]-markedMeasure[0,0]
        vyinit = markedMeasure[1,1]-markedMeasure[0,1]
        initState = [xinit,yinit,vxinit,vyinit]    #mu0
        initCovariance = 1.0e-3*np.eye(4)          #sigma0
        transistionCov = 1.0e-4*np.eye(4)          #Q
        observationCov = 1.0e-1*np.eye(2)          #R
        
        # Kalman Filter bias
        kf = KalmanFilter(transition_matrices = transitionMatrix,
            observation_matrices = observationMatrix,
            initial_state_mean = initState,
            initial_state_covariance = initCovariance,
            transition_covariance = transistionCov,
            observation_covariance = observationCov)
        
        self.measuredTrack = measuredTrack
        # Kalman Filter
        (self.filteredStateMeans, self.filteredStateCovariances) = kf.filter(markedMeasure)
        # Kalman Smoother
        (self.filterStateMeanSmooth, self.filterStateCovariancesSmooth) = kf.smooth(markedMeasure)
        newClip = clip.fl_image( self.crop ) 
        return newClip 
        
    
    
    @counterFunction
    def crop (self,frame):

        self.frameCounter = self.crop.count
        #print self.frameCounter
        windowSize=(2*640,2*360)        
        newFrames = np.zeros((windowSize[0],windowSize[1],3))
        
        if self.frameCounter <= self.numFrames:
            
            imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Use Kalman Filter Smoother results to crop the frames with corresponding window size
            x1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter)-1][0] - windowSize[1]/2)
            y1 = np.floor(self.filterStateMeanSmooth[(self.frameCounter)-1][1] - windowSize[0]/2)
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
            #print x1, y1 , x2, y2
            newFrames = frame[x1:x2,y1:y2,:]    
            
        return newFrames

    
    
if __name__ == '__main__':


    '''
    targetVideo = "/media/pbahar/Data Raid/Videos/18.03.2015/video_13.mp4"
    obj = CMT_algorithm_kalman_filter(targetVideo)   
    new_clip = obj.speakerTracker()
    new_clip.write_videofile("speaker_original_13.mp4")
    '''
    
    targetVideo = "/media/pbahar/Data Raid/Videos/18.03.2015/video_6.mp4"
    obj = CMT_algorithm_kalman_filter(targetVideo)   
    new_clip = obj.speakerTracker()    
    new_clip.write_videofile("video_CMT_algorithm_kalman_filter.mp4")
    

 

