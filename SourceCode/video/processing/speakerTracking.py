

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


CMT = CMT.CMT.CMT()

class CMT_algorithm():



    def __init__(self,inputPath, outputDir = None, bBox = None , skip = None):
        self.inputPath = inputPath     # 'The input path.'
        self.outputDir = outputDir      # 'Specify a directory for output data.'
        self.bBox = bBox                # 'Specify initial bounding box.'
        self.skip = skip                # 'Skip the first n frames.'
        

       
    def speakerTracker(self):
        

        # Clean up
        cv2.destroyAllWindows()

        if self.inputPath is not None:

            # If a path to a file was given, assume it is a single video file
            if os.path.isfile(self.inputPath):
                cap = cv2.VideoCapture(self.inputPath)
                clip  = VideoFileClip(self.inputPath)

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
            sys.exit("Error: no input path was specified")

        # Read first frame
        status, im0 = cap.read()
        im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        im_draw = np.copy(im0)

        if self.bBox is not None:
            # Try to disassemble user specified bounding box
            values = self.bBox.split(',')
            try:
                values = [int(v) for v in values]
            except:
                raise Exception('Unable to parse bounding box')
            if len(values) != 4:
                raise Exception('Bounding box must have exactly 4 elements')
            bbox = np.array(values)

            # Convert to point representation, adding singleton dimension
            bbox = cmtutil.bb2pts(bbox[None, :])

            # Squeeze
            bbox = bbox[0, :]

            tl = bbox[:2]
            br = bbox[2:4]
        else:
            # Get rectangle input from user
            (tl, br) = cmtutil.get_rect(im_draw)

        print 'using', tl, br, 'as init bb'

        CMT.initialise(im_gray0, tl, br)
        new_clip = clip.fl_image( self.crop )
                   
        return new_clip 
    
    
    def crop (self,frame):
    
        windowSize = (640,360)
        new_frames = np.zeros((windowSize[0],windowSize[1],3))
        im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        tic = time.time()
        CMT.process_frame(im_gray)
        toc = time.time()

        if not (math.isnan(CMT.center[0]) or math.isnan(CMT.center[1])
                or (CMT.center[0] <= 0) or (CMT.center[1] <= 0)): 
            
            x1 = np.floor(CMT.center[1] - windowSize[1]/2)
            y1 = np.floor(CMT.center[0] - windowSize[0]/2)
            x2 = np.floor(x1 + windowSize[1])
            y2 = np.floor(y1 + windowSize[0])
            
            # Corner correction (Height)
            if (x1 <= 0):                 
                x1 = 0
                x2 = np.floor(x1 + windowSize[1])
            if (x2 >= im_gray.shape[0]):
                x2 = np.floor(im_gray.shape[0])
                x1 = np.floor(x2 - windowSize[1])
            # Corner correction (Width)
            if (y1 <= 0):                 
                y1 = 0
                y2 = np.floor(y1 + windowSize[0])
            if (y2 >= im_gray.shape[1]):
                y2 = np.floor(im_gray.shape[1])
                y1 = np.floor(y2 - windowSize[0])
            new_frames = frame[x1:x2,y1:y2,:]
            
        #print 'Center: {0:.2f},{1:.2f}'.format(CMT.center[0], CMT.center[1])
        return new_frames


    
    
if __name__ == '__main__':
    
    targetVideo = "/media/pbahar/Data Raid/Videos/18.03.2015/video_13.mp4"
    obj = CMT_algorithm(targetVideo)   
    new_clip = obj.speakerTracker()
    new_clip.write_videofile("speakerpart.mp4") 
    
 
    
