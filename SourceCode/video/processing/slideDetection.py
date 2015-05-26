
"""
This module implements various algorithms for slide detection. For each algorithm, 
one class has been defined in which there is a method named "slideDetector" to call.
Different classes have different attributes, but in order to call the corresponding
function "slideDetector", all classes are the same.
classes:
- templateMatch
- connectedComponent
- polygonDetection
- harrisCornerDetection
- getUserCropping (use one in the Demo)
"""

import numpy as np
import os 
import math

from moviepy.editor import *
from moviepy.Clip import *
from moviepy.video.VideoClip import *
from moviepy.config import get_setting # ffmpeg, ffmpeg.exe, etc...
import matplotlib.pyplot as plt
from pylab import *
from scipy.ndimage.measurements import label

import cv2
from util.tools import *
from util.templateMatching import templateMatching as tm
from video.processing.postProcessing import transformation3D



        
class templateMatch():
    
    # documentation string, which can be accessed via ClassName.__doc__ 
    """ This class can be served to find the corners by means of four filter images. 
    To this end, one has to specify the filter size and the corresponding method (either SSD or NCC)
    to compute the template matching algorithm."""
    
    '''
    Inputs:
    
        inputImage: The gray-level image
        filterSize: The size of filters for matching (eX: 50x50 -> (50,50)) - It is a list with two values
        method: a string to specify the method. default: NCC (Normalized Cross Correlation)
    
    Outputs:
        A 4x2 numpy array including four pair of points
    
    '''
    
    def __init__(self,inputImage, filterSize, method = 'NCC'):
        self.inputImage = inputImage
        self.filterSize = filterSize
        # Method can be either 'SSD' or 'NCC'
        self.method = method 
        
        
    def slideDetector(self):
                       
        # Get the edge filter as a template
        # Four Different edgeFilters
        '''
        1. LT -> Left Top
        2. RT -> Right Top
        3. LB -> Left Bottom
        4. RB -> Right Bottom
        '''
        size = self.filterSize
        image = self.inputImage
        edgeFilterSizeX = size[0]
        edgeFilterSizeY = size[1]

        edgeFilterLT = -1*(np.ones((edgeFilterSizeX,edgeFilterSizeY), dtype=int))
        edgeFilterLT[edgeFilterSizeX/2:edgeFilterSizeX, edgeFilterSizeY/2:edgeFilterSizeY] = 1

        edgeFilterRT = -1*(np.ones((edgeFilterSizeX,edgeFilterSizeY), dtype=int))
        edgeFilterRT[edgeFilterSizeX/2:edgeFilterSizeX, 0:edgeFilterSizeY/2] = 1

        edgeFilterLB = -1*(np.ones((edgeFilterSizeX,edgeFilterSizeY), dtype=int))
        edgeFilterLB[0:edgeFilterSizeX/2, edgeFilterSizeY/2:edgeFilterSizeY] = 1

        edgeFilterRB = -1*(np.ones((edgeFilterSizeX,edgeFilterSizeY), dtype=int))
        edgeFilterRB[0:edgeFilterSizeX/2, 0:edgeFilterSizeY/2] = 1

        if (self.method=='SSD'):
            
            heigthLT, widthLT = edgeFilterLT.shape
            [resultSSDLT,resultNCCLT,IdataLT,TdataLT] = tm.templateMatching(image,edgeFilterLT, type = "SSD")
            # To find the place of maximum
            [peakY_SSDLT, peakX_SSDLT]  = np.unravel_index(np.argmax(resultSSDLT), resultSDDLT.shape)

            heigthRT, widthRT = edgeFilterRT.shape
            [resultSSDRT,resultNCCRT,IdataRT,TdataRT] = tm.templateMatching(image,edgeFilterRT, type = "SSD")
            # To find the place of maximum 
            [peakY_SSDRT, peakX_SSDRT]  = np.unravel_index(np.argmax(resultSSDRT), resultSDDRT.shape)

            heigthLB, widthLB = edgeFilterLB.shape
            [resultSSDLB,resultNCCLB,IdataLB,TdataLB] = tm.templateMatching(image,edgeFilterLB, type = "SSD")
            # To find the place of maximum 
            [peakY_SSDLB, peakX_SSDLB]  = np.unravel_index(np.argmax(resultSSDLB), resultSDDLB.shape)

            heigthRB, widthRB = edgeFilterRB.shape
            [resultSSDRB,resultNCCRB,IdataRB,TdataRB] = tm.templateMatching(image,edgeFilterRB, type = "SSD")
            # To find the place of maximum 
            [peakY_SSDRB, peakX_SSDRB]  = np.unravel_index(np.argmax(resultSSDRB), resultSDDRB.shape)
        
                        
            # Create a 4x2 numpy array
            pointsSSD = np.array([[ peakY_SSDLT, peakX_SSDLT], 
                                  [ peakY_SSDRT, peakX_SSDRT], 
                                  [ peakY_SSDLB, peakX_SSDLB], 
                                  [ peakY_SSDRB, peakX_SSDRB]])
            
            # Return the value in order
            return rectify_coordinates(pointsSSD)
    
    
        if (self.method=='NCC'):
    
            heigthLT, widthLT = edgeFilterLT.shape
            [resultSSDLT,resultNCCLT,IdataLT,TdataLT] = tm.templateMatching(image,edgeFilterLT, type = "NCC")
            # To find the place of maximum 
            [peakY_NCCLT, peakX_NCCLT]  = np.unravel_index(np.argmax(resultNCCLT), resultNCCLT.shape)

            heigthRT, widthRT = edgeFilterRT.shape
            [resultSSDRT,resultNCCRT,IdataRT,TdataRT] = tm.templateMatching(image,edgeFilterRT, type = "NCC")
            # To find the place of maximum 
            [peakY_NCCRT, peakX_NCCRT]  = np.unravel_index(np.argmax(resultNCCRT), resultNCCRT.shape)

            heigthLB, widthLB = edgeFilterLB.shape
            [resultSSDLB,resultNCCLB,IdataLB,TdataLB] = tm.templateMatching(image,edgeFilterLB, type = "NCC")
            # To find the place of maximum 
            [peakY_NCCLB, peakX_NCCLB]  = np.unravel_index(np.argmax(resultNCCLB), resultNCCLB.shape)

            heigthRB, widthRB = edgeFilterRB.shape
            [resultSSDRB,resultNCCRB,IdataRB,TdataRB] = tm.templateMatching(image,edgeFilterRB, type = "NCC")
            # To find the place of maximum 
            [peakY_NCCRB, peakX_NCCRB]  = np.unravel_index(np.argmax(resultNCCRB), resultNCCRB.shape)
            
            # Create a 4x2 numpy array
            pointsNCC = np.array([[ peakY_NCCLT, peakX_NCCLT], 
                                  [ peakY_NCCRT, peakX_NCCRT], 
                                  [ peakY_NCCLB, peakX_NCCLB], 
                                  [ peakY_NCCRB, peakX_NCCRB]])
            
            # Return the value in order
            return rectify_coordinates(pointsNCC)
        
        
class connectedComponent():
    
    
    """ This class can be served to find the corners by means of connecting all components in 
    black and white image."""
    
    '''
    Inputs:
    
        inputImage: The gray-level image
    
    Outputs:
        A 4x2 numpy array including four pair of points
    
    '''
    
    def __init__(self, inputImage):       
        self.inputImage = inputImage
                
    
    def slideDetector(self):
        
        image = self.inputImage
        img = np.asarray(image)
        imgNorm = img/float(np.amax(img)) 

        # Let numpy do the heavy lifting for converting pixels to pure black or white
        bw = np.asarray(imgNorm).copy()

        # Pixel range is 0...255, converting to bw by 0.4 threshold ratio
        bw[bw < 0.4] = 0    # Black
        bw[bw >= 0.4] = 1 # White

        # bw labeling
        labeledArray, numFeatures = label(bw)
        
        # proning out the connected area with area smaller than 10000 pixels
        areaPixel = 10000
        for n in range (1, numFeatures):
            idx1, idx2 = np.where(labeledArray==n)
            if idx1.size < areaPixel:
                bw[labeledArray == n] = 0
        
        
        # bw labeling
        labeledArrayProned, numFeaturesProned = label(bw)
        
        # CornerCoordinates = [x1 y1 x2 y2]
        CornerCoordinates = np.zeros((numFeaturesProned,5), dtype=float)        
        screenCoordinate = np.zeros((4,2), dtype=float)
        for n in range (0, numFeaturesProned):
            idx1, idx2 = np.where(labeledArrayProned==n)
            CornerCoordinates[n,0] = np.amin(idx2)
            CornerCoordinates[n,1] = np.amin(idx1)
            CornerCoordinates[n,2] = np.amax(idx2)
            CornerCoordinates[n,3] = np.amax(idx1)
            CornerCoordinates[n,4] = idx1.size
  
        idx = np.where(CornerCoordinates[1:,4]==np.amax(CornerCoordinates[1:,4]))
        pol = idx[0][0]
        screenCoordinate[0,:] = [CornerCoordinates[pol+1,0],CornerCoordinates[pol+1,1]]
        screenCoordinate[1,:] = [CornerCoordinates[pol+1,0],CornerCoordinates[pol+1,3]]
        screenCoordinate[2,:] = [CornerCoordinates[pol+1,2],CornerCoordinates[pol+1,1]]
        screenCoordinate[3,:] = [CornerCoordinates[pol+1,2],CornerCoordinates[pol+1,3]]
        
        return rectify_coordinates(screenCoordinate)
        
        

class polygonDetection():
    
    
    """ This class can be served to find the corners by means of canny edge detection and finding contours.
    """
    
    '''
    Inputs:
    
        inputImage: The gray-level image
    
    Outputs:
        A 4x2 numpy array including four pair of points
    
    '''
    
    def __init__(self, inputImage):
        self.inputImage = inputImage
        
    def slideDetector(self):
        
        image = self.inputImage
        gray = cv2.bilateralFilter(image, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        # find contours in the edged image, keep only the largest
        # ones, and initialize our screen contour
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None
        
        
        counter =0
        for i in range(10):
            box = contours[i]
            peri = cv2.arcLength(box,True)
            approx = cv2.approxPolyDP(box,0.01*peri,True)  # Different shapes have different approximations (rectangle has to have 4 coordinates)
            if len(approx) > 2:
                rect = cv2.minAreaRect(contours[i])
                r = cv2.cv.BoxPoints(rect)
                d = rectify_coordinates(np.asarray(r))
        
                if (d[0][0] < (image.shape[0])/2) and (d[1][0] > (image.shape[0])/2) and (np.abs(d[0][0]-d[1][0]) > 500):
                    screenCnt = d

        # loop over our contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            '''
            if len(approx) == 4:
                screenCnt = approx
                break
            '''
            if len(approx) > 2:
                rect = cv2.minAreaRect(c[i])
                r = cv2.cv.BoxPoints(rect)
                d = tools.rectify_coordinates(np.asarray(r))
        
                if (d[0][0] < W/2) and (d[1][0] > W/2) and (np.abs(d[0][0]-d[1][0]) > 100):
                    screenCnt = d
                
        return rectify_coordinates(screenCnt.reshape(4,2))

    
class harrisCornerDetection():

    # documentation string, which can be accessed via ClassName.__doc__ 
    """
        This function serves for Haris Corner Detector
        Inputs:
 
            inputImage: The gray-level image.
            blockSize:  Neighborhood size (see the details on cornerEigenValsAndVecs() ).
            kernelSize: Aperture parameter for the Sobel() operator.
            k: Harris detector free parameter. See the formula below.
            borderType: Pixel extrapolation method. See borderInterpolate() .
        Outputs:
            dst: Image to store the Harris detector responses. It has the type CV_32FC1 and the same size as inputImage .

        Example:
        
        harris_corner_detection(inputImage, blockSize=31, kernelSize=3, k=0.04,thersh=0.01, borderType)
        
    """
      
    
    def __init__(self, inputImage, blockSize, kernelSize, k, borderType, thersh, flagShow):       
        self.inputImage = inputImage
        self.blockSize = blockSize
        self.kernelSize = kernelSize 
        self.k = k
        self.borderType = borderType
        self.thresh = thresh
        self.flagShow = flagShow
        
    def slideDetector(self):
        
     
        image = self.inputImage
        grayImage = np.float32(image)
        dst = cv2.cornerHarris(grayImage,self.blockSize,self.kernelSize,self.k, self.borderType)

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        imgc = image.copy()
        imgc[dst > (self.thersh)*dst.max()]=[0,255,0]
        x,y = np.where(dst > (self.thersh)*dst.max())
        arrayHarrisPoints = np.zeros((len(x),2), dtype=float)
        arrayHarrisPoints[:,0] = x
        arrayHarrisPoints[:,1] = y
        
        if self.flagShow :
            plt.imshow(imgc, cmap = cm.Greys_r)
            plt.show()
            
        return arrayHarrisPoints


class getUserCropping():

    # documentation string, which can be accessed via ClassName.__doc__ 
    """ This class can be served for user cropping. After creating an instance of the class and
    call the corresponding method, a window will be open to get the user for points and then return 
    the points back."""
    
    '''
    Inputs:
    
        inputImage: The gray-level image
    
    Outputs:
        A 4x2 numpy array including four pair of points
        
    '''
    
    
    def __init__(self, inputImage):    
        
        self.inputImage = inputImage
    
    def slideDetector(self):
        
        # Pop up the frame
        inputFrame = self.inputImage
        fig = plt.figure()
        plt.imshow(inputFrame)
        plt.title('Please click the four corners of the slide and then close the window.')
        
        # Call the function call connect from class CallbacksPoints and mouse-event
        objectCallbacksPoints = CallbacksPoints(inputFrame.shape[:2])
        objectCallbacksPoints.connect(fig)    
        plt.show()
        selectedPoints = objectCallbacksPoints.get_points()
        
        # Return four pair of points
        return rectify_coordinates(np.array(selectedPoints, dtype=np.float32))



if __name__ == '__main__':
    
    targetVideo = "/media/pbahar/Data Raid/Videos/18.03.2015/newVideo.mp4"
    main_clip = VideoFileClip(targetVideo)

    desiredFrame = main_clip.get_frame(t=20) # output is a numpy array
    file_slide = cv2.cvtColor(desiredFrame,cv2.COLOR_BGR2GRAY)
    obj = getUserCropping(file_slide)
    
    points = obj.slideDetector()
    print points
    
    slideClip = main_clip.fx(transformation3D, points ,(1280, 960))
    slideClip.write_videofile("video_test.mp4")

    
    

    
