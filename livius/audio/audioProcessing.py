# Import Basic modules

import numpy as np
import os 

# Import everything needed to edit video clips
from moviepy.editor import *
from moviepy.Clip import *
from moviepy.video.VideoClip import *
from moviepy.config import get_setting # ffmpeg, ffmpeg.exe, etc...




class AudioProcessing:
    
    # documentation string, which can be accessed via ClassName.__doc__ (slide_detection.__doc__ )
    """ This class include all required attributes and methods for slide detection.
    It includes different algorithms for slide detection such as harris corner detection,
    Histogram thresholding, Hough Transform, sum of differences of all frames and etc.
    The input of the functions is the input image/frame/video and the output is the four
    coordinates of the position of the detected slide.
    Built-In Class Attributes:
    Every Python class keeps following built-in attributes and they can be accessed using
    dot operator like any other attribute:

    __dict__ : Dictionary containing the class's namespace.
    
    __doc__ : Class documentation string or None if undefined.

    __name__: Class name.

    __module__: Module name in which the class is defined. This attribute is "__main__" in interactive mode.
    
    __bases__ : A possibly empty tuple containing the base classes, in the order of their occurrence
    in the base class list."""
    
    
    def __init__(self, inputFile):       
      self.inputFile = inputFile

    
    
    #def template_matching(self):
        
    
        
    def equalizer(self):
        
        '''
        This function serves for Haris Corner Detector
        Inputs:

        Outputs:

        Example:
        
        
        '''
        


    def signal_improvement(self):
        
        '''
        This function serves for sum of the differences of all frames
        Inputs:


        Outputs:


        Example:

        
        '''
        
   
    
    def audio_coding(self, bitrate, codecformat):
        
        '''
        This function serves for max of the differences of all frames
        Inputs:


        Outputs:


        Example:

        
        '''

        
    
    def audio_clip(self):
            
        '''
        This function serves for max of all frames
        Inputs:

        Outputs:


        Example:

        
        '''   
 
        


if __name__ == '__main__':
    
    print "done"
