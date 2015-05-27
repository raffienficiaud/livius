

'''
This scripts serves as a demo to test the whole Livius framework.
It creates all required instances, calls methods, adjusts the desired values and
forms the whole framework. Therefore, the only things to do is to specify the following input argumnets 
and get the final video layout to stream it on the web.

Input needed:
	1. path to your video file
    2. path to the background image
    3. path to the final image
	4. any input arguments which you need to modify (please use "--help" to check all possibilities.)

output: a video with your desired layout

Description: At first, by specifying the compelete path to your input file, it applies "Moviepy" 
modules to extract audio and video files. Then, it uses two packages named "audio" and "video" to 
process and modify the audio and video signals respectively. At the end, by means of video editing
subpackage, it attaches the improved version of audio file into the modified version of the video files.

Example:

python demo.py ../video_5.mp4  
../background_image.png 
../final_image.png 
--t 1 
--talkInfo 'How to estimate missing data' 
--speakerInfo 'Prof. Dr. Schmidt' 
--instituteInfo 'Empirical Inference' 
--dateInfo 'August 23th 2015'

'''

#--------------------------------------------------------------------
# Import basic packages and modules
#--------------------------------------------------------------------
 
from moviepy.editor import *
from moviepy.Clip import *
import argparse
import os.path
import sys
import time


#--------------------------------------------------------------------
# Import our predefined modules and packages 
#--------------------------------------------------------------------

from util.tools import *
from video.processing.slideDetection import *
from video.editing.layout import createFinalVideo
from video.processing.postProcessing import transformation3D
from video.processing.speakerTracking import *

#--------------------------------------------------------------------
# Parsing the input arguments
#--------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Welcome to Livius Project.')
parser.add_argument('pathToVideoFile', nargs='?', 
                    help='The input path to video file.')
parser.add_argument('pathToBackgroundImage', nargs='?', 
                    help='The input path to background image.')
parser.add_argument('pathToFinalImage', nargs='?', 
                    help='The input path to final image for sponsers and team members.')
parser.add_argument('--t', dest='timeFrame', default=20,
                    help='Specify the time to select the frame for cropping purpose.')
parser.add_argument('--speakerInfo', dest='speakerInfo', default=None, 
                    help='enter the speaker\' name inside the quotation marks')
parser.add_argument('--talkInfo', dest='talkInfo', default=None, 
                    help='enter the tile of talk inside the quotation marks')
parser.add_argument('--instituteInfo', dest='instituteInfo', default=None, 
                    help='enter the name of institute.')
parser.add_argument('--dateInfo', dest='dateInfo', default='July 2015', 
                    help='enter the date of the talk. default= (July 2015)')
parser.add_argument('--sizeOfLayout', dest='sizeOfLayout', default=(1920, 1080), 
                    help='the video layout size for final broadcasting. default=(1920, 1080)')
parser.add_argument('--sizeOfScreen', dest='sizeOfScreen', default=(1280, 960), 
                    help='the size of the screen portion in final layout. default=(1280, 960)')
parser.add_argument('--sizeOfSpeaker', dest='sizeOfSpeaker', default=(620, 360), 
                    help='the size of the speaker portion in final layout. default=(620, 360)')
parser.add_argument('--skip', dest='skip', action='store', default=None, 
                    help='Skip the first n frames', type=int)


#--------------------------------------------------------------------
# Getting the user arguments
#--------------------------------------------------------------------

args = parser.parse_args()

pathToFile = args.pathToVideoFile
if not pathToFile:
    sys.exit("[Demo] Error: You have not specified the path of input file.")

pathToBackgroundImage = args.pathToBackgroundImage
if not pathToBackgroundImage:
    sys.exit("[Demo] Error: You have not specified the input path to the background image.")
    
pathToFinalImage = args.pathToFinalImage
if not pathToFinalImage:
    sys.exit("[Demo] Error: You have not specified the input path to the final image.")
    
pathBase = os.path.basename(pathToFile)
pathDirectory = os.path.dirname(pathToFile)
baseName = os.path.join(pathDirectory, os.path.splitext(pathBase)[0] + '_')
    
speakerInfo      = args.speakerInfo
talkInfo         = args.talkInfo
instituteInfo    = args.instituteInfo
dateInfo         = args.dateInfo
sizeOfLayout     = args.sizeOfLayout
sizeOfSpeaker    = args.sizeOfSpeaker
sizeOfScreen     = args.sizeOfScreen
timeFrame        = args.timeFrame
skip             = args.skip

# Reading the video file - video is a moviepy video object
video = VideoFileClip(pathToFile,audio=False)
# Reading the audio file - audio is a moviepy audio object
audio = AudioFileClip(pathToFile) 


if speakerInfo is None:
    print "[Demo] Warning: You do not specify any speaker information. It is replaced by \"Max Planck Institute for Intelligent systems.\""
    speakerInfo = 'Max Planck Institute for Intelligent systems'
    

if talkInfo is None:
    print "[Demo] Warning: You do not specify any talk information. It is replaced by spaces."
    talkInfo = ' '
    
if instituteInfo is None:
    print "[Demo] Warning: You do not specify any institute name. It is replaced by Empirical Inference."
    instituteInfo = 'Empirical Inference'

    

#--------------------------------------------------------------------
# Prossesing the video for slide detection
#--------------------------------------------------------------------

# Getting a frame to pop-up in t = timeFrame, if timeFrame is not specified, it pops up the first frame

desiredFrame = video.get_frame(float(timeFrame)) 

# Create an object from "getUserCropping" class for slide detection
# If you need to use other classes, simply change this line
objGetUserCropping = getUserCropping(desiredFrame)  
# Call method "slideDetector" to get four corners - slideCoordinates is a 4x2 numpy array
# with the order of [TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT ]
slideCoordinates = objGetUserCropping.slideDetector()

print "[Demo] Info: The selected coordinates of the slide are: \n" , slideCoordinates

# Save the coordinates in a .txt file

saveNameSlide = baseName + 'slide_coordinates' + '.txt'

if os.path.isfile(saveNameSlide):
    if (prompt_yes_no_terminal("[Demo] Warning: The file to store the coordinates exists. Replace?")):
        np.savetxt(saveNameSlide, slideCoordinates)
    else:
        print "[Demo] Warning: No new file was created."
else:
    np.savetxt(saveNameSlide, slideCoordinates)

    
#--------------------------------------------------------------------
# Post-Prossesing the video for slide detection and stream writing
#--------------------------------------------------------------------

# Call "transformation3D" from postProcesing module for perspective transformation
# video.fx and fl_image are predefined moviepy functions to do that "streaming". \
# you only have one frame at once in the RAM.
slideClip = video.fx(transformation3D, slideCoordinates ,(1280, 960))
    
       
#--------------------------------------------------------------------
# Prossing the video for speaker tracking
#--------------------------------------------------------------------

# Create an object from "CMT_algorithm_kalman_filter" class for speaker tracking
# If you need to use other classes, simply change this line
objCMTAlgorithm = CMT_algorithm_kalman_filter(pathToFile,skip)  
# Call method "speakerTracker" to get four corners - slideCoordinates is a 4x2 numpy array
# output is a single frame and it is updated. It is using streaming function to have one frame at once in the RAM.
speakerClip = objCMTAlgorithm.speakerTracker()

#--------------------------------------------------------------------
# Prossing the audio file
#--------------------------------------------------------------------

    
    
#--------------------------------------------------------------------
# editing the video and audio and forming the final layout
#--------------------------------------------------------------------

slideClip = video.fx(transformation3D, slideCoordinates ,(1280, 960))

nameToSaveFile = baseName +  'output'

# call "createFinalVideo" method from "layout.py" module in editing package to form the final layout 
# and concatenate all required information and files together. 
# If "flagWrite" sets to True, the output video will be written in he same path of the input file.
# You may modify the input arguments as you intend.
createFinalVideo(slideClip,speakerClip,
                        pathToBackgroundImage,
                        pathToFinalImage,
                        audio,
                        fps = video.fps, 
                        sizeOfLayout = sizeOfLayout, 
                        sizeOfScreen = sizeOfScreen,
                        sizeOfSpeaker = sizeOfSpeaker,
                        talkInfo = talkInfo,
                        speakerInfo = speakerInfo,
                        instituteInfo = instituteInfo,
                        dateInfo = dateInfo,
                        firstPause = 10,
                        nameToSaveFile = nameToSaveFile,
                        codecFormat = 'libx264',
                        container = '.mp4',
                        flagWrite = True)    
    
    
