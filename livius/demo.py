

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
from livius.video.processing.vault.postProcessing import PostProcessor
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
# Processing the video for slide detection
#--------------------------------------------------------------------

# Getting a frame to pop-up in t = timeFrame, if timeFrame is not specified, it pops up the first frame

# desiredFrame = video.get_frame(float(timeFrame))

# # Create an object from "getUserCropping" class for slide detection
# # If you need to use other classes, simply change this line
# objGetUserCropping = getUserCropping(desiredFrame)
# # Call method "slideDetector" to get four corners - slideCoordinates is a 4x2 numpy array
# # with the order of [TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT ]
# slideCoordinates = objGetUserCropping.slideDetector()

# print "[Demo] Info: The selected coordinates of the slide are: \n" , slideCoordinates

# # Save the coordinates in a .txt file

saveNameSlide = baseName + 'slide_coordinates' + '.txt'

# if os.path.isfile(saveNameSlide):
#     if (prompt_yes_no_terminal("[Demo] Warning: The file to store the coordinates exists. Replace?")):
#         np.savetxt(saveNameSlide, slideCoordinates)
#     else:
#         print "[Demo] Warning: No new file was created."
# else:
#     np.savetxt(saveNameSlide, slideCoordinates)


# Bypassing the User cropping, just loading the slide coordinates from file
slideCoordinates = np.loadtxt(saveNameSlide, dtype=float32)

slideCoordinates_01 = np.copy(slideCoordinates)
slideCoordinates_01[:,0] /= video.size[0]
slideCoordinates_01[:,1] /= video.size[1]

print slideCoordinates_01

#--------------------------------------------------------------------
# Processing the video for speaker tracking
#--------------------------------------------------------------------

# # Create an object from "CMT_algorithm_kalman_filter" class for speaker tracking
# # If you need to use other classes, simply change this line
# objCMTAlgorithm = CMT_algorithm_kalman_filter(pathToFile,skip)
# # Call method "speakerTracker" to get four corners - slideCoordinates is a 4x2 numpy array
# # output is a single frame and it is updated. It is using streaming function to have one frame at once in the RAM.
# speakerClip = objCMTAlgorithm.speakerTracker()


speakerClip = None

#--------------------------------------------------------------------
# Processing the audio file
#--------------------------------------------------------------------


#--------------------------------------------------------------------
# Post-Processing the video for slide detection and stream writing
#--------------------------------------------------------------------


histogram_correlations, histogram_boundaries = read_histogram_correlations_and_boundaries_from_json_file('Example Data/video7_full_stripe.json', slide_stripe=0)


# @todo(Stephan): Hardcoded for video7 right now
# histogram_correlations = [0.9993370923799508, 0.9982369732081567, 0.8274766510760692, 0.9981254542217526, 0.9992443868217512, 0.9985528938092855, 0.998755704784441,
#                           0.9993968772316724, 0.9994575939439109, 0.9984760901017141, 0.9986031857817385, 0.9992463331766867, 0.9970439821096971, 0.9990769486372814,
#                           0.9976950561842186, 0.9992922369022539, 0.999493045787124, 0.999250076595509, 0.9996035924607937, 0.9995645644267759, 0.9987091577872489,
#                           0.9993047353415527, 0.9958824028190917, 0.9985588212562849, 0.9990434612642667, 0.9995674628568643, 0.999332696860954, 0.9993063000315259,
#                           0.9996031383099585, 0.9996189405773498, 0.9970453030880848, 0.9993458870019144, 0.9996109306985971, 0.9995837493216747, 0.9994109553248358,
#                           0.9997380311868116, 0.9994452559290463, 0.999415374735256, 0.9954753347159024, 0.9980777929872854, 0.9997567302409215, 0.9989969156562254,
#                           0.9991770096161414, 0.9996354970207004, 0.9994027625371246, 0.9995877709046798, 0.9997053844431577, 0.996627344649838, 0.9995042727242337,
#                           0.9997210334216253, 0.9994610828491993, 0.999299613618139, 0.9996237620046838, 0.9995857091278242, 0.9996933577290982, 0.9958667032476024,
#                           0.999344900375841, 0.9996040773234843, 0.9997683631294824, 0.9996325223289707]

# histogram_boundaries = [(52, 143), (52, 143), (52, 143), (52, 136), (52, 136), (52, 136), (52, 136), (52, 136),
#                         (52, 136), (52, 136), (52, 136), (52, 136), (52, 136), (51, 136), (52, 136), (52, 136),
#                         (52, 136), (52, 136), (52, 136), (52, 136), (52, 136), (52, 136), (52, 136), (52, 136),
#                         (52, 136), (52, 136), (52, 136), (52, 136), (52, 136), (52, 136), (52, 136), (52, 136),
#                         (52, 136), (52, 136), (52, 136), (52, 136), (52, 136), (52, 136), (51, 136), (51, 136),
#                         (51, 136), (51, 136), (51, 136), (51, 136), (52, 136), (51, 136), (52, 136), (52, 136),
#                         (51, 136), (51, 136), (51, 136), (51, 136), (51, 136), (51, 136), (51, 136), (51, 136),
#                         (51, 136), (51, 136), (51, 136), (51, 136), (51, 136)]

desiredScreenLayout = (1280, 960)

postProcessor = PostProcessor(video, slideCoordinates, desiredScreenLayout, histogram_correlations, histogram_boundaries)

slideClip = postProcessor.process()

#--------------------------------------------------------------------
# editing the video and audio and forming the final layout
#--------------------------------------------------------------------
nameToSaveFile = baseName +  'output_contrast_enhanced'

# Write slideclip to disk
slideClip.write_videofile(nameToSaveFile + '.mp4')

# Just write individual frames for testing
slideClip.save_frame(baseName + "0.png", t=0)
slideClip.save_frame(baseName + "1.png", t=0.5)
slideClip.save_frame(baseName + "2.png", t=1)
slideClip.save_frame(baseName + "3.png", t=1.5)
slideClip.save_frame(baseName + "4.png", t=2)
slideClip.save_frame(baseName + "5.png", t=2.5)
slideClip.save_frame(baseName + "6.png", t=3)
slideClip.save_frame(baseName + "7.png", t=3.5)
slideClip.save_frame(baseName + "8.png", t=4)


# call "createFinalVideo" method from "layout.py" module in editing package to form the final layout
# and concatenate all required information and files together.
# If "flagWrite" sets to True, the output video will be written in he same path of the input file.
# You may modify the input arguments as you intend.
# createFinalVideo(slideClip,speakerClip,
#                         pathToBackgroundImage,
#                         pathToFinalImage,
#                         audio,
#                         fps = video.fps,
#                         sizeOfLayout = sizeOfLayout,
#                         sizeOfScreen = sizeOfScreen,
#                         sizeOfSpeaker = sizeOfSpeaker,
#                         talkInfo = talkInfo,
#                         speakerInfo = speakerInfo,
#                         instituteInfo = instituteInfo,
#                         dateInfo = dateInfo,
#                         firstPause = 10,
#                         nameToSaveFile = nameToSaveFile,
#                         codecFormat = 'libx264',
#                         container = '.mp4',
#                         flagWrite = True)


