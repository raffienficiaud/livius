

'''
This scripts serves aa a demo to test the whole Livius framework.
It creates all required instances, calls methods, adjusts the desired values and
forms the whole framework. Therefore, the only things to do is to specify the following input argumnets 
and get the final video layout to stream it on the web.

Input needed:
	1. path to your video file
	2. any input arguments which you need to modify (please use "--help" to check all possibilities.

output: a video with your desired layout

Description: At first, by specifying the compelete path to your input file, it applies "Moviepy" 
modules to extract audio and video files. Then, it uses to packages named "audio" and "video" to 
process and modify the audio and video signals respectively. At the end, by means of video editing
subpackage, it attaches the improved version of audio file into the modified version of the video files.

'''


from moviepy.editor import *
from moviepy.Clip import *
import argparse
import os.path
import sys
import time


from util.tools import prompt_yes_no_terminal
from video.processing.slideDetection import *
from video.editing.layout import createFinalVideo
from moviepy.video.fx.crop import crop


parser = argparse.ArgumentParser(description='Welcome to Livius Project.')
parser.add_argument('pathToVideoFile', nargs='?', 
                    help='The input path to video file.')
parser.add_argument('pathToBackgroundImage', nargs='?', 
                    help='The input path to background image.')
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
parser.add_argument('--outputDirectory', dest='outputDirectory', 
                    help='Specify a directory for output data.')


#--------------------------------------------------------------------
# getting the user arguments
#--------------------------------------------------------------------

args = parser.parse_args()

if args.outputDirectory is not None:
    if not os.path.exists(args.outputDirectory):
        os.mkdir(args.outputDirectory)
    elif not os.path.isdir(args.outputDirectory):
        raise Exception("[Demo] Error:" + args.outputDirectory + " exists, but is not a directory.")

#pathToFile = "/media/pbahar/Data Raid/Videos/18.03.2015/newVideo.mp4"
pathToFile = args.pathToVideoFile
if not pathToFile:
    sys.exit("[Demo] Error: You have not specified the path of input file.")

#pathToBackgroundImage = "~/Downloads/EdgarFiles/background_images_mlss2013/background_example.png"
pathToBackgroundImage = args.pathToBackgroundImage
if not pathToBackgroundImage:
    sys.exit("[Demo] Error: You have not specified the input path to the background image.")
    


pathBase = os.path.basename(pathToFile)
pathDirectory = os.path.dirname(pathToFile)
baseName = pathDirectory + '/' + os.path.splitext(pathBase)[0] + '_'
    
speakerInfo      = args.speakerInfo
talkInfo         = args.talkInfo
instituteInfo    = args.instituteInfo
dateInfo         = args.dateInfo
sizeOfLayout     = args.sizeOfLayout
sizeOfSpeaker    = args.sizeOfSpeaker
sizeOfScreen     = args.sizeOfScreen
timeFrame        = args.timeFrame

# Reading the video file
video = VideoFileClip(pathToFile,audio=False)
# Reading the audio file
audio = AudioFileClip(pathToFile) 


if speakerInfo is None:
    print "[Demo] Warning: You do not specify any speaker information. It is replaced by \"Max Planck Institute for Intelligent systems.\""
    speakerInfo = 'Max Planck Institute for Intelligent systems'
    

if talkInfo is None:
    print "[Demo] Warning: You do not specify any talk information. It is replaced by spaces."
    talkInfo = ' '
    
if instituteInfo is None:
    print "[Demo] Warning: You do not specify any institute name. It is replaced by spaces."
    instituteInfo = ' '

    

#--------------------------------------------------------------------
# Prossing the video for slide detection
#--------------------------------------------------------------------



desiredFrame = video.get_frame(float(timeFrame)) # output is a numpy array
#file_slide = cv2.cvtColor(desiredFrame,cv2.COLOR_BGR2GRAY)
obj_get_user_cropping = getUserCropping(desiredFrame)  
slideCoordinates = obj_get_user_cropping.slideDetector()
print "[Demo] Info: The selected coordinates are: \n" , slideCoordinates

saveNameSlide = baseName + 'slide_coordinates' + '.txt'

if os.path.isfile(saveNameSlide):
    if (prompt_yes_no_terminal("[Demo] Warning: The file to store the coordinates exists. Replace?")):
        np.savetxt(saveNameSlide, slideCoordinates)
    else:
        print "[Demo] Warning: No new file was created."
else:
    np.savetxt(saveNameSlide, slideCoordinates)

    
#--------------------------------------------------------------------
# Prossing the video for speaker tracking
#--------------------------------------------------------------------

from video.processing.speakerTracking import *

obj_CMT_algorithm = CMT_algorithm(pathToFile)  
speakerClip = obj_CMT_algorithm.speakerTracker()

#targetSpeaker = baseName + 'speaker.mp4'
#speakerFrames.write_videofile(targetSpeaker,fps= video.fps, codec='libx264' ) 

'''

saveNameSpeaker = baseName +  'speaker_coordinates' + '.txt'

if os.path.isfile(saveNameSpeaker):
    if (prompt_yes_no_terminal("[Demo] Warning: The file to store the coordinates exists. Replace?")):
        np.savetxt(saveNameSpeaker, centerpoints)
    else:
        print "[Demo] Warning: No new file was created."
else:
    np.savetxt(saveNameSpeaker, centerpoints)    
'''    

    
#--------------------------------------------------------------------
# Prossing the audio file
#--------------------------------------------------------------------

    
    
#--------------------------------------------------------------------
# editing the video and audio and forming the final layout
#--------------------------------------------------------------------


slideClip = crop(video, x1=slideCoordinates[0][0], y1=slideCoordinates[0][1], 
                        x2=slideCoordinates[2][0], y2=slideCoordinates[2][1])

nameToSaveFile = baseName +  'output'
createFinalVideo(slideClip,speakerClip,
                        pathToBackgroundImage,
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
    
    