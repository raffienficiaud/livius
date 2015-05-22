----------------------------------------------------------------------------
Max Planck Institute for intelligent System -- Welcome to the Livius Project
----------------------------------------------------------------------------

The livius directory contains the main parts like:

 - Documentation: It includes the documentaion related to the project.
 - SourceCode: It includes all of the required all Python packages and modules to run the framework.
   I side this folder, there is a Python file named 'demo.py' running the whole framework.
 - Videos: It includes some video samples.
 - background_example: which is a sample of background image.

Please note that:
_________________ 

In order to use the Livius framework, it is highly recommneded to read "documentation.pdf".

In order to use the 'demo.py', simply go to the directory where the SourceCodes located and then type in terminal:

	python demp.py --help

the output will be:

usage: demo.py [-h] [--t TIMEFRAME] [--speakerInfo SPEAKERINFO]
               [--talkInfo TALKINFO] [--instituteInfo INSTITUTEINFO]
               [--dateInfo DATEINFO] [--sizeOfLayout SIZEOFLAYOUT]
               [--sizeOfScreen SIZEOFSCREEN] [--sizeOfSpeaker SIZEOFSPEAKER]
               [--skip SKIP]
               [pathToVideoFile] [pathToBackgroundImage]

Welcome to Livius Project.

positional arguments:
  pathToVideoFile       The input path to video file.
  pathToBackgroundImage
                        The input path to background image.

optional arguments:
  -h, --help            show this help message and exit
  --t TIMEFRAME         Specify the time to select the frame for cropping
                        purpose.
  --speakerInfo SPEAKERINFO
                        enter the speaker' name inside the quotation marks
  --talkInfo TALKINFO   enter the tile of talk inside the quotation marks
  --instituteInfo INSTITUTEINFO
                        enter the name of institute.
  --dateInfo DATEINFO   enter the date of the talk. default= (July 2015)
  --sizeOfLayout SIZEOFLAYOUT
                        the video layout size for final broadcasting.
                        default=(1920, 1080)
  --sizeOfScreen SIZEOFSCREEN
                        the size of the screen portion in final layout.
                        default=(1280, 960)
  --sizeOfSpeaker SIZEOFSPEAKER
                        the size of the speaker portion in final layout.
                        default=(620, 360)
  --skip SKIP           Skip the first n frames


----------------------
Command Line Arguments
----------------------

Input:
	1. The path to your video file and the path to the background image (Required)
	2. Any input arguments which you need to modify (Optional)

output: 
	A video with your desired layout

--------------------------------------------------------------------------
Example:

python demo.py /media/pbahar/Data\ Raid/Videos/18.03.2015/video_5.mp4  
~/Downloads/EdgarFiles/background_images_mlss2013/background_example.png 
--t 1 
--talkInfo 'How to estimate missing data' 
--speakerInfo 'Prof. Dr. Schmidt' 
--instituteInfo 'Empirical Inference' 
--dateInfo 'August 23th 2015'
--------------------------------------------------------------------------

