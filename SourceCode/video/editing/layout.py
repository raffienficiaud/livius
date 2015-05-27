
"""
This module implements layout and video formation for video editing part
main function:
- createFinalVideo
"""

import numpy as np
import os.path
from moviepy.editor import *
from moviepy.Clip import * 
import matplotlib.pyplot as plt
from pylab import *


def createFinalVideo(slideClip,speakerClip,
                        pathToBackgroundImage,
                        pathToFinalImage,
                        audio,
                        fps,
                        sizeOfLayout = (1920, 1080), 
                        sizeOfScreen = (1280, 960),
                        sizeOfSpeaker = (620, 360),
                        talkInfo = ' ',
                        speakerInfo = ' ',
                        instituteInfo = ' ',
                        dateInfo = 'July 2015',
                        firstPause = 10.0, 
                        nameToSaveFile = 'Output', 
                        codecFormat = 'libx264',
                        container = '.mp4',
                        flagWrite = True):
    
    """
    This function serves to form the video layout, create the final video and write it.
    Inputs:
        slideClip: The slide clip -- video object of moviepy
        speakerClip: The screen clip -- video object of moviepy
        pathToBackgroundImage: The path of background image
        pathToFinalImage: The path of final image for supporting groups and organizations
        audio: The audio file to be attached to the video file -- audio object of moviepy
        fps: frame per second
        sizeOfLayout: The desired size of whole layout, default=(1920, 1080) -- tuple
        sizeOfScreen: The desired size of screen part, default=(1280, 960) -- tuple
        sizeOfSpeaker: The desired size of speaker part, default=(620, 360) -- tuple
        talkInfo: Information about talk like the title, the subject, etc. -- string 
        speakerInfo: Information about talk like the title, the subject, etc. -- string 
        instituteInfo: Information about instiute/University. -- string 
        dateInfo: Information about the date of talk. -- string 
        firstPause: The amount of time in seconds to start the lecture. Actually, 
        this amount of time can be used for showing some information at the beginning of the clip.
        default: 10.0 sec         
        nameToSaveFile: A name to save the file, default: 'Output' -- string 
        codecFormat = The codec format to write the new video into the file, default: 'libx264', -- string
        Codec to use for image encoding. Can be any codec supported by ffmpeg, but the container
        must be set accordingly.
        container: The video container, '.mp4' -- string
        flagWrite: A flag to set whether write the new video or not, default = True, boolean
        
        Hint:
        you may simply change the codecs, fps and etc inside the function.
        ``'libx264'`` (use file extension ``.mp4``)
        makes well-compressed videos (quality tunable using 'bitrate').


        ``'mpeg4'`` (use file extension ``.mp4``) can be an alternative
        to ``'libx264'``, and produces higher quality videos by default.


        ``'rawvideo'`` (use file extension ``.avi``) will produce 
        a video of perfect quality, of possibly very huge size.

        ``png`` (use file extension ``.avi``) will produce a video
        of perfect quality, of smaller size than with ``rawvideo``

        ``'libvorbis'`` (use file extension ``.ogv``) is a nice video
        format, which is completely free/ open source. However not
        everyone has the codecs installed by default on their machine.
        
        ``'libvpx'`` (use file extension ``.webm``) is tiny a video
        format well indicated for web videos (with HTML5). Open source.


        audio
        Either ``True``, ``False``, or a file name.
        If ``True`` and the clip has an audio clip attached, this
        audio clip will be incorporated as a soundtrack in the movie.
        If ``audio`` is the name of an audio file, this audio file
        will be incorporated as a soundtrack in the movie.

        audiofps
        frame rate to use when writing the sound.

        temp_audiofile
        the name of the temporary audiofile to be generated and
        incorporated in the the movie, if any.

        audio_codec
        Which audio codec should be used. Examples are 'libmp3lame'
        for '.mp3', 'libvorbis' for 'ogg', 'libfdk_aac':'m4a',
        'pcm_s16le' for 16-bit wav and 'pcm_s32le' for 32-bit wav.

        audio_bitrate
        Audio bitrate, given as a string like '50k', '500k', '3000k'.
        Will determine the size/quality of audio in the output file.
        Note that it mainly an indicative goal, the bitrate won't
        necessarily be the this in the final file.

        write_logfile
        If true, will write log files for the audio and the video.
        These will be files ending with '.log' with the name of the
        output file in them.


    Example:
        createFinalVideo(slideClip,speakerClip,
                        pathToBackgroundImage,
                        pathToFinalImage,
                        audio,
                        fps = 30, 
                        sizeOfLayout = (1920, 1080), 
                        sizeOfScreen = (1280, 960),
                        sizeOfSpeaker = (620, 360),
                        talkInfo = 'How to use SVM kernels',
                        speakerInfo = 'Prof. Bernhard Schoelkopf',
                        instituteInfo = 'Empirical Inference',
                        dateInfo = 'July 2015',
                        firstPause = 10,
                        nameToSaveFile = 'video', 
                        codecFormat = 'libx264',
                        container = '.mp4',
                        flagWrite = True)


    """
    
    '''
    if slideClip.fps != speakerClip.fps:
        print "[layout] Error: The fps of two videos should be the same"
    '''
    backgroundImage = ImageClip(pathToBackgroundImage)
    finalImage = ImageClip(pathToFinalImage)
    # Resize the videos and images to the desire size
    backgroundImageResized = backgroundImage.resize((sizeOfLayout[0],sizeOfLayout[1]))
    finalImageResized = finalImage.resize((sizeOfLayout[0],sizeOfLayout[1]))
    slideClipResized = slideClip.resize((sizeOfScreen[0],sizeOfScreen[1]))
    speakerClipResized = speakerClip.resize((sizeOfSpeaker[0],sizeOfSpeaker[1]))

    if (sizeOfLayout[0] < (sizeOfScreen[0] or sizeOfSpeaker[0])) and (sizeOfLayout[1] < (sizeOfScreen[1] or sizeOfSpeaker[1])):
        print "[layout] Warning: The selected sizes are not appropriate."


    desiredInfo1 = talkInfo  
    # Generate a text clip. You can customize the font, color, etc.
    txtClipInfo1 = TextClip(desiredInfo1,fontsize=25,color='white',font="Amiri")
    # Say that you want it to appear 10s 
    txtClipInfo1 = txtClipInfo1.set_pos((640, 1030)).set_duration(slideClip.duration)
    
    desiredInfo2 = speakerInfo  
    # Generate a text clip. You can customize the font, color, etc.
    txtClipInfo2 = TextClip(desiredInfo2,fontsize=25,color='white',font="Amiri")
    # Say that you want it to appear 10s 
    txtClipInfo2 = txtClipInfo2.set_pos((640, 1060)).set_duration(slideClip.duration)
    
    '''
    desiredStringBeginningLeft = 'Main Organizer'
    desiredStringBeginningRight = 'Video Team'
    desiredStringBeginningMiddle = 'Audio Team'
    
    
    txtClipBeginLeft = TextClip(desiredStringBeginningLeft,fontsize=40,color='white',font="Amiri")
    txtClipBeginLeft = txtClipBeginLeft.set_pos((100 , sizeOfLayout[1]/3)).set_duration(firstPause/2)
    
    txtClipBeginRight = TextClip(desiredStringBeginningRight,fontsize=40,color='white',font="Amiri")
    txtClipBeginRight = txtClipBeginRight.set_pos(((sizeOfLayout[0]/3) +100 , sizeOfLayout[1]/3)).set_duration(firstPause/2)
    
    txtClipBeginMiddle = TextClip(desiredStringBeginningMiddle,fontsize=40,color='white',font="Amiri")
    txtClipBeginMiddle = txtClipBeginMiddle.set_pos(((2*sizeOfLayout[0]/3) +100, sizeOfLayout[1]/3)).set_duration(firstPause/2)
    '''
    
    # The template for the second show
    
    titleText = talkInfo
    speakerText = speakerInfo
    instituteText = instituteInfo
    defaultText = 'Machine Learning Summer School 2015'
    dateText = dateInfo
    
    pixelLeftMargin = int(sizeOfLayout[0]/3)-100
    pixelRightMargin = 100
    maxNumHorizontalPixel = sizeOfLayout[0] - pixelLeftMargin - pixelRightMargin  
    
    lenStringTitleText = len(titleText)
    #print lenStringTitleText
    pixelPerCharTitleText = int((maxNumHorizontalPixel / lenStringTitleText))
    
    
    lenStringSpeakerText = len(speakerText)
    pixelPerCharSpeakerText = int((maxNumHorizontalPixel / lenStringSpeakerText))

    
    
    txtClipTitleText = TextClip(titleText,fontsize=pixelPerCharTitleText,color='white',font="Amiri")
    txtClipTitleText = txtClipTitleText.set_pos(((sizeOfLayout[0]/3)-100 , sizeOfLayout[1]/3 )).set_duration(firstPause)
    
    txtClipSpeakerText = TextClip(speakerText,fontsize=pixelPerCharSpeakerText,color='white',font="Amiri")
    txtClipSpeakerText = txtClipSpeakerText.set_pos(((sizeOfLayout[0]/3)-100 , sizeOfLayout[1]/3 +100)).set_duration(firstPause)
    
    
    txtClipInstituteText = TextClip(instituteText,fontsize=36,color='white',font="Amiri")
    txtClipInstituteText = txtClipInstituteText.set_pos(((sizeOfLayout[0]/3)-100, sizeOfLayout[1]/3 + 170)).set_duration(firstPause)
    
    txtClipDefaultText = TextClip(defaultText,fontsize=40,color='white',font="Amiri")
    txtClipDefaultText = txtClipDefaultText.set_pos(((sizeOfLayout[0]/3)-100, sizeOfLayout[1]/3 + 300 )).set_duration(firstPause)

    txtClipDateText = TextClip(dateText,fontsize=40,color='white',font="Amiri")
    txtClipDateText = txtClipDateText.set_pos(((sizeOfLayout[0]/3)-100, sizeOfLayout[1]/3 +350)).set_duration(firstPause)

    
    outputVideo = CompositeVideoClip([backgroundImageResized.set_duration((slideClip.duration+firstPause)),
                                    speakerClipResized.set_pos((10,360)).set_start(firstPause),
                                    slideClipResized.set_pos((640,60)).set_start(firstPause), 
                                    txtClipInfo1.set_start(firstPause),
                                    txtClipInfo2.set_start(firstPause),
                                    #txtClipBeginLeft.set_start(0),
                                    #txtClipBeginRight.set_start(0),
                                    #txtClipBeginMiddle.set_start(0),
                                    txtClipTitleText.set_start(0),
                                    txtClipSpeakerText.set_start(0),
                                    txtClipInstituteText.set_start(0),
                                    txtClipDefaultText.set_start(0),
                                    txtClipDateText.set_start(0),
                                    finalImageResized.set_duration(10).set_start(slideClip.duration)])
    
    

    
    
    if flagWrite:    
        #base = os.path.basename(pathToBackgroundImage)
        #directory = os.path.dirname(pathToBackgroundImage)
        saveName = nameToSaveFile + container
        compo = CompositeAudioClip([audio.set_start(firstPause)])
        outputVideo = outputVideo.set_audio(compo)
        outputVideo.write_videofile(saveName, fps, codec=codecFormat)
        
    return outputVideo


if __name__ == '__main__':
     
    
    #video_duration_shrink (targetVideo, tStart=(40,50.0), tEnd=(45,0.0), writeFlie = True)
    targetVideo = "/media/pbahar/Data Raid/Videos/18.03.2015/video_6.mp4"
    slideClip = VideoFileClip(targetVideo,audio=False)
    targetVideo2 = "/media/pbahar/Data Raid/Videos/18.03.2015/video_6.mp4"
    speakerClip = VideoFileClip(targetVideo2,audio=False)
    pathToBackgroundImage = "/is/ei/pbahar/Downloads/EdgarFiles/background_images_mlss2013/background_example.png"
    audio = AudioFileClip(targetVideo)
    
    createFinalVideo(slideClip,speakerClip,
                        pathToBackgroundImage,
                        pathToBackgroundImage,
                        audio,
                        fps = 30, 
                        sizeOfLayout = (1920, 1080), 
                        sizeOfScreen = (1280, 960),
                        sizeOfSpeaker = (620, 360),
                        talkInfo = 'How to use svm kernels',
                        speakerInfo = 'Prof. Bernhard Schoelkopf',
                        instituteInfo = 'Empirical Inference',
                        dateInfo = 'July 25th 2015',
                        firstPause = 10,
                        nameToSaveFile = 'video_test', 
                        codecFormat = 'libx264',
                        container = '.mp4',
                        flagWrite = True)
                        