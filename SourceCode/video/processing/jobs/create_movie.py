"""
Output video creation
=====================

This file contains the Job and functions for creating the final video for moviePy.

.. autosummary::

  ClipsToMovie




"""

from ..job import Job


class ClipsToMovie(Job):
    """Job taking two Clip job (and metadata) and produce the output video.


    .. rubric:: Workflow input

    The input consists in two moviePy clips, respectively for the slides and the speaker:

    * slide

    """

    name = "clips_to_movie"
    attributes_to_serialize = ['output_file', 'video_filename']
    outputs_to_cache = ['processing_time']

    def run(self, *args, **kwargs):

        slide_clip = args[0]
        speaker_clip = args[1]


        # video_duration_shrink (input_video, tStart=(40,50.0), tEnd=(45,0.0), writeFlie = True)
        input_video = self.video_filename
        # slideClip = VideoFileClip(input_video, audio=False)
        targetVideo2 = "/media/pbahar/Data Raid/Videos/18.03.2015/video_6.mp4"
        # speakerClip = VideoFileClip(targetVideo2, audio=False)
        pathToBackgroundImage = "/is/ei/pbahar/Downloads/EdgarFiles/background_images_mlss2013/background_example.png"
        audio = AudioFileClip(input_video)

        createFinalVideo(slide_clip,
                         speaker_clip,
                         pathToBackgroundImage,
                         pathToBackgroundImage,
                         audio,
                         fps=30,
                         sizeOfLayout=(1920, 1080),
                         sizeOfScreen=(1280, 960),
                         sizeOfSpeaker=(620, 360),
                         talkInfo='How to use svm kernels',
                         speakerInfo='Prof. Bernhard Schoelkopf',
                         instituteInfo='Empirical Inference',
                         dateInfo='July 25th 2015',
                         firstPause=10,
                         nameToSaveFile=self.output_file,
                         codecFormat='libx264',
                         container='.mp4',
                         flagWrite=True)




        pass


    pass
