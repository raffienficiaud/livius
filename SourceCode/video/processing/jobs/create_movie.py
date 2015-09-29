"""
Output video creation
=====================

This file contains the Job and functions for creating the final video for moviePy.

.. autosummary::

  ClipsToMovie




"""

from ..job import Job

import datetime
import os
from moviepy.editor import AudioFileClip

from ...editing.layout import createFinalVideo

class ClipsToMovie(Job):
    """Job taking two Clip job (and metadata) and produce the output video.

    .. rubric:: Runtime parameters

    * ``video_edit_images_folder`` location of the background folders. This parameter is not cached as
      it should be possible to relocate the files
    * ``background_image_name`` file name of the background image (composed using ``video_edit_images_folder``).
    * ``epilog_image_name`` image shown at the end of the video. Should be in the same path as ``background_image_name``
      (hence given by ``video_edit_images_folder``).

    .. rubric:: Workflow input

    The input consists in two moviePy clips, respectively for the slides and the speaker:

    * slides: have been warped and corrected accordingly. Such a clip may be provided by :py:class:`ExtractSlideClipJob`.
    * speaker: tracking the speaker and stabilizing the tracked region over time.

    """

    #: neame of the job in the workflow
    name = "clips_to_movie"

    #: Cached input:
    #:
    #: * ``output_file`` name of the output file
    #: * ``video_filename`` input video file
    #: * ``slide_clip_desired_format`` final size of the slides
    #: * ``background_image_name`` filename of the background image
    #: * ``epilog_image_name`` image shown at the end of the video
    #: * ``canvas_size`` size of the final video
    attributes_to_serialize = ['output_file',
                               'video_filename',
                               'slide_clip_desired_format',
                               'background_image_name',
                               'epilog_image_name',
                               'canvas_size']

    #: Cached output:
    #:
    #: * ``processing_time`` processing time
    outputs_to_cache = ['processing_time']

    def __init__(self, *args, **kwargs):
        """
        :param str video_edit_images_folder: location of the background image. Changing this value
          will not trigger a recomputation so that it is possible to relocate the files to another
          folder.
        :param str background_image_name: the name of the image used for background. If not specified, the function
          :py:func:`get_background_image` will be used instead. This value is cached.
        :param str background_image_name: the name of the image used for ending the video. If not specified, the function
          :py:func:`get_epilog_image` will be used instead. This value is cached.
        :param tuple canvas_size: the final size of the vide. Default to (1920, 1080). This value is cached
        """

        super(ClipsToMovie, self).__init__(*args, **kwargs)

        assert('slide_clip_desired_format' in kwargs)

        self.background_image_name = kwargs.get('background_image_name', self.get_background_image())
        self.epilog_image_name = kwargs.get('epilog_image_name', self.get_epilog_image())
        self.canvas_size = kwargs.get('canvas_size', (1920, 1080))

    def get_background_image(self):
        """Returns the name of the background image"""

        return None

    def get_epilog_image(self):
        """Returns the name of the epilog image"""

        return None

    def is_up_to_date(self):
        """Checks the existance of the video file and their timestamp. Then fallsback on the default method"""

        # checks if the file exists and is not outdated
        if(not os.path.exists(self.output_file)):
            return False

        if(os.stat(self.output_file).st_mtime > os.stat(self.video_filename).st_mtime):
            return False

        # default
        return super(ClipsToMovie, self).is_up_to_date()


    def run(self, *args, **kwargs):

        slide_clip = args[0]
        speaker_clip = args[1]

        assert(hasattr(self, 'processing_time'))
        self.processing_time = None

        # start time, just to save something
        start = datetime.datetime.now()

        input_video = self.video_filename

        pathToBackgroundImage = os.path.join(self.video_edit_images_folder, self.background_image_name)
        pathToEpilogImage = os.path.join(self.video_edit_images_folder, self.epilog_image_name)

        # TODO add time begin cut
        # maybe actually by modifying the clips that are arriving into this job
        audio = AudioFileClip(input_video)

        createFinalVideo(slide_clip,
                         speaker_clip,
                         pathToBackgroundImage,  # pathToBackgroundImage
                         pathToEpilogImage,  # pathToFinalImage
                         audio,
                         fps=30,
                         sizeOfLayout=self.canvas_size,
                         sizeOfScreen=self.slide_clip_desired_format,
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


        # stop time
        stop = datetime.datetime.now()
        self.processing_time = (stop - start).seconds

        pass


    pass
