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
import logging

from moviepy.editor import AudioFileClip
from ...editing.layout import createFinalVideo

logger = logging.getLogger()

class ClipsToMovie(Job):
    """Job taking two Clip job (and metadata) and produce the output video.

    .. rubric:: Runtime parameters

    * ``video_edit_images_folder`` location of the background folders. This parameter is not cached as
      it should be possible to relocate the files
    * ``background_image_name`` file name of the background image (composed using ``video_edit_images_folder``).
    * ``credit_image_names`` image shown at the end of the video. Should be in the same path as ``background_image_name``
      (hence given by ``video_edit_images_folder``).
    * ``output_video_file`` name (without folder) of the output video.
    * ``output_video_folder`` folder where the videos are stored. This value is not cached for the same rationale as the
      other parameters.

    .. rubric:: Workflow input

    The input consists in two moviePy clips, respectively for the slides and the speaker:

    * slides: have been warped and corrected accordingly. Such a clip may be provided by :py:class:`ExtractSlideClipJob`.
    * speaker: tracking the speaker and stabilizing the tracked region over time.

    """

    #: neame of the job in the workflow
    name = "clips_to_movie"

    #: Cached input:
    #:
    #: * ``video_output_file`` name of the output file
    #: * ``video_filename`` input video file
    #: * ``slide_clip_desired_format`` final size of the slides
    #: * ``background_image_name`` filename of the background image
    #: * ``credit_image_names`` image shown at the end of the video. By default their location is in the ressource folder of the
    #:   python livius package.
    #: * ``canvas_size`` size of the final video
    #: * ``intro_image_names`` the name of the image shown in introduction
    attributes_to_serialize = ['output_video_file',
                               'video_filename',
                               'slide_clip_desired_format',
                               'background_image_name',
                               'intro_image_names',
                               'credit_image_names',
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
          :py:func:`get_default_background_image` will be used instead. This value is cached.
        :param str background_image_name: the name of the image used for ending the video. If not specified, the function
          :py:func:`get_default_credit_images` will be used instead. This value is cached.
        :param tuple canvas_size: the final size of the vide. Default to (1920, 1080). This value is cached
        :param str output_video_folder: folder where the output file will be created. Default to value returned by
          :py:func:`get_output_video_folder`.
        :param str output_video_file: full path name of the output video. Default to the value returned by
          :py:func:`get_output_video_file`.

        """

        super(ClipsToMovie, self).__init__(*args, **kwargs)

        assert('slide_clip_desired_format' in kwargs)
        assert('video_filename' in kwargs)

        # unicode is for comparison issues when retrieved from the json
        self.video_filename = unicode(self.video_filename)

        self.background_image_name = unicode(kwargs.get('background_image_name', self.get_default_background_image()))
        self.credit_image_names = kwargs.get('credit_image_names', self.get_default_credit_images())
        self.credit_image_names = [unicode(i) for i in self.credit_image_names]
        self.intro_image_names = kwargs.get('intro_image_names')
        self.intro_image_names = [unicode(i) for i in self.intro_image_names]
        self.canvas_size = list(kwargs.get('canvas_size', (1920, 1080)))  # from json comparison
        self.output_video_folder = unicode(kwargs.get('output_video_folder', self.get_output_video_folder()))
        self.output_video_file = kwargs.get('output_video_file', self.get_output_video_file())
        self.video_edit_images_folder = unicode(kwargs.get('video_edit_images_folder', '/media/alme/processing_mahdi/PNG images/'))  # TODO change THAT THING

    def get_default_background_image(self):
        """Returns the name of the background image"""

        return "background_mlss2015.png"  # TODO change THIS

    def get_default_credit_images(self):
        """Returns a list of images used for the credit segment"""

        return ["credit_images/processing_team_mlss2015.png", "credit_images/organizers_sponsors_mlss2015.png"]

    def get_output_video_folder(self):
        """Returns the location of the output video file. This is the same location as the json prefix file"""
        return os.path.dirname(self.json_filename)

    def get_output_video_file(self):
        """Returns the name without folder of the output video file. The output file is created using the name
        of the input file"""
        return os.path.splitext(os.path.basename(self.video_filename))[0] + "_out" + self.get_container()

    def get_container(self):
        """Returns the container used for storing the output video streams"""
        return ".mp4"

    def is_up_to_date(self):
        """Checks the existance of the video file and their timestamp. Then fallsback on the default method"""
        # checks if the file exists and is not outdated
        if(not os.path.exists(os.path.join(self.output_video_folder, self.output_video_file))):
            return False

        if(os.stat(os.path.join(self.output_video_folder, self.output_video_file)).st_mtime < os.stat(self.video_filename).st_mtime):
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
        output_video = os.path.join(self.output_video_folder, self.output_video_file)
        output_video_no_container = os.path.splitext(output_video)[0]  # container will be appended by the layout function


        ressource_folder = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "ressources")
        video_background_image = os.path.join(ressource_folder, self.background_image_name)


        credit_images_and_durations = [(os.path.join(ressource_folder, i), None) for i in self.credit_image_names]  # None sets the duration to the default
        intro_images_and_durations = [(os.path.join(self.video_edit_images_folder, i), None) for i in self.intro_image_names]  # None sets the duration to the default

        audio_clip = AudioFileClip(input_video)

        createFinalVideo(slide_clip=slide_clip,
                         speaker_clip=speaker_clip,
                         audio_clip=audio_clip,
                         video_background_image=video_background_image,
                         intro_image_and_durations=intro_images_and_durations,
                         credit_images_and_durations=credit_images_and_durations,
                         fps=30,
                         talkInfo='How to use svm kernels',
                         speakerInfo='Prof. Bernhard Schoelkopf',
                         instituteInfo='Empirical Inference',
                         dateInfo='July 25th 2015',
                         first_segment_duration=10,
                         output_file_name=output_video_no_container,
                         codecFormat='libx264',
                         container=self.get_container(),
                         flagWrite=True)


        # stop time
        stop = datetime.datetime.now()
        self.processing_time = (stop - start).seconds

        pass


    pass

    def get_outputs(self):
        super(ClipsToMovie, self).get_outputs()
        logger.info('[CREATEMOVIE] processed video %s in %s seconds', self.get_output_video_file(), self.processing_time)
        return None


