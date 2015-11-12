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
from ...editing.layout import createFinalVideo, default_layout as video_default_layout

logger = logging.getLogger()


class ClipsToMovie(Job):
    """Job taking two Clip job (and metadata) and produce the output video.

    .. rubric:: Runtime parameters

    * ``video_filename`` name of the video to process (only the base name)
    * ``video_location`` location of the video to process (not cached)
    * ``video_intro_images_folder`` location of the background folders. This parameter is not cached as
      it should be possible to relocate the files
    * ``background_image_name`` file name of the background image (composed using ``video_intro_images_folder``).
    * ``credit_image_names`` image shown at the end of the video. Should be in the same path as ``background_image_name``
      (hence given by ``video_intro_images_folder``).
    * ``output_video_file`` name (without folder) of the output video.
    * ``output_video_folder`` folder where the videos are stored. This value is not cached for the same rationale as the
      other parameters.

    .. rubric:: Workflow input

    The input consists in two moviePy clips, and a metadata feed. The video clips are respectively
    for the slides and the speaker:

    * slides: have been warped and corrected accordingly. Such a clip may be provided by :py:class:`ExtractSlideClipJob`.
    * speaker: tracking the speaker and stabilizing the tracked region over time.

    The metadata feed is a dictionaty containing the informations about the video.

    """

    #: name of the job in the workflow
    name = "clips_to_movie"

    #: Cached input:
    #:
    #: * ``video_output_file`` name of the output file
    #: * ``video_filename`` input video file (not containing the path)
    #: * ``slide_clip_desired_format`` final size of the slides
    #: * ``background_image_name`` filename of the background image
    #: * ``credit_image_names`` image shown at the end of the video. By default their location is in the ressource folder of the
    #:   python livius package.
    #: * ``video_layout`` the layout of the final video. See :py:func:`createFinalVideo` for a description
    #:   of the layout.
    attributes_to_serialize = ['output_video_file',
                               'video_filename',
                               'slide_clip_desired_format',
                               'background_image_name',
                               'credit_image_names',
                               'video_layout']

    #: Cached output:
    #:
    #: * ``processing_time`` processing time
    outputs_to_cache = ['processing_time']

    def __init__(self, *args, **kwargs):
        """
        :param str video_location: directory under which the video is placed. This is not cached as the
          directories might be different from one processing computer to the other (relocation).
        :param str video_intro_images_folder: location of the background image. Changing this value
          will not trigger a recomputation so that it is possible to relocate the files to another
          folder. Defaults to ``None``, in which case full paths are expected from the Metadata provider
          (under the key ``intro_images``). See :py:class:`Metadata`.
        :param str background_image_name: the name of the image used for background. If not specified, the function
          :py:func:`get_default_background_image` will be used instead. This value is cached.
        :param str background_image_name: the name of the image used for ending the video. If not specified, the function
          :py:func:`get_default_credit_images` will be used instead. This value is cached.
        :param dict video_layout: the layout of the video. This value is cached
        :param str output_video_folder: folder where the output file will be created. Default to value returned by
          :py:func:`get_output_video_folder`.
        :param str output_video_file: full path name of the output video. Default to the value returned by
          :py:func:`get_output_video_file`.
        :param bool is_visual_test: if set to ``True`` limits the processing of the videos to 10 seconds. Defaults to
          ``False``.

        """

        super(ClipsToMovie, self).__init__(*args, **kwargs)

        assert('slide_clip_desired_format' in kwargs)
        assert('video_filename' in kwargs)
        assert('video_location' in kwargs)

        # unicode is for comparison issues when retrieved from the json
        self.video_filename = unicode(self.video_filename)

        self.background_image_name = unicode(kwargs.get('background_image_name', self.get_default_background_image()))
        self.credit_image_names = kwargs.get('credit_image_names', self.get_default_credit_images())
        self.credit_image_names = [unicode(i) for i in self.credit_image_names]  # unicode for json
        self.video_layout = kwargs.get('video_layout', video_default_layout)
        self.output_video_folder = unicode(kwargs.get('output_video_folder', self.get_output_video_folder()))
        self.output_video_file = kwargs.get('output_video_file', self.get_output_video_file())
        self.is_test = kwargs.get('is_visual_test', False)

        self.video_intro_images_folder = None  # not cached, hence not created automatically
        if 'video_intro_images_folder' in kwargs:
            self.video_intro_images_folder = unicode(kwargs['video_intro_images_folder'])

        # JSON constraint: keys are unicode, locations are lists
        self.video_layout = dict([(unicode(k), list(v)) for k, v in self.video_layout.items()])

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
        return os.path.splitext(self.video_filename)[0] + "_out" + self.get_container()

    def get_container(self):
        """Returns the container used for storing the output video streams"""
        return ".mp4"

    def is_up_to_date(self):
        """Checks the existance of the video file and their timestamp. Then fallsback on the default method"""
        # checks if the file exists and is not outdated
        if(not os.path.exists(os.path.join(self.output_video_folder, self.output_video_file))):
            return False

        if(os.stat(os.path.join(self.output_video_folder, self.output_video_file)).st_mtime < os.stat(os.path.join(self.video_location, self.video_filename)).st_mtime):
            return False

        # default
        return super(ClipsToMovie, self).is_up_to_date()

    def run(self, *args, **kwargs):

        slide_clip = args[0]
        speaker_clip = args[1]
        meta = args[2] if len(args) > 2 else None
        audio_clip = args[3] if len(args) > 3 else None

        assert(hasattr(self, 'processing_time'))
        self.processing_time = None

        # start time, just to save something
        start = datetime.datetime.now()

        input_video = os.path.join(self.video_location, self.video_filename)
        output_video = os.path.join(self.output_video_folder, self.output_video_file)
        output_video_no_container = os.path.splitext(output_video)[0]  # container will be appended by the layout function

        # the folder containing the images common to all videos
        ressource_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "ressources"))
        video_background_image = os.path.join(ressource_folder, self.background_image_name)

        credit_images_and_durations = [(os.path.join(ressource_folder, i), None) for i in self.credit_image_names]  # None sets the duration to the default

        # introduction image
        intro_images_and_durations = []
        if meta is not None and "intro_images" in meta:
            intro_images_and_durations = []
            for current in meta['intro_images']:

                image_file = current

                if not os.path.exists(image_file):
                    # fallback in case the image does not exist (full path not given)
                    if self.video_intro_images_folder is not None:
                        image_file = os.path.join(self.video_intro_images_folder, current)

                if not os.path.exists(image_file):
                    logger.error("[INTRO] image %s not found", image_file)
                    raise RuntimeError("[INTRO] image %s does not exist" % image_file)

                intro_images_and_durations += [(image_file, None)]  # None sets the duration to the default

        # pauses / including video begin/end
        pauses = []
        if meta is not None and "video_begin" in meta and meta['video_begin'] is not None:
            pauses += [(None, meta['video_begin'])]  # None means video begin

        if meta is not None and "video_end" in meta and meta['video_end'] is not None:
            pauses += [(meta['video_end'], None)]  # None means video end

        # if no specific handling of the audio is performed upstream, we default to the one of the video
        if audio_clip is None:
            audio_clip = AudioFileClip(input_video)

        createFinalVideo(slide_clip=slide_clip,
                         speaker_clip=speaker_clip,
                         audio_clip=audio_clip,
                         video_background_image=video_background_image,
                         intro_image_and_durations=intro_images_and_durations,
                         credit_images_and_durations=credit_images_and_durations,
                         fps=30,
                         talk_title=meta['talk_title'] if meta is not None else 'title',
                         speaker_name=meta['speaker_name'] if meta is not None else 'name',
                         talk_date=meta['talk_date'] if meta is not None else 'today',
                         first_segment_duration=10,
                         pauses=pauses,
                         output_file_name=output_video_no_container,
                         codecFormat='libx264',
                         container=self.get_container(),
                         flagWrite=True,
                         is_test=self.is_test)

        # stop time
        stop = datetime.datetime.now()
        self.processing_time = (stop - start).seconds

        pass

    pass

    def get_outputs(self):
        super(ClipsToMovie, self).get_outputs()
        logger.info('[CREATEMOVIE] processed video %s in %s seconds', self.get_output_video_file(), self.processing_time)
        return self.processing_time

