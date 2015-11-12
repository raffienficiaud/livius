"""
Audio mixing jobs
=================

This file contains jobs able to process the audio stream of moviePy.

.. autosummary::

  AudioMixerJob

"""

from ..job import Job

import datetime
import os
import logging

from moviepy.editor import AudioFileClip

logger = logging.getLogger()

class AudioMixerJob(Job):
    """
    Job for mixing audio channels.

    The output video will have a mono audio stream, by mixing the two input streams. If the input stream
    is mono, the output is just returned as is.

    .. rubric:: Runtime parameters

    * ``video_filename`` name of the video to process (only the base name)
    * ``video_location`` location of the video to process (not cached)
    * ``mixing_left`` amount of the left channel in the final stream
    * ``mixing_right`` amount of the right channel in the final stream

    .. rubric:: Workflow inputs

    ``None``

    .. rubric:: Workflow outputs

    A transformed moviePy audio clip.

    :note:
        The transformations are only applied at write-time.
    """

    #: name of the job in the workflow
    name = 'audio_mixing'

    #: Cached inputs:
    #:
    #: * ``video_filename`` name of the input video (should be relocatable according to
    #:   ``video_location``).
    #: * ``mixing_left`` the amount of the left channel in the final output
    #: * ``mixing_right`` the amount of the right channel in the final output
    attributes_to_serialize = ['video_filename', 'mixing_left', 'mixing_right']

    parents = []

    def __init__(self, *args, **kwargs):
        super(AudioMixerJob, self).__init__(*args, **kwargs)
        assert('video_filename' in kwargs)
        assert('video_location' in kwargs)

        self.mixing_left = float(kwargs['mixing_left']) if 'mixing_left' in kwargs else 0.2
        self.mixing_right = float(kwargs['mixing_right']) if 'mixing_right' in kwargs else 0.8

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        super(AudioMixerJob, self).get_outputs()

        input_video = os.path.join(self.video_location, self.video_filename)

        clip = AudioFileClip(input_video)

        def apply_effects(get_frame, t):
            """Function that chains together all the post processing effects."""
            frame = get_frame(t)

            if frame.shape[1] < 2:
                return frame

            mixed = self.mixing_left * frame[:, 0] + self.mixing_right * frame[:, 1]
            return mixed

        # retains the duration of the clip
        return clip.fl(apply_effects)
