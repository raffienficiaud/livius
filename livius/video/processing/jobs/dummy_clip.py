"""
Dummy clip
==========

This file contains one unique Job for creating a dummy mmoviePy clip.

.. autosummary::

  RandomImageClipJob

"""

from ..job import Job
from moviepy.editor import VideoClip, VideoFileClip
import os


class RandomImageClipJob(Job):

    """
    Creates a dummy (random noise) clip for moviePy, mainly for testing purposes.

    .. rubric:: Runtime parameters

    * ``random_frame_size`` size of the generated frame in the final movie

    .. rubric:: Workflow inputs

    None

    .. rubric:: Workflow outputs

    The output is a random noise video clip
    """

    #: name of the job in the workflow
    name = 'random_image_clip'

    #: Nothing to cache, the generated output is a moviepy object
    #: that is accessed lazily.
    attributes_to_serialize = []

    def __init__(self, *args, **kwargs):
        """
        :param tuple random_frame_size: indicates the size of the random frames. Default to
            the runtime parameter ``slide_clip_desired_format`` if available, ``(640, 480)``
            otherwise.
        """
        super(RandomImageClipJob, self).__init__(*args, **kwargs)

        self.frame_size = kwargs.get('random_frame_size', None)
        if self.frame_size is None:
            self.frame_size = kwargs.get('slide_clip_desired_format', (640, 480))

        if hasattr(self, 'video_filename'):
            assert('video_location' in kwargs)
            self.video_location = kwargs['video_location']

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        super(RandomImageClipJob, self).get_outputs()

        import numpy as np

        def make_frame(t):
            """Function that chains together all the post processing effects."""
            return np.random.random(self.frame_size + [3]) * 255

        clip = VideoClip(make_frame)

        if hasattr(self, 'video_filename'):
            clip = clip.set_duration(VideoFileClip(os.path.join(self.video_location, self.video_filename)).duration)
        return clip


class OriginalVideoClipJob(Job):
    """
    Creates a moviePy clip containing the original video.

    .. rubric:: Runtime parameters

    .. rubric:: Workflow inputs

    None

    .. rubric:: Workflow outputs

    MoviePy videoClip containing the original video.

    .. note::

       Does not preserve aspect ratio.
    """

    #: name of the job in the workflow
    name = 'original_video_clip'

    #: Nothing to cache, the generated output is a moviepy object
    #: that is accessed lazily.
    attributes_to_serialize = []

    def __init__(self, *args, **kwargs):
        """
        :param tuple original_video_clip_size: indicates the size of the video. Default to
            ``None`` (no resize).

        """
        super(OriginalVideoClipJob, self).__init__(*args, **kwargs)

        self.frame_size = kwargs.get('original_video_clip_size', None)

        assert('video_location' in kwargs and self.video_location is not None)
        assert('video_filename' in kwargs and self.video_filename is not None)
        assert(os.path.exists(os.path.join(self.video_location, self.video_filename)))

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        super(OriginalVideoClipJob, self).get_outputs()
        clip = VideoFileClip(os.path.join(self.video_location, self.video_filename))
        if(self.frame_size is not None):
            clip = clip.resize(self.frame_size)
        return clip
