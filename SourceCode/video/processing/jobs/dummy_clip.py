"""
Dummy clip
==========

This file contains one unique Job for creating a dummy mmoviePy clip.

.. autosummary::

  DummyClipJob

"""

from ..job import Job


class DummyClipJob(Job):

    """
    Creates a dummy (random noise) clip for moviePy, mainly for testing purposes.

    .. rubric:: Workflow inputs

    None

    .. rubric:: Workflow outputs

    The output is a random noise video clip
    """

    name = 'dummy_clip'
    attributes_to_serialize = ['frame_size']

    def __init__(self, *args, **kwargs):
        super(DummyClipJob, self).__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        pass

    def get_outputs(self):
        super(DummyClipJob, self).get_outputs()

        import numpy as np
        def make_frame(t):
            """Function that chains together all the post processing effects."""
            return np.random(*self.frame_size, 3) * 255

        clip = VideoClip(make_frame)
        return clip
