'''
Defines a job for extracting the segments from the histogram correlations
'''

import os
import json
import cv2
from random import randint

from ..job import Job
from ....util.user_interaction import get_polygon_from_user


class SegmentComputationJob(Job):
    """

    """

    name = "compute_segments"
    attributes_to_serialize = ['tolerance',
                               'segments']

    def __init__(self,
                 *args,
                 **kwargs):

        super(SegmentComputationJob, self).__init__(*args,
                                                    **kwargs)
        self._get_previously_computed_segments()

    def _get_previously_computed_segments(self):
        if not os.path.exists(self.json_filename):
            return None

        with open(self.json_filename) as f:
            d = json.load(f)

            if 'segments' in d:
                setattr(self, 'segments', d['segments'])

    def run(self, *args, **kwargs):

        self.segments = []

        # @todo(Stephan): implement!

        pass

    def get_outputs(self):
        super(SegmentComputationJob, self).get_outputs()
        if self.segments is None:
            raise RuntimeError('The segments have not been computed yet')
        return self.segments


if __name__ == '__main__':

    import logging
    FORMAT = '[%(asctime)-15s] %(message)s'
    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    root_folder = os.path.join(os.path.dirname(__file__),
                               os.pardir,
                               os.pardir,
                               os.pardir,
                               os.pardir)

    video_folder = os.path.join(root_folder, 'Videos')
    current_video = os.path.join(video_folder, 'Video_7.mp4')
    proc_folder = os.path.abspath(os.path.join(root_folder, 'tmp'))
    if not os.path.exists(proc_folder):
        os.makedirs(proc_folder)

    from .histogram_computation import HistogramLABDiff
    SegmentComputationJob.add_parent(HistogramLABDiff)

    # import ipdb
    d = {'video_filename': current_video,
         'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
         'json_prefix': os.path.join(proc_folder, 'processing_video_7_')}

    job_instance = SegmentComputationJob(**d)
    job_instance.process()

