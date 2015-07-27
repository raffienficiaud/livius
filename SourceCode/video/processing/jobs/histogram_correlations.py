"""
This file describes the computation of the Histogram correlations that are used to determine the segments of the video.
"""

from ..job import Job


import numpy as np
import cv2
import os
import json

from ....util.functor import Functor
from ....util.tools import sort_dictionary_by_integer_key


class HistogramCorrelationJob(Job):
    """
    Computation of the Histogram correlations.

    The inputs of this Job are (in this order):
        - Function :: (frame_index, area_name) -> Histogram
    - Number of Files

    The output of this Job is:
        A Function :: frame_index -> HistogramCorrelation
    """

    name = 'histogram_correlation'
    attributes_to_serialize = ['histogram_correlations']

    def __init__(self, *args, **kwargs):
        super(HistogramCorrelationJob, self).__init__(*args, **kwargs)

    def load_state(self):
        state = super(HistogramCorrelationJob, self).load_state()

        if state is None:
            return None

        correlations = state['histogram_correlations']
        correlations = sort_dictionary_by_integer_key(correlations)

        state['histogram_correlations'] = correlations

        return state

    def run(self, *args, **kwargs):

        # The first parent is the HistogramComputation
        get_histogram = args[0]

        # Second parent is the NumberOfFiles
        number_of_files = args[1]

        # init
        self.histogram_correlations = {}

        # @todo(Stephan): We need the number of vertical stripes
        nb_vertical_stripes = 10
        speaker_area_names = ['speaker_%.2d' % i for i in range(nb_vertical_stripes)]

        def get_speaker_histogram_plane(frame_index):
            return map(lambda area_name: get_histogram(area_name, frame_index), speaker_area_names)

        previous_slide_histogram = get_histogram('slides', 1)
        previous_speaker_histogram_plane = get_speaker_histogram_plane(1)

        # @todo(Stephan): Get the number of thumbnails
        for frame_index in range(2, number_of_files):

            slide_histogram = get_histogram('slides', frame_index)
            speaker_histogram_plane = get_speaker_histogram_plane(frame_index)

            if previous_slide_histogram is not None:
                self.histogram_correlations[frame_index] = \
                    cv2.compareHist(slide_histogram, previous_slide_histogram, cv2.cv.CV_COMP_CORREL)

            if previous_speaker_histogram_plane is not None:
                # @todo(Stephan): Compute the energy here? Maybe create a new Job and a common subclass (cf. GatherSelections)
                pass


            previous_slide_histogram = slide_histogram
            previous_speaker_histogram_plane = speaker_histogram_plane

        self.serialize_state()

    def get_outputs(self):
        super(HistogramCorrelationJob, self).get_outputs()

        if self.histogram_correlations is None:
            raise RuntimeError('The Correlations between the histograms have not been computed yet.')

        return Functor(self.histogram_correlations)


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
    current_video = os.path.join(video_folder, 'video_7.mp4')
    proc_folder = os.path.abspath(os.path.join(root_folder, 'tmp'))
    if not os.path.exists(proc_folder):
        os.makedirs(proc_folder)

    from .histogram_computation import HistogramsLABDiff, GatherSelections
    from .ffmpeg_to_thumbnails import FFMpegThumbnailsJob
    HistogramsLABDiff.add_parent(GatherSelections)
    HistogramsLABDiff.add_parent(FFMpegThumbnailsJob)

    HistogramCorrelationJob.add_parent(HistogramsLABDiff)

    d = {'video_filename': current_video,
         'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
         'json_prefix': os.path.join(proc_folder, 'processing_video_7_')}

    job_instance = HistogramCorrelationJob(**d)
    job_instance.process()


